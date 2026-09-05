/*
 * FMA decode kernel (MetaX mcflashinfer-style, adapted for our paged GQA case).
 *
 * Structure (from MetaX's SingleDecodeWithKVCacheKernel, which uses CUDA-core
 * FMA not the 16x16 MMA):
 *   - Block threads = (bdx=16, bdy=gqa): bdx spans the 128-dim head (vec_size=8
 *     dims/thread via 2-wide FMA), bdy = query heads sharing this kv head.
 *   - One block handles one (batch, kv_head); the K/V for a page tile is read
 *     into smem ONCE and shared by all bdy heads.
 *   - QK: each (tx,ty) thread dot-products its 8-dim q slice against the same
 *     8-dim K slice of each token, then a shuffle-tree (xor 1,2,4,8) over the 16
 *     bdx threads yields the full score[ty-head][token]. Softmax per head over
 *     the running range; PV accumulates o[head][tx*8..+7] += p*V.
 *
 * This is the reference-style approach: no MMA row waste (each of the 16 bdx
 * threads does real work on 8 dims), FMA on CUDA cores which MetaX proves works
 * for decode.
 */
#include <stdint.h>
#include <stdlib.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#define HEAD 128
#define PAGE 16
#define FULLMASK 0xffffffffffffffffull
#define BDX 16            /* 128 / 8: threads spanning the head dim */
#define VEC 8             /* dims per thread */

__global__ void paged_gqa_fma_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ l_part,       /* [ns*32] */
    float* __restrict__ acc_part,     /* [ns*32*HEAD] */
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    __nv_bfloat16* __restrict__ output,
    int num_heads, int num_heads_k, int headdim,
    int page_block_size, int blocks_per_batch,
    int n_splits, int gqa, float sm_scale,
    int tokens_per_split)
{
    const int tx = threadIdx.x;            /* 0..15  = dim group (8 dims each) */
    const int ty = threadIdx.y;            /* 0..gqa-1 = query head within kv */
    const int b = blockIdx.y / num_heads_k;
    const int kv = blockIdx.y % num_heads_k;
    const int split = blockIdx.x;
    const int lane1d = ty * BDX + tx;      /* 1-D thread id for shuffle math */

    const int seqlen = cache_seqlens[b];
    const int t_start = split * tokens_per_split;
    if (t_start >= seqlen) return;
    int t_end = t_start + tokens_per_split; if (t_end > seqlen) t_end = seqlen;

    const int KVSTR = num_heads_k * HEAD;   /* halves per token */
    const int PGSZ = PAGE * KVSTR;          /* halves per page */
    const int h = kv * gqa + ty;            /* global query head */

    /* q slice for this (head, dim-group): 8 bf16 -> 8 float regs */
    float qv[VEC];
    {
        const __nv_bfloat16* qp = q + ((int64_t)b * num_heads + h) * HEAD + tx * VEC;
        #pragma unroll
        for (int i = 0; i < VEC; i++) qv[i] = (float)qp[i];
    }

    /* K/V page tile in smem: [PAGE][HEAD] bf16 each. */
    __shared__ __nv_bfloat16 Ks[PAGE * HEAD];
    __shared__ __nv_bfloat16 Vs[PAGE * HEAD];

    float o[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) o[i] = 0.f;
    float lsum = 0.f;   /* absolute-domain softmax denominator */

    const int nfull = (t_end - t_start) >> 4;
    const int rem = t_end & (PAGE - 1);

    /* stage a page: Ks[tok][dim], Vs[tok][dim]. 64*? threads: block = bdx*bdy
     * = 16*gqa threads (64 for gqa4, 128 for gqa8). Load PAGE*HEAD = 2048
     * halves per K/V. threads * (2048*2 / (bdx*bdy)) ... each thread loads
     * 2048*2/(bdx*bdy) halves. For gqa4 (64 thr): 64 halves = 4 uint4? each
     * 16B=8 halves -> 2048 halves/8 = 256 uint4 for K, 256 for V = 512.
     * 64 thr -> 8 uint4 each. gqa8 (128 thr) -> 4 each. Use dynamic loop. */
    auto stage_page = [&](int pid) {
        const __nv_bfloat16* kbase = k_cache_paged + (int64_t)pid * PGSZ + kv * HEAD;
        const __nv_bfloat16* vbase = v_cache_paged + (int64_t)pid * PGSZ + kv * HEAD;
        const int nthr = BDX * gqa;
        const int total_uint4 = (PAGE * HEAD * 2) / 8;   /* K+V uint4 = 512 */
        for (int u = ty * BDX + tx; u < total_uint4; u += nthr) {
            /* u in 0..511: 0..255 = K, 256..511 = V; within, element index in halves
             * = u*8 (0..2047) mapping to tok=u/16? 2048 halves/page, 128/tok ->
             * 16 tok. u*8 halve = (u/16)*128 + (u%16)*8 for K (256 uint4 = 16 tok
             * * 16 uint4). So for K: tok=u/16, d8=u%16, dim=d8*8. */
            if (u < 256) {
                int tok = u >> 4, d8 = u & 15, dim = d8 * 8;
                *(uint4*)&Ks[tok * HEAD + dim] = *(const uint4*)(kbase + tok * KVSTR + dim);
            } else {
                int w = u - 256;
                int tok = w >> 4, d8 = w & 15, dim = d8 * 8;
                *(uint4*)&Vs[tok * HEAD + dim] = *(const uint4*)(vbase + tok * KVSTR + dim);
            }
        }
    };

    /* full pages */
    for (int i = 0; i < nfull; i++) {
        int pid = block_table[b * blocks_per_batch + (t_start / PAGE) + i];
        stage_page(pid);
        __syncthreads();

        /* QK for each of the 16 tokens: partial dot over my 8 dims, then
         * tree-reduce across the 16 tx (they are BDX apart in lane1d... ty fixed,
         * tx 0..15 -> lane1d = ty*16+tx, consecutive tx = consecutive lane1d). */
        float sc[PAGE];
        #pragma unroll
        for (int t = 0; t < PAGE; t++) {
            const __nv_bfloat16* krow = Ks + t * HEAD + tx * VEC;
            float s = 0.f;
            #pragma unroll
            for (int d = 0; d < VEC; d += 2)
                s = fmaf(qv[d], (float)krow[d], fmaf(qv[d+1], (float)krow[d+1], s));
            /* reduce over tx (0..15): xor 1,2,4,8 within the ty group */
            s += __shfl_xor_sync(FULLMASK, s, 1);
            s += __shfl_xor_sync(FULLMASK, s, 2);
            s += __shfl_xor_sync(FULLMASK, s, 4);
            s += __shfl_xor_sync(FULLMASK, s, 8);
            sc[t] = s;
        }
        /* softmax absolute-domain; p per token, shared across the 16 tx. */
        float p[PAGE];
        #pragma unroll
        for (int t = 0; t < PAGE; t++) p[t] = __expf(sc[t] * sm_scale);
        /* lsum and PV: only ONE tx per (head) should own lsum... but p is the same
         * for all tx of a head. Accumulate o per tx and lsum per ty. Use tx==0 for
         * lsum to avoid 16x double count. */
        if (tx == 0) {
            #pragma unroll
            for (int t = 0; t < PAGE; t++) lsum += p[t];
        }
        /* PV: o[dim] += sum_t p[t]*V[t][dim]; tx owns dims tx*8..+7 */
        #pragma unroll
        for (int t = 0; t < PAGE; t++) {
            const __nv_bfloat16* vrow = Vs + t * HEAD + tx * VEC;
            float pt = p[t];
            #pragma unroll
            for (int d = 0; d < VEC; d++) o[d] = fmaf(pt, (float)vrow[d], o[d]);
        }
        __syncthreads();
    }

    /* tail partial page */
    if (rem != 0) {
        int pid = block_table[b * blocks_per_batch + (t_start / PAGE) + nfull];
        stage_page(pid);
        __syncthreads();
        float sc[PAGE];
        #pragma unroll
        for (int t = 0; t < PAGE; t++) {
            const __nv_bfloat16* krow = Ks + t * HEAD + tx * VEC;
            float s = 0.f;
            #pragma unroll
            for (int d = 0; d < VEC; d += 2)
                s = fmaf(qv[d], (float)krow[d], fmaf(qv[d+1], (float)krow[d+1], s));
            s += __shfl_xor_sync(FULLMASK, s, 1);
            s += __shfl_xor_sync(FULLMASK, s, 2);
            s += __shfl_xor_sync(FULLMASK, s, 4);
            s += __shfl_xor_sync(FULLMASK, s, 8);
            sc[t] = (t < rem) ? s : -INFINITY;
        }
        float p[PAGE];
        #pragma unroll
        for (int t = 0; t < PAGE; t++) p[t] = __expf(sc[t] * sm_scale);
        if (tx == 0) {
            #pragma unroll
            for (int t = 0; t < PAGE; t++) if (t < rem) lsum += p[t];
        }
        #pragma unroll
        for (int t = 0; t < PAGE; t++) {
            const __nv_bfloat16* vrow = Vs + t * HEAD + tx * VEC;
            float pt = (t < rem) ? p[t] : 0.f;
            #pragma unroll
            for (int d = 0; d < VEC; d++) o[d] = fmaf(pt, (float)vrow[d], o[d]);
        }
        __syncthreads();
    }

    /* finalize: divide o by lsum. lsum lives only on tx==0 threads; share via smem. */
    __shared__ float lsm[32];
    if (tx == 0) lsm[h] = lsum;
    __syncthreads();
    float invl = 1.0f / lsm[h];

    int slot = (b * num_heads_k + kv) * n_splits + split;
    if (n_splits == 1) {
        __nv_bfloat16* op = output + ((int64_t)b * num_heads + h) * HEAD + tx * VEC;
        #pragma unroll
        for (int d = 0; d < VEC; d++) op[d] = __float2bfloat16(o[d] * invl);
    } else {
        float* ap = acc_part + ((int64_t)(slot * gqa + ty)) * HEAD + tx * VEC;
        #pragma unroll
        for (int d = 0; d < VEC; d++) ap[d] = o[d];
        if (tx == 0) l_part[slot * gqa + ty] = lsum;
    }
}

__global__ void combine_kernel(
    const float* __restrict__ l_part, const float* __restrict__ acc_part,
    const int32_t* __restrict__ cache_seqlens, __nv_bfloat16* __restrict__ output,
    int num_heads, int num_heads_k, int n_splits, int gqa, int tokens_per_split)
{
    const int b = blockIdx.x; const int kv = blockIdx.y; const int tid = threadIdx.x;
    const int hh = tid >> 6; const int d2 = tid & 63;
    const int base = (b * num_heads_k + kv) * n_splits;
    const int seqlen = cache_seqlens[b];
    int nsplit_eff = (seqlen + tokens_per_split - 1) / tokens_per_split;
    if (nsplit_eff > n_splits) nsplit_eff = n_splits;
    float l = 0.f, acc = 0.f, acc2 = 0.f;
    const int dim = d2 * 2; const int dim2 = dim + 1;
    for (int s2 = 0; s2 < nsplit_eff; s2++) {
        int idx = base + s2;
        l += l_part[idx * gqa + hh];
        const float* ap = &acc_part[(idx * gqa + hh) * HEAD + dim];
        acc += ap[0]; acc2 += ap[1];
    }
    if (hh < gqa) {
        __nv_bfloat16* op = output + (b * num_heads + kv * gqa + hh) * HEAD;
        op[dim] = __float2bfloat16(acc / l);
        op[dim2] = __float2bfloat16(acc2 / l);
    }
}

extern "C" void run_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged, __nv_bfloat16* output,
    const int32_t* cache_seqlens, const int32_t* block_table,
    int64_t batch_size, int64_t seqlen_k, int64_t seqlen_q,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim,
    int64_t page_block_size, int64_t num_blocks, int64_t causal)
{
    int blocks_per_batch = (int)(num_blocks / batch_size);
    int gqa = (int)(num_heads / num_heads_k);
    float sm_scale = 1.0f / sqrtf((float)headdim);
    int64_t wu = batch_size * num_heads_k;
    int64_t pages = (seqlen_k + page_block_size - 1) / page_block_size;
    int64_t ns;
    if (pages <= 4) ns = 1;
    else if (pages <= 16 && wu >= 32) ns = 3;
    else if (num_heads_k == 8 && pages >= 17 && pages <= 64 && wu <= 256) {
        ns = (int64_t)(12.0 * sqrt((double)(pages*wu))/(double)wu + 0.5); if (ns<2) ns=2;
    } else if (wu <= 8) {
        if (num_heads_k == 8) ns = (pages <= 1024) ? 64 : 90;
        else ns = (pages <= 1024) ? 64 : 100;   /* case14 tuned ns=100 */
    } else if (num_heads_k == 4 && wu == 64 && pages >= 64) { ns = 22; if (ns>pages) ns=pages; }
    else if (num_heads_k == 8 && wu == 256) ns = (pages >= 64) ? 11 : 4;
    else if (num_heads_k == 8 && wu == 128) ns = (pages >= 17) ? 5 : 3;
    else {
        double mult = 20.0;
        if (num_heads_k == 8) { if (wu==512) mult=10; else if (wu==256) mult=30; else if (wu==64) mult=10.5; }
        ns = (int64_t)(mult*sqrt((double)(pages*wu))/(double)wu + 0.5); if (ns<1) ns=1;
    }
    int64_t tps_pages = (pages + ns - 1) / ns; if (tps_pages<1) tps_pages=1;
    int tokens_per_split = (int)(tps_pages * page_block_size);
    ns = (pages + tps_pages - 1) / tps_pages;
    int64_t nspl = (int64_t)batch_size * num_heads_k * ns * gqa;
    dim3 grid((unsigned)ns, (unsigned)(batch_size * num_heads_k));
    if (ns == 1) {
        paged_gqa_fma_kernel<<<grid, dim3(16, (unsigned)gqa)>>>(
            q,k_cache_paged,v_cache_paged,nullptr,nullptr,cache_seqlens,block_table,output,
            (int)num_heads,(int)num_heads_k,(int)headdim,(int)page_block_size,blocks_per_batch,
            1,gqa,sm_scale,tokens_per_split);
        return;
    }
    static float* lp=nullptr; static float* ap=nullptr; static int64_t cap=0;
    int64_t need_a = nspl*HEAD; int64_t need_b = nspl*4 + need_a*4;
    if (lp==nullptr || need_b>cap){ if(lp){cudaFree(lp);cudaFree(ap);} cudaMalloc((void**)&lp,nspl*4); cudaMalloc((void**)&ap,need_a*4); cap=need_b; }
    paged_gqa_fma_kernel<<<grid, dim3(16, (unsigned)gqa)>>>(
        q,k_cache_paged,v_cache_paged,lp,ap,cache_seqlens,block_table,output,
        (int)num_heads,(int)num_heads_k,(int)headdim,(int)page_block_size,blocks_per_batch,
        (int)ns,gqa,sm_scale,tokens_per_split);
    dim3 grid2((unsigned)batch_size,(unsigned)num_heads_k);
    combine_kernel<<<grid2,(unsigned)(gqa*64)>>>(lp,ap,cache_seqlens,output,
        (int)num_heads,(int)num_heads_k,(int)ns,gqa,tokens_per_split);
}
