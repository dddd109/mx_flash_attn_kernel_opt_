/* 2-kv-per-CTA MMA kernel: reads 2 adjacent kv slices contiguously (512B/token)
 * to improve DRAM burst efficiency vs the 1-kv strided (256B/2KB) pattern.
 * Proof-of-concept for the contiguous-read hypothesis within the MMA design. */
#include <stdint.h>
#include <stdlib.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#define HEAD 128
#define PAGE 16
#define FULLMASK 0xffffffffffffffffull
#define NKV2 2

__device__ __forceinline__ uint32_t pack_bf16_2(float lo, float hi) {
    __nv_bfloat16 l = __float2bfloat16(lo);
    __nv_bfloat16 h = __float2bfloat16(hi);
    return ((uint32_t)(unsigned short&)h << 16) | (uint32_t)(unsigned short&)l;
}
__device__ __forceinline__ uint32_t vgather2(
    const __nv_bfloat16 (*Vb)[HEAD+8], int t0, int d) {
    uint32_t lo = *(const unsigned short*)&Vb[t0][d];
    uint32_t hi = *(const unsigned short*)&Vb[t0 + 1][d];
    return lo | (hi << 16);
}

__global__ void paged_gqa_mma2kv_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ l_part,
    float* __restrict__ acc_part,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    __nv_bfloat16* __restrict__ output,
    int num_heads, int num_heads_k, int headdim,
    int page_block_size, int blocks_per_batch,
    int n_splits, int gqa, float sm_scale,
    int tokens_per_split)
{
    const int lane = threadIdx.x;             // 0..63
    const int b = blockIdx.y / (num_heads_k / NKV2);
    const int kvg = blockIdx.y % (num_heads_k / NKV2);  // which kv-pair
    const int kv0 = kvg * NKV2;
    const int split = blockIdx.x;
    const int fused = (n_splits == 1);

    __shared__ __nv_bfloat16 K_b[NKV2][PAGE][HEAD+4];
    __shared__ __nv_bfloat16 V_b[NKV2][PAGE][HEAD+8];

    const int seqlen = cache_seqlens[b];
    const int t_start = split * tokens_per_split;
    if (t_start >= seqlen) return;
    const int KVSTR = num_heads_k * HEAD;     // halves per token (ALL kv)
    const int PGSZ = PAGE * KVSTR;            // halves per physical page

    const int row = lane & 15;
    const int grp = lane >> 4;

    int t_end = t_start + tokens_per_split; if (t_end > seqlen) t_end = seqlen;
    int pg0 = t_start / PAGE;
    int pg1 = (t_end - 1) / PAGE;
    const int nfull = (t_end - t_start) >> 4;
    const int rem = t_end & (PAGE - 1);

    using VectorType = __NATIVE_VECTOR__(2, uint32_t);

    /* q fragment cache: for both kv groups. head row index within group.
     * For kv head = kv0+k, its query heads are (kv0+k)*gqa + 0..gqa-1.
     * Thread row -> local head; but two kv groups -> up to 2*gqa <= 16 rows for gqa<=8. */
    /* We cache q for the 16 MMA rows = 2 kv groups x gqa each (gqa<=8 so 2*gqa<=16). */
    uint32_t qf[NKV2][8][2];
    const int h0_global = kv0 * gqa;
    for (int k = 0; k < NKV2; k++) {
        int local_row = row;   // row in 0..15 = which head of the kv's gqa-group? gqa may be 4/8
        /* row<gqa means head (kv0+k)*gqa + row is real for THIS kv; else zero. */
        int gh = b * num_heads + (kv0 + k) * gqa + row;
        if (row < gqa) {
            const __nv_bfloat16* qp = q + gh * HEAD;
            #pragma unroll
            for (int st = 0; st < 8; st++) {
                qf[k][st][0] = *(const uint32_t*)(qp + st * 16 + grp * 4);
                qf[k][st][1] = *(const uint32_t*)(qp + st * 16 + grp * 4 + 2);
            }
        } else {
            #pragma unroll
            for (int st = 0; st < 8; st++) { qf[k][st][0] = qf[k][st][1] = 0; }
        }
    }

    /* combined accumulators per kv group: D[k][8][4] */
    float D[NKV2][8][4];
    #pragma unroll
    for (int k = 0; k < NKV2; k++)
        #pragma unroll
        for (int st = 0; st < 8; st++)
            #pragma unroll
            for (int i = 0; i < 4; i++) D[k][st][i] = 0.f;
    float l[NKV2];
    l[0] = l[1] = 0.f;

    /* stage one page's 2-kv contiguous K+V block. uint4 units per page = 
     * 16 tok * 2 kv * 128 halve /8 = 512 for K and 512 for V.
     * 64 lanes -> 8 iters cover 512. */
    auto stage2 = [&](int pid, __nv_bfloat16 (*Kb)[PAGE][HEAD+4], __nv_bfloat16 (*Vb)[PAGE][HEAD+8]) {
        const __nv_bfloat16* kbase = k_cache_paged + (int64_t)pid * PGSZ + kv0 * HEAD;
        const __nv_bfloat16* vbase = v_cache_paged + (int64_t)pid * PGSZ + kv0 * HEAD;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int u = lane + i * 64;              // 0..511 uint4 (K)
            int tok = u / 32;                   // 16 uint4 per (tok,kv)?? no: per (tok,kv) 16 uint4
            /* u over 512 = 16 tok x 2 kv x 16. tok=u/32? 16 uint4 per kv-token -> per token 32.
               u: tok=u>>5 (32/tok), r=u&31, kk=r>>4, d16=r&15. halve offset = u*8 within the 2kv slice. */
            int hx = u * 8;                      // halve index within [16 tok][2 kv][128]
            int tt = hx / (NKV2 * HEAD);         // token
            int rr = hx % (NKV2 * HEAD);
            int kk = rr / HEAD;
            int dd = rr % HEAD;
            /* read K: global halve = tt*KVSTR + kv0*HEAD + kk*HEAD + dd  -- but u*8 stride == that */
            uint4 kv4 = *(const uint4*)(kbase + tt * KVSTR + kk * HEAD + dd);
            *(uint4*)&Kb[kk][tt][dd] = kv4;
            uint4 v4 = *(const uint4*)(vbase + tt * KVSTR + kk * HEAD + dd);
            *(uint4*)&Vb[kk][tt][dd] = v4;
        }
    };

    /* full-page steady loop */
    int cur_pid = block_table[b * blocks_per_batch + pg0];
    for (int i = 0; i < nfull; i++) {
        int pg = pg0 + i;
        int nxt_pid = (i + 1 < nfull) ? block_table[b * blocks_per_batch + pg0 + i + 1] : cur_pid;
        stage2(cur_pid, K_b, V_b);
        __syncwarp();
        cur_pid = nxt_pid;

        for (int k = 0; k < NKV2; k++) {
            __nv_bfloat16 (*Kb)[HEAD+4] = K_b[k];
            __nv_bfloat16 (*Vb)[HEAD+8] = V_b[k];
            float s[4] = {0,0,0,0};
            #pragma unroll
            for (int st = 0; st < 8; st++) {
                uint64_t bk = *(const uint64_t*)(&Kb[row][grp*4+st*16]);
                uint32_t b0=(uint32_t)(bk&0xffffffffu), b1=(uint32_t)(bk>>32);
                VectorType av={qf[k][st][0],qf[k][st][1]}, bv={b0,b1};
                auto r=__builtin_mxc_mma_16x16x16bf16(bv,av,{s[0],s[1],s[2],s[3]});
                s[0]=r[0];s[1]=r[1];s[2]=r[2];s[3]=r[3];
            }
            #pragma unroll
            for (int i2=0;i2<4;i2++) s[i2]*=sm_scale;
            float p[4];
            #pragma unroll
            for (int i2=0;i2<4;i2++) p[i2]=__expf(s[i2]);
            float ls=(p[0]+p[1])+(p[2]+p[3]);
            ls+=__shfl_xor_sync(FULLMASK,ls,16);
            ls+=__shfl_xor_sync(FULLMASK,ls,32);
            l[k]+=ls;
            uint32_t pa0,pa1;
            *(__NATIVE_VECTOR__(2,unsigned short)*)&pa0=__builtin_mxc_cvt_pk_f32tobf16({p[0],p[1]});
            *(__NATIVE_VECTOR__(2,unsigned short)*)&pa1=__builtin_mxc_cvt_pk_f32tobf16({p[2],p[3]});
            VectorType pav={pa0,pa1};
            #pragma unroll
            for (int st=0;st<8;st++){
                int d=row+st*16;
                uint32_t b0=vgather2(Vb,grp*4,d);
                uint32_t b1=vgather2(Vb,grp*4+2,d);
                VectorType bv={b0,b1};
                auto r=__builtin_mxc_mma_16x16x16bf16(bv,pav,{D[k][st][0],D[k][st][1],D[k][st][2],D[k][st][3]});
                D[k][st][0]=r[0];D[k][st][1]=r[1];D[k][st][2]=r[2];D[k][st][3]=r[3];
            }
        }
    }

    /* tail partial page */
    if (rem != 0) {
        int pg = pg0 + nfull;
        int ntok = rem;
        int pid = block_table[b * blocks_per_batch + pg];
        stage2(pid, K_b, V_b);
        __syncwarp();
        for (int k = 0; k < NKV2; k++) {
            __nv_bfloat16 (*Kb)[HEAD+4] = K_b[k];
            __nv_bfloat16 (*Vb)[HEAD+8] = V_b[k];
            float s[4]={0,0,0,0};
            #pragma unroll
            for (int st=0;st<8;st++){
                uint64_t bk=*(const uint64_t*)(&Kb[row][grp*4+st*16]);
                uint32_t b0=(uint32_t)(bk&0xffffffffu),b1=(uint32_t)(bk>>32);
                VectorType av={qf[k][st][0],qf[k][st][1]},bv={b0,b1};
                auto r=__builtin_mxc_mma_16x16x16bf16(bv,av,{s[0],s[1],s[2],s[3]});
                s[0]=r[0];s[1]=r[1];s[2]=r[2];s[3]=r[3];
            }
            #pragma unroll
            for (int i2=0;i2<4;i2++) s[i2]*=sm_scale;
            int tok0=grp*4;
            #pragma unroll
            for (int i2=0;i2<4;i2++) if(tok0+i2>=ntok) s[i2]=-INFINITY;
            float p[4];
            #pragma unroll
            for (int i2=0;i2<4;i2++) p[i2]=__expf(s[i2]);
            float ls=(p[0]+p[1])+(p[2]+p[3]);
            ls+=__shfl_xor_sync(FULLMASK,ls,16);
            ls+=__shfl_xor_sync(FULLMASK,ls,32);
            l[k]+=ls;
            uint32_t pa0,pa1;
            *(__NATIVE_VECTOR__(2,unsigned short)*)&pa0=__builtin_mxc_cvt_pk_f32tobf16({p[0],p[1]});
            *(__NATIVE_VECTOR__(2,unsigned short)*)&pa1=__builtin_mxc_cvt_pk_f32tobf16({p[2],p[3]});
            VectorType pav={pa0,pa1};
            #pragma unroll
            for (int st=0;st<8;st++){
                int d=row+st*16;
                uint32_t b0=vgather2(Vb,grp*4,d);
                uint32_t b1=vgather2(Vb,grp*4+2,d);
                VectorType bv={b0,b1};
                auto r=__builtin_mxc_mma_16x16x16bf16(bv,pav,{D[k][st][0],D[k][st][1],D[k][st][2],D[k][st][3]});
                D[k][st][0]=r[0];D[k][st][1]=r[1];D[k][st][2]=r[2];D[k][st][3]=r[3];
            }
        }
    }

    /* write partials: each kv group's gqa rows (row<gqa) at slot (b,kv0+k) */
    for (int k = 0; k < NKV2; k++) {
        if (row < gqa) {
            int slot = (b * num_heads_k + kv0 + k) * n_splits + split;
            if (fused) {
                float inv = 1.0f / l[k];
                __nv_bfloat16* op = output + (b * num_heads + (kv0+k)*gqa + row) * HEAD;
                #pragma unroll
                for (int st=0;st<8;st++){
                    op[st*16+grp*4+0]=__float2bfloat16(D[k][st][0]*inv);
                    op[st*16+grp*4+1]=__float2bfloat16(D[k][st][1]*inv);
                    op[st*16+grp*4+2]=__float2bfloat16(D[k][st][2]*inv);
                    op[st*16+grp*4+3]=__float2bfloat16(D[k][st][3]*inv);
                }
            } else {
                float* ap=&acc_part[(slot*gqa+row)*HEAD];
                #pragma unroll
                for (int st=0;st<8;st++){
                    ap[st*16+grp*4+0]=D[k][st][0];
                    ap[st*16+grp*4+1]=D[k][st][1];
                    ap[st*16+grp*4+2]=D[k][st][2];
                    ap[st*16+grp*4+3]=D[k][st][3];
                }
                if (grp==0) l_part[slot*gqa+row]=l[k];
            }
        }
    }
}

__global__ void combine_kernel(
    const float* __restrict__ l_part,
    const float* __restrict__ acc_part,
    const int32_t* __restrict__ cache_seqlens,
    __nv_bfloat16* __restrict__ output,
    int num_heads, int num_heads_k, int n_splits, int gqa,
    int tokens_per_split)
{
    const int b = blockIdx.x;
    const int kv = blockIdx.y;
    const int tid = threadIdx.x;
    const int hh = tid >> 6;
    const int d2 = tid & 63;
    const int base = (b * num_heads_k + kv) * n_splits;
    const int seqlen = cache_seqlens[b];
    int nsplit_eff = (seqlen + tokens_per_split - 1) / tokens_per_split;
    if (nsplit_eff > n_splits) nsplit_eff = n_splits;
    float l = 0.f, acc = 0.f, acc2 = 0.f;
    const int dim = d2 * 2;
    const int dim2 = dim + 1;
    for (int s = 0; s < nsplit_eff; s++) {
        int idx = base + s;
        float ls = l_part[idx * gqa + hh];
        const float* ap = &acc_part[(idx * gqa + hh) * HEAD + dim];
        acc += ap[0]; acc2 += ap[1]; l += ls;
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
        ns = (int64_t)(12.0 * sqrt((double)(pages * wu)) / (double)wu + 0.5); if (ns<2) ns=2;
    } else if (wu <= 8) {
        if (num_heads_k == 8) ns = (pages <= 1024) ? 64 : 90;
        else ns = (pages <= 1024) ? 64 : 148;
    } else if (num_heads_k == 4 && wu == 64 && pages >= 64) { ns = 22; if (ns>pages) ns=pages; }
    else if (num_heads_k == 8 && wu == 256) ns = (pages >= 64) ? 11 : 4;
    else if (num_heads_k == 8 && wu == 128) ns = (pages >= 17) ? 5 : 3;
    else {
        double mult = 20.0;
        if (num_heads_k == 8) { if (wu==512) mult=10; else if (wu==256) mult=30; else if (wu==64) mult=10.5; }
        ns = (int64_t)(mult * sqrt((double)(pages*wu)) / (double)wu + 0.5); if (ns<1) ns=1;
    }
    int64_t tps_pages = (pages + ns - 1) / ns; if (tps_pages<1) tps_pages=1;
    int tokens_per_split = (int)(tps_pages * page_block_size);
    ns = (pages + tps_pages - 1) / tps_pages;
    int64_t nkvgroups = batch_size * (num_heads_k / 2);
    int64_t nspl = batch_size * num_heads_k * ns * gqa;   /* slots for ALL kv heads */
    (void)nkvgroups;
    dim3 grid((unsigned)ns, (unsigned)nkvgroups);
    if (ns == 1) {
        paged_gqa_mma2kv_kernel<<<grid,64>>>(q,k_cache_paged,v_cache_paged,nullptr,nullptr,
            cache_seqlens,block_table,output,(int)num_heads,(int)num_heads_k,(int)headdim,
            (int)page_block_size,blocks_per_batch,1,gqa,sm_scale,tokens_per_split);
        return;
    }
    static float* lp=nullptr; static float* ap=nullptr; static int64_t cap=0;
    int64_t need_a = nspl*HEAD; int64_t need_b = nspl*4 + need_a*4;
    if (lp==nullptr || need_b>cap){ if(lp){cudaFree(lp);cudaFree(ap);} cudaMalloc((void**)&lp,nspl*4); cudaMalloc((void**)&ap,need_a*4); cap=need_b; }
    paged_gqa_mma2kv_kernel<<<grid,64>>>(q,k_cache_paged,v_cache_paged,lp,ap,
        cache_seqlens,block_table,output,(int)num_heads,(int)num_heads_k,(int)headdim,
        (int)page_block_size,blocks_per_batch,(int)ns,gqa,sm_scale,tokens_per_split);
    dim3 grid2((unsigned)batch_size,(unsigned)num_heads_k);
    combine_kernel<<<grid2,(unsigned)(gqa*64)>>>(lp,ap,cache_seqlens,output,
        (int)num_heads,(int)num_heads_k,(int)ns,gqa,tokens_per_split);
}
