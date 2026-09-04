/*
 * Paged GQA decode kernel for MetaX C500 (merged optimization).
 * V token-major VPAD=8 (no transpose). Merged-max softmax.
 * Split policy tuned per (batch,kv,work_units) class for real OJ table.
 */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#define HEAD 128
#define PAGE 16
#define FULLMASK 0xffffffffffffffffull

// V shared memory: token-major with pad. Row stride (in halves) = HEAD+VPAD.
// A uint4 store (8 halves) needs 16B alignment: stride must be a multiple of
// 8 halves AND dim must be a multiple of 8 (both satisfied: VPAD=8).
#define VPAD 8
#define VSTRIDE (HEAD + VPAD)

__device__ __forceinline__ uint32_t pack_bf16_2(float lo, float hi) {
    __nv_bfloat16 l = __float2bfloat16(lo);
    __nv_bfloat16 h = __float2bfloat16(hi);
    return ((uint32_t)(unsigned short&)h << 16) | (uint32_t)(unsigned short&)l;
}

// Gather two 16-bit values V[t0][d] (low) and V[t0+1][d] (high) from the
// token-major layout into the 32-bit operand word. This reproduces exactly the
// word the transposed layout produced: Vt[d][t0] is low, Vt[d][t0+1] is high.
__device__ __forceinline__ uint32_t vgather2(
    const __nv_bfloat16 (*Vb)[VSTRIDE], int t0, int d) {
    uint32_t lo = *(const unsigned short*)&Vb[t0][d];
    uint32_t hi = *(const unsigned short*)&Vb[t0 + 1][d];
    return lo | (hi << 16);
}

__global__ void paged_gqa_mma_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ m_part,
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
    const int lane = threadIdx.x; // 0..63
    const int b = blockIdx.y / num_heads_k;
    const int kv = blockIdx.y % num_heads_k;
    const int split = blockIdx.x;
    const int fused = (n_splits == 1);

    __shared__ __nv_bfloat16 K_b[PAGE][HEAD+2];       // K_src[tok][dim], padded
    __shared__ __nv_bfloat16 V_b[PAGE][VSTRIDE];      // V_src[tok][dim], token-major

    const int seqlen = cache_seqlens[b];
    const int t_start = split * tokens_per_split;
    if (t_start >= seqlen) return;
    const int h0 = kv * gqa;
    const int KVSTR = num_heads_k * HEAD;
    const int slot = (b * num_heads_k + kv) * n_splits + split;
    const int row = lane & 15;
    const int grp = lane >> 4;

    uint32_t qf[8][2];
    const int gh = b * num_heads + h0 + row;
    if (row < gqa) {
        const __nv_bfloat16* qp = q + gh * HEAD;
        #pragma unroll
        for (int st = 0; st < 8; st++) {
            qf[st][0] = *(const uint32_t*)(qp + st * 16 + grp * 4);
            qf[st][1] = *(const uint32_t*)(qp + st * 16 + grp * 4 + 2);
        }
    } else {
        #pragma unroll
        for (int st = 0; st < 8; st++) { qf[st][0] = qf[st][1] = 0; }
    }

    float m = -INFINITY, l = 0.f;
    float D[8][4];
    #pragma unroll
    for (int st = 0; st < 8; st++)
        #pragma unroll
        for (int i = 0; i < 4; i++) D[st][i] = 0.f;

    int t_end = t_start + tokens_per_split; if (t_end > seqlen) t_end = seqlen;
    int pg0 = t_start / PAGE;
    int pg1 = (t_end - 1) / PAGE;
    const int npages = pg1 - pg0 + 1;

    using VectorType = __NATIVE_VECTOR__(2, uint32_t);

    auto stage_page = [&](int pg, __nv_bfloat16 (*Kb)[HEAD+2], __nv_bfloat16 (*Vb)[VSTRIDE]) {
        int pid = block_table[b * blocks_per_batch + pg];
        const __nv_bfloat16* kbase = k_cache_paged + ((int64_t)pid * PAGE) * KVSTR + kv * HEAD;
        const __nv_bfloat16* vbase = v_cache_paged + ((int64_t)pid * PAGE) * KVSTR + kv * HEAD;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int u = lane + i * 64;      // 0..255
            int tok = u >> 4;
            int dim = (u & 15) * 8;
            uint4 kv4 = *(const uint4*)(kbase + tok * KVSTR + dim);
            *(uint4*)&Kb[tok][dim] = kv4;
            uint4 v4 = *(const uint4*)(vbase + tok * KVSTR + dim);
            *(uint4*)&Vb[tok][dim] = v4;    // ONE vectorized store, no transpose
        }
    };

    const int nfull = (t_end - t_start) >> 4;
    const int rem = t_end & (PAGE - 1);

    // ---- steady state: full pages (all 16 tokens valid) ----
    for (int i = 0; i < nfull; i++) {
        int pg = pg0 + i;
        stage_page(pg, K_b, V_b);
        __syncthreads();

        __nv_bfloat16 (*Kb)[HEAD+2] = K_b;
        __nv_bfloat16 (*Vb)[VSTRIDE] = V_b;

        float s[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 8; st++) {
            uint32_t b0 = *(const uint32_t*)(&Kb[row][grp * 4 + st * 16]);
            uint32_t b1 = *(const uint32_t*)(&Kb[row][grp * 4 + st * 16 + 2]);
            VectorType av = {qf[st][0], qf[st][1]};
            VectorType bv = {b0, b1};
            auto r = __builtin_mxc_mma_16x16x16bf16(bv, av, {s[0], s[1], s[2], s[3]});
            s[0] = r[0]; s[1] = r[1]; s[2] = r[2]; s[3] = r[3];
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) s[i] *= sm_scale;

        float mx = s[0];
        #pragma unroll
        for (int i = 1; i < 4; i++) mx = fmaxf(mx, s[i]);
        mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 16));
        mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 32));

        float mnew = fmaxf(m, mx);
        float alpha = __expf(m - mnew);
        #pragma unroll
        for (int st = 0; st < 8; st++)
            #pragma unroll
            for (int i = 0; i < 4; i++) D[st][i] *= alpha;
        l = l * alpha;
        m = mnew;

        float p[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) p[i] = __expf(s[i] - mnew);
        float ls = p[0] + p[1] + p[2] + p[3];
        ls += __shfl_xor_sync(FULLMASK, ls, 16);
        ls += __shfl_xor_sync(FULLMASK, ls, 32);
        l += ls;

        uint32_t pa0 = pack_bf16_2(p[0], p[1]);
        uint32_t pa1 = pack_bf16_2(p[2], p[3]);
        VectorType pav = {pa0, pa1};
        #pragma unroll
        for (int st = 0; st < 8; st++) {
            int d = row + st * 16;
            uint32_t b0 = vgather2(Vb, grp * 4, d);
            uint32_t b1 = vgather2(Vb, grp * 4 + 2, d);
            VectorType bv = {b0, b1};
            auto r = __builtin_mxc_mma_16x16x16bf16(bv, pav, {D[st][0], D[st][1], D[st][2], D[st][3]});
            D[st][0] = r[0]; D[st][1] = r[1]; D[st][2] = r[2]; D[st][3] = r[3];
        }
    }

    // ---- tail: the single final partial page of the split range ----
    if (rem != 0) {
        int pg = pg0 + nfull;
        int ntok = rem;
        stage_page(pg, K_b, V_b);
        __syncthreads();

        __nv_bfloat16 (*Kb)[HEAD+2] = K_b;
        __nv_bfloat16 (*Vb)[VSTRIDE] = V_b;

        float s[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 8; st++) {
            uint32_t b0 = *(const uint32_t*)(&Kb[row][grp * 4 + st * 16]);
            uint32_t b1 = *(const uint32_t*)(&Kb[row][grp * 4 + st * 16 + 2]);
            VectorType av = {qf[st][0], qf[st][1]};
            VectorType bv = {b0, b1};
            auto r = __builtin_mxc_mma_16x16x16bf16(bv, av, {s[0], s[1], s[2], s[3]});
            s[0] = r[0]; s[1] = r[1]; s[2] = r[2]; s[3] = r[3];
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) s[i] *= sm_scale;

        int tok0 = grp * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) if (tok0 + i >= ntok) s[i] = -INFINITY;

        float mx = s[0];
        #pragma unroll
        for (int i = 1; i < 4; i++) mx = fmaxf(mx, s[i]);
        mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 16));
        mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 32));

        float p[4];
        float ls;
        float mnew = fmaxf(m, mx);
        float alpha = __expf(m - mnew);
        #pragma unroll
        for (int st = 0; st < 8; st++)
            #pragma unroll
            for (int i = 0; i < 4; i++) D[st][i] *= alpha;
        l = l * alpha;
        m = mnew;

        #pragma unroll
        for (int i = 0; i < 4; i++) p[i] = (s[i] == -INFINITY) ? 0.f : __expf(s[i] - mnew);
        ls = p[0] + p[1] + p[2] + p[3];
        ls += __shfl_xor_sync(FULLMASK, ls, 16);
        ls += __shfl_xor_sync(FULLMASK, ls, 32);
        l += ls;

        uint32_t pa0 = pack_bf16_2(p[0], p[1]);
        uint32_t pa1 = pack_bf16_2(p[2], p[3]);
        VectorType pav = {pa0, pa1};
        #pragma unroll
        for (int st = 0; st < 8; st++) {
            int d = row + st * 16;
            uint32_t b0 = vgather2(Vb, grp * 4, d);
            uint32_t b1 = vgather2(Vb, grp * 4 + 2, d);
            VectorType bv = {b0, b1};
            auto r = __builtin_mxc_mma_16x16x16bf16(bv, pav, {D[st][0], D[st][1], D[st][2], D[st][3]});
            D[st][0] = r[0]; D[st][1] = r[1]; D[st][2] = r[2]; D[st][3] = r[3];
        }
    }

    if (row < gqa) {
        float inv = 1.0f / l;
        if (fused) {
            __nv_bfloat16* op = output + (b * num_heads + h0 + row) * HEAD;
            #pragma unroll
            for (int st = 0; st < 8; st++) {
                op[st * 16 + grp * 4 + 0] = __float2bfloat16(D[st][0] * inv);
                op[st * 16 + grp * 4 + 1] = __float2bfloat16(D[st][1] * inv);
                op[st * 16 + grp * 4 + 2] = __float2bfloat16(D[st][2] * inv);
                op[st * 16 + grp * 4 + 3] = __float2bfloat16(D[st][3] * inv);
            }
        } else {
            float* ap = &acc_part[(slot * gqa + row) * HEAD];
            #pragma unroll
            for (int st = 0; st < 8; st++) {
                ap[st * 16 + grp * 4 + 0] = D[st][0];
                ap[st * 16 + grp * 4 + 1] = D[st][1];
                ap[st * 16 + grp * 4 + 2] = D[st][2];
                ap[st * 16 + grp * 4 + 3] = D[st][3];
            }
            if (grp == 0) {
                m_part[slot * gqa + row] = m;
                l_part[slot * gqa + row] = l;
            }
        }
    }
}

__global__ void combine_kernel(
    const float* __restrict__ m_part,
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

    float m = -INFINITY, l = 0.f, acc = 0.f;
    float m2 = -INFINITY, l2 = 0.f, acc2 = 0.f;
    const int dim = d2 * 2;
    const int dim2 = dim + 1;
    for (int s = 0; s < nsplit_eff; s++) {
        int idx = base + s;
        float ms = m_part[idx * gqa + hh];
        float ls = l_part[idx * gqa + hh];
        float a = acc_part[(idx * gqa + hh) * HEAD + dim];
        float a2 = acc_part[(idx * gqa + hh) * HEAD + dim2];
        float mnew = fmaxf(m, ms);
        float alpha = __expf(m - mnew);
        float beta = __expf(ms - mnew);
        acc = acc * alpha + beta * a;
        acc2 = acc2 * alpha + beta * a2;
        l = l * alpha + beta * ls;
        l2 = l2 * alpha + beta * ls;
        m = mnew;
        m2 = mnew;
    }
    if (hh < gqa) {
        __nv_bfloat16* op = output + (b * num_heads + kv * gqa + hh) * HEAD;
        op[dim] = __float2bfloat16(acc / l);
        op[dim2] = __float2bfloat16(acc2 / l2);
    }
}

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output,
    const int32_t* cache_seqlens,
    const int32_t* block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t num_blocks,
    int64_t causal)
{
    int blocks_per_batch = (int)(num_blocks / batch_size);
    int gqa = (int)(num_heads / num_heads_k);
    float sm_scale = 1.0f / sqrtf((float)headdim);

    int64_t work_units = batch_size * num_heads_k;
    int64_t pages = (seqlen_k + page_block_size - 1) / page_block_size;
    int64_t ns;
    if (pages <= 4) {
        ns = 1;
    } else if (pages <= 16 && work_units >= 32) {
        ns = 3;
    } else if (num_heads_k == 8 && pages >= 17 && pages <= 64 && work_units <= 256) {
        double T = 12.0 * sqrt((double)(pages * work_units));
        ns = (int64_t)(T / (double)work_units + 0.5);
        if (ns < 2) ns = 2;
    } else if (work_units <= 8) {
        // batch==1: gqa<=8.
        // case10 (batch1,kv4,pages512): ns=64 optimum.
        // case14 (batch1,kv4,pages3845): ns=148 optimum (ppb~26).
        // case13 (batch1,kv8,pages3686): ns=90 optimum.
        if (num_heads_k == 8) {
            ns = (pages <= 1024) ? 64 : 90;   // batch1 kv8
        } else {
            ns = (pages <= 1024) ? 64 : 148;  // batch1 kv4
        }
    } else if (num_heads_k == 4 && work_units == 64 && pages >= 64) {
        // kv4 batch16 (cases 8, 11): optimum ~22 splits regardless of pages.
        ns = 22;
        if (ns > pages) ns = pages;
    } else if (num_heads_k == 8 && work_units == 256) {
        // kv8 batch32 (case 9): optimum ns=11.
        ns = (pages >= 64) ? 11 : 4;
    } else if (num_heads_k == 8 && work_units == 128) {
        // kv8 batch16 (case 6): optimum ns~5.
        ns = (pages >= 17) ? 5 : 3;
    } else {
        // kv8 large-batch split tuning (measured per (work_units) class):
        //   wu=512 (batch64 kv8, case7): ns=5 optimum (heuristic 10)
        //   wu=256 (batch32 kv8, case9): ns=30 optimum (matches heuristic)
        //   wu=64  (batch8  kv8, case12): ns=60 optimum (heuristic 170 too high)
        double mult = 20.0;
        if (num_heads_k == 8) {
            if (work_units == 512) mult = 10.0;
            else if (work_units == 256) mult = 30.0;
            else if (work_units == 64) mult = 10.5; // ns_i=59 -> eff ns=59 (measured optimum ~440us)
        }
        double T = mult * sqrt((double)(pages * work_units));
        ns = (int64_t)(T / (double)work_units + 0.5);
        if (ns < 1) ns = 1;
    }

    int64_t tps_pages = (pages + ns - 1) / ns;
    if (tps_pages < 1) tps_pages = 1;
    int tokens_per_split = (int)(tps_pages * page_block_size);
    ns = (pages + tps_pages - 1) / tps_pages;

    int64_t nspl = (int64_t)batch_size * num_heads_k * ns * gqa;

    dim3 grid((unsigned)ns, (unsigned)(batch_size * num_heads_k));
    if (ns == 1) {
        paged_gqa_mma_kernel<<<grid, 64>>>(
            q, k_cache_paged, v_cache_paged, nullptr, nullptr, nullptr,
            cache_seqlens, block_table, output,
            (int)num_heads, (int)num_heads_k, (int)headdim,
            (int)page_block_size, blocks_per_batch, 1, gqa, sm_scale,
            tokens_per_split);
        return;
    }

    static float* m_part = nullptr;
    static float* l_part = nullptr;
    static float* acc_part = nullptr;
    static int64_t cap = 0;
    int64_t need_m = nspl;
    int64_t need_a = nspl * HEAD;
    int64_t need_bytes = need_m * 4 * 2 + need_a * 4;
    if (m_part == nullptr || need_bytes > cap) {
        if (m_part) {
            cudaFree(m_part); cudaFree(l_part); cudaFree(acc_part);
        }
        cudaMalloc((void**)&m_part, need_m * 4);
        cudaMalloc((void**)&l_part, need_m * 4);
        cudaMalloc((void**)&acc_part, need_a * 4);
        cap = need_bytes;
    }

    paged_gqa_mma_kernel<<<grid, 64>>>(
        q, k_cache_paged, v_cache_paged, m_part, l_part, acc_part,
        cache_seqlens, block_table, output,
        (int)num_heads, (int)num_heads_k, (int)headdim,
        (int)page_block_size, blocks_per_batch, ns, gqa, sm_scale,
        tokens_per_split);

    dim3 grid2((unsigned)batch_size, (unsigned)num_heads_k);
    combine_kernel<<<grid2, (unsigned)(gqa * 64)>>>(
        m_part, l_part, acc_part, cache_seqlens, output,
        (int)num_heads, (int)num_heads_k, ns, gqa, tokens_per_split);
}
