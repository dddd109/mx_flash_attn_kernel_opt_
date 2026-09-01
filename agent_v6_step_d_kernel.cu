#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "/opt/maca/mxgpu_llvm/lib/clang/19/include/__clang_maca_mma_functions.h"

#define HEAD 128
#define PAGE 16
#define FULLMASK 0xffffffffffffffffull

__device__ __forceinline__ uint32_t pack_bf16_2(float lo, float hi) {
    __nv_bfloat16 l = __float2bfloat16(lo);
    __nv_bfloat16 h = __float2bfloat16(hi);
    return ((uint32_t)(unsigned short&)h << 16) | (uint32_t)(unsigned short&)l;
}

// MMA-based paged GQA split kernel.
// Each block: 64 threads (= one 16x16 MMA wave), handles (batch, kv_head) + split.
// M dimension = 16 query heads (only gqa used), N = 16 tokens (page).
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

    __shared__ __nv_bfloat16 K_b[PAGE][HEAD];       // K_src[tok][dim]
    __shared__ __nv_bfloat16 V_T[HEAD][PAGE+2];     // V_T[dim][tok], padded
    __shared__ __nv_bfloat16 K_b2[PAGE][HEAD];
    __shared__ __nv_bfloat16 V_T2[HEAD][PAGE+2];

    const int seqlen = cache_seqlens[b];
    const int t_start = split * tokens_per_split;
    if (t_start >= seqlen) return;   // empty split (batch's seqlen shorter than split start)
    const int h0 = kv * gqa;
    const int KVSTR = num_heads_k * HEAD;
    const int slot = (b * num_heads_k + kv) * n_splits + split;
    const int row = lane & 15;
    const int grp = lane >> 4;

    // Q fragment: [8 steps][2 regs], each reg = 2 consecutive dims
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

    // Software pipeline: each thread holds one page's worth of K/V in registers,
    // issued before the previous page's MMA work. Stage A reads buffer X while
    // staging B reads (registers) fill the next page into buffer 1-X.
    // Register footprint per thread: 8 uint4 (4 K + 4 V) = 32 regs.
    uint4 kvR[4], vvR[4];

    auto stage_page = [&](int pg, __nv_bfloat16 (*Kb)[HEAD], __nv_bfloat16 (*Vt)[PAGE+2]) {
        int pid = block_table[b * blocks_per_batch + pg];
        const __nv_bfloat16* kbase = k_cache_paged + ((int64_t)pid * PAGE) * KVSTR + kv * HEAD;
        const __nv_bfloat16* vbase = v_cache_paged + ((int64_t)pid * PAGE) * KVSTR + kv * HEAD;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int u = lane + i * 64;      // 0..255
            int tok = u >> 4;
            int dim = (u & 15) * 8;
            *(uint4*)&Kb[tok][dim] = kvR[i];
            __nv_bfloat16* vv = (__nv_bfloat16*)&vvR[i];
            #pragma unroll
            for (int e = 0; e < 8; e++) Vt[dim + e][tok] = vv[e];
        }
    };

    auto prefetch = [&](int pg) {
        int pid = block_table[b * blocks_per_batch + pg];
        const __nv_bfloat16* kbase = k_cache_paged + ((int64_t)pid * PAGE) * KVSTR + kv * HEAD;
        const __nv_bfloat16* vbase = v_cache_paged + ((int64_t)pid * PAGE) * KVSTR + kv * HEAD;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int u = lane + i * 64;      // 0..255
            int tok = u >> 4;
            int dim = (u & 15) * 8;
            kvR[i] = *(const uint4*)(kbase + tok * KVSTR + dim);
            vvR[i] = *(const uint4*)(vbase + tok * KVSTR + dim);
        }
    };

    // prologue: prefetch first page
    prefetch(pg0);

    for (int i = 0; i < npages; i++) {
        int pg = pg0 + i;
        int ntok = seqlen - pg * PAGE; if (ntok > PAGE) ntok = PAGE;
        // stage buffer for this page into smem
        if (i == 0) {
            stage_page(pg, K_b, V_T);
            __syncthreads();
        } else {
            stage_page(pg, (i & 1) ? K_b2 : K_b, (i & 1) ? V_T2 : V_T);
            __syncthreads();
        }
        // issue prefetch for next page (overlaps with this page's MMA)
        if (i + 1 < npages) prefetch(pg0 + i + 1);

        // compute on the buffer just staged
        __nv_bfloat16 (*Kb)[HEAD] = (i & 1) ? K_b2 : K_b;
        __nv_bfloat16 (*Vt)[PAGE+2] = (i & 1) ? V_T2 : V_T;

        // QK: 8 MMA steps -> s[4]
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

        // mask out padding tokens
        int tok0 = grp * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) if (tok0 + i >= ntok) s[i] = -INFINITY;

        // row max over the 4 grp threads (16 tokens of this head)
        float mx = s[0];
        #pragma unroll
        for (int i = 1; i < 4; i++) mx = fmaxf(mx, s[i]);
        mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 16));
        mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 32));

        float p[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) p[i] = (s[i] == -INFINITY) ? 0.f : __expf(s[i] - mx);
        float ls = p[0] + p[1] + p[2] + p[3];
        ls += __shfl_xor_sync(FULLMASK, ls, 16);
        ls += __shfl_xor_sync(FULLMASK, ls, 32);

        float mnew = fmaxf(m, mx);
        float alpha = __expf(m - mnew);
        float beta = __expf(mx - mnew);
        #pragma unroll
        for (int st = 0; st < 8; st++)
            #pragma unroll
            for (int i = 0; i < 4; i++) D[st][i] *= alpha;
        l = l * alpha + beta * ls;
        m = mnew;

        // PV: A = P frag (scaled by beta), B = V_T frag
        uint32_t pa0 = pack_bf16_2(p[0] * beta, p[1] * beta);
        uint32_t pa1 = pack_bf16_2(p[2] * beta, p[3] * beta);
        VectorType pav = {pa0, pa1};
        #pragma unroll
        for (int st = 0; st < 8; st++) {
            uint32_t b0 = *(const uint32_t*)(&Vt[row + st * 16][grp * 4]);
            uint32_t b1 = *(const uint32_t*)(&Vt[row + st * 16][grp * 4 + 2]);
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
    const int hq = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv = hq / gqa;
    const int hh = hq % gqa;
    const int base = (b * num_heads_k + kv) * n_splits;
    const int seqlen = cache_seqlens[b];

    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (int s = 0; s < n_splits; s++) {
        if (s * tokens_per_split >= seqlen) break;   // empty split
        int idx = base + s;
        float ms = m_part[idx * gqa + hh];
        float ls = l_part[idx * gqa + hh];
        float a = acc_part[(idx * gqa + hh) * HEAD + tid];
        float mnew = fmaxf(m, ms);
        float alpha = __expf(m - mnew);
        float beta = __expf(ms - mnew);
        acc = acc * alpha + beta * a;
        l = l * alpha + beta * ls;
        m = mnew;
    }
    output[(b * num_heads + hq) * HEAD + tid] = __float2bfloat16(acc / l);
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

    // ---- adaptive split selection ----
    // Page-aligned tokens per split (multiple of 16). Tuned to observed optima:
    // the benchmark has skewed cache_seqlens (batch 0 full, rest seqlen=1), so the
    // full batch is the critical path. bs=1 needs many splits to fill SMs; large bs
    // already fills SMs from short batches so few splits suffice.
    int ns, tokens_per_split;
    if (seqlen_k <= 64) {
        ns = 1;
        tokens_per_split = (int)seqlen_k;
    } else if (seqlen_k <= 200) {
        tokens_per_split = 32;
        ns = (int)((seqlen_k + tokens_per_split - 1) / tokens_per_split);
    } else if (batch_size == 1) {
        int tpb = (int)(144.0 * sqrt((double)seqlen_k / 8192.0));
        if (tpb < 80) tpb = 80;
        if (tpb > 500) tpb = 500;
        tokens_per_split = ((tpb + 15) / 16) * 16;
        ns = (int)((seqlen_k + tokens_per_split - 1) / tokens_per_split);
    } else if (batch_size <= 8) {
        int tpb = 384;
        tokens_per_split = tpb;
        ns = (int)((seqlen_k + tokens_per_split - 1) / tokens_per_split);
    } else if (batch_size <= 32) {
        int tpb = (int)(seqlen_k / 10.0);
        if (tpb < 64) tpb = 64;
        if (tpb > 600) tpb = 600;
        tokens_per_split = ((tpb + 15) / 16) * 16;
        ns = (int)((seqlen_k + tokens_per_split - 1) / tokens_per_split);
    } else {
        int tpb = (int)(seqlen_k / 10.0);
        if (tpb < 64) tpb = 64;
        if (tpb > 400) tpb = 400;
        tokens_per_split = ((tpb + 15) / 16) * 16;
        ns = (int)((seqlen_k + tokens_per_split - 1) / tokens_per_split);
    }

    int64_t nspl = (int64_t)batch_size * num_heads_k * ns * gqa;

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

    dim3 grid((unsigned)ns, (unsigned)(batch_size * num_heads_k));
    paged_gqa_mma_kernel<<<grid, 64>>>(
        q, k_cache_paged, v_cache_paged, m_part, l_part, acc_part,
        cache_seqlens, block_table, output,
        (int)num_heads, (int)num_heads_k, (int)headdim,
        (int)page_block_size, blocks_per_batch, ns, gqa, sm_scale,
        tokens_per_split);

    if (ns > 1) {
        dim3 grid2((unsigned)batch_size, (unsigned)num_heads);
        combine_kernel<<<grid2, 128>>>(
            m_part, l_part, acc_part, cache_seqlens, output,
            (int)num_heads, (int)num_heads_k, ns, gqa, tokens_per_split);
    }
}
