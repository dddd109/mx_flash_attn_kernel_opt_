#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

/*
 * Audit-only hook. The default build neither parses nor calls MCTLASS.
 * Defining METAX_AUDIT_MCTLASS_HEADERS only checks whether its top-level
 * header can coexist with this translation unit; no MCTLASS symbol is used.
 */
#if defined(METAX_AUDIT_MCTLASS_HEADERS)
#include <mctlass/mctlass.h>
#endif

namespace c500_paged_gqa {

constexpr int kNumHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 16;
constexpr int kThreads = 256;
constexpr int kMaxSplits = 32;
constexpr int kMaxSplitBatch = 16;
constexpr float kSoftmaxScale = 0.08838834764831844055f;  // 1 / sqrt(128)
constexpr float kNegInf = -1.0e30f;

static_assert(kThreads == kPageSize * 16, "dot-product mapping assumes 16 lanes/token");

__device__ __align__(16)
float g_partial_o[kMaxSplitBatch * kNumHeads * kMaxSplits * kHeadDim];

__device__ __align__(16)
float g_partial_m[kMaxSplitBatch * kNumHeads * kMaxSplits];

__device__ __align__(16)
float g_partial_l[kMaxSplitBatch * kNumHeads * kMaxSplits];

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
  return __float2bfloat16_rn(x);
}

__device__ __forceinline__ int partial_stat_index(int b, int h, int split) {
  return (b * kNumHeads + h) * kMaxSplits + split;
}

__device__ __forceinline__ int partial_o_index(
    int b, int h, int split, int d) {
  return partial_stat_index(b, h, split) * kHeadDim + d;
}

template <int Group>
struct __align__(16) SharedStorage {
  __nv_bfloat16 q[Group][kHeadDim];
  __nv_bfloat16 k[kPageSize][kHeadDim];
  __nv_bfloat16 v[kPageSize][kHeadDim];

  // 16 threads reduce one (token, query-head) dot product.
  float dot[kPageSize][Group][16];
  float logits[Group][kPageSize];
  float weights[Group][kPageSize];

  float alpha[Group];
  float row_m[Group];
  float row_l[Group];
  int physical_page;
};

template <int Group, bool WritePartial>
__global__ __launch_bounds__(kThreads)
void paged_gqa_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int batch_size,
    int num_heads_k,
    int blocks_per_batch,
    int num_splits) {

  static_assert(Group == 4 || Group == 8, "supported GQA groups are 4 and 8");

  __shared__ SharedStorage<Group> s;

  const int tid = static_cast<int>(threadIdx.x);
  const int kv_head = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const int split = WritePartial ? static_cast<int>(blockIdx.z) : 0;

  if (b >= batch_size || kv_head >= num_heads_k) {
    return;
  }

  const int q_head_base = kv_head * Group;
  const int seq_len = cache_seqlens[b];
  const int valid_pages = (seq_len + kPageSize - 1) / kPageSize;

  const int start_page =
      WritePartial ? (valid_pages * split) / num_splits : 0;
  const int end_page =
      WritePartial ? (valid_pages * (split + 1)) / num_splits : valid_pages;

  // Q is reused for every page in this split.
  for (int idx = tid; idx < Group * kHeadDim; idx += kThreads) {
    const int g = idx / kHeadDim;
    const int d = idx - g * kHeadDim;
    const int h = q_head_base + g;
    s.q[g][d] = q[(b * kNumHeads + h) * kHeadDim + d];
  }

  if (tid < Group) {
    s.row_m[tid] = kNegInf;
    s.row_l[tid] = 0.0f;
    s.alpha[tid] = 0.0f;
  }

  constexpr int kOutputsPerThread =
      (Group * kHeadDim + kThreads - 1) / kThreads;
  float acc[kOutputsPerThread];
  int owned_idx[kOutputsPerThread];

#pragma unroll
  for (int j = 0; j < kOutputsPerThread; ++j) {
    owned_idx[j] = tid + j * kThreads;
    acc[j] = 0.0f;
  }

  __syncthreads();

  for (int logical_page = start_page;
       logical_page < end_page;
       ++logical_page) {

    const int token_base = logical_page * kPageSize;
    int valid_tokens = seq_len - token_base;
    valid_tokens = valid_tokens > kPageSize ? kPageSize : valid_tokens;
    valid_tokens = valid_tokens < 0 ? 0 : valid_tokens;

    if (tid == 0) {
      // Never inspect padding entries. logical_page is strictly below
      // ceil(cache_seqlens[b] / page_size).
      s.physical_page =
          block_table[b * blocks_per_batch + logical_page];
    }
    __syncthreads();

    // One 16-byte vector per thread for K and one for V.
    // 256 threads * 8 bf16 = one full [16, 128] page/head tile.
    const int tile_elem = tid * 8;
    const int tok = tile_elem / kHeadDim;
    const int d0 = tile_elem - tok * kHeadDim;

    int4 zero4;
    zero4.x = 0;
    zero4.y = 0;
    zero4.z = 0;
    zero4.w = 0;

    int4 kval = zero4;
    int4 vval = zero4;

    if (tok < valid_tokens) {
      const int64_t cache_offset =
          (static_cast<int64_t>(s.physical_page * kPageSize + tok)
               * num_heads_k
           + kv_head)
          * kHeadDim
          + d0;

      kval = *reinterpret_cast<const int4*>(
          k_cache_paged + cache_offset);
      vval = *reinterpret_cast<const int4*>(
          v_cache_paged + cache_offset);
    }

    *reinterpret_cast<int4*>(&s.k[tok][d0]) = kval;
    *reinterpret_cast<int4*>(&s.v[tok][d0]) = vval;
    __syncthreads();

    // Sixteen adjacent threads own one token. Each thread covers eight
    // strided dimensions, then writes one partial for each shared Q head.
    const int dot_tok = tid >> 4;
    const int dot_lane = tid & 15;

#pragma unroll
    for (int g = 0; g < Group; ++g) {
      float sum = 0.0f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int d = dot_lane + k * 16;
        sum = fmaf(
            bf16_to_float(s.q[g][d]),
            bf16_to_float(s.k[dot_tok][d]),
            sum);
      }
      s.dot[dot_tok][g][dot_lane] = sum;
    }
    __syncthreads();

    // One thread per query head performs the 16-token page reduction and
    // advances online-softmax state.
    if (tid < Group) {
      const int g = tid;
      float page_m = kNegInf;

#pragma unroll
      for (int t = 0; t < kPageSize; ++t) {
        float logit = kNegInf;
        if (t < valid_tokens) {
          float dot = 0.0f;
#pragma unroll
          for (int lane = 0; lane < 16; ++lane) {
            dot += s.dot[t][g][lane];
          }
          logit = dot * kSoftmaxScale;
        }
        s.logits[g][t] = logit;
        page_m = fmaxf(page_m, logit);
      }

      const float old_m = s.row_m[g];
      const float old_l = s.row_l[g];
      const float new_m = fmaxf(old_m, page_m);
      const float alpha =
          old_l == 0.0f ? 0.0f : __expf(old_m - new_m);

      float page_l = 0.0f;
#pragma unroll
      for (int t = 0; t < kPageSize; ++t) {
        float w = 0.0f;
        if (t < valid_tokens) {
          w = __expf(s.logits[g][t] - new_m);
        }
        s.weights[g][t] = w;
        page_l += w;
      }

      s.alpha[g] = alpha;
      s.row_m[g] = new_m;
      s.row_l[g] = old_l * alpha + page_l;
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < kOutputsPerThread; ++j) {
      const int idx = owned_idx[j];
      if (idx < Group * kHeadDim) {
        const int g = idx / kHeadDim;
        const int d = idx - g * kHeadDim;

        float page_o = 0.0f;
#pragma unroll
        for (int t = 0; t < kPageSize; ++t) {
          if (t < valid_tokens) {
            page_o = fmaf(
                s.weights[g][t],
                bf16_to_float(s.v[t][d]),
                page_o);
          }
        }
        acc[j] = acc[j] * s.alpha[g] + page_o;
      }
    }

    // All consumers must finish before the next physical page overwrites s.k/s.v.
    __syncthreads();
  }

  if (WritePartial) {
#pragma unroll
    for (int j = 0; j < kOutputsPerThread; ++j) {
      const int idx = owned_idx[j];
      if (idx < Group * kHeadDim) {
        const int g = idx / kHeadDim;
        const int d = idx - g * kHeadDim;
        const int h = q_head_base + g;
        g_partial_o[partial_o_index(b, h, split, d)] = acc[j];
      }
    }

    if (tid < Group) {
      const int h = q_head_base + tid;
      const int stat = partial_stat_index(b, h, split);
      g_partial_m[stat] = s.row_m[tid];
      g_partial_l[stat] = s.row_l[tid];
    }
  } else {
#pragma unroll
    for (int j = 0; j < kOutputsPerThread; ++j) {
      const int idx = owned_idx[j];
      if (idx < Group * kHeadDim) {
        const int g = idx / kHeadDim;
        const int d = idx - g * kHeadDim;
        const int h = q_head_base + g;
        const float inv_l = 1.0f / s.row_l[g];
        output[(b * kNumHeads + h) * kHeadDim + d] =
            float_to_bf16(acc[j] * inv_l);
      }
    }
  }
}

__global__ __launch_bounds__(kHeadDim)
void combine_splits_kernel(
    __nv_bfloat16* __restrict__ output,
    int batch_size,
    int num_splits) {

  const int row = static_cast<int>(blockIdx.x);
  const int d = static_cast<int>(threadIdx.x);
  const int b = row / kNumHeads;
  const int h = row - b * kNumHeads;

  if (b >= batch_size || d >= kHeadDim) {
    return;
  }

  float global_m = kNegInf;
  for (int s = 0; s < num_splits; ++s) {
    const int stat = partial_stat_index(b, h, s);
    const float l = g_partial_l[stat];
    if (l > 0.0f) {
      global_m = fmaxf(global_m, g_partial_m[stat]);
    }
  }

  float global_l = 0.0f;
  float out = 0.0f;

  for (int s = 0; s < num_splits; ++s) {
    const int stat = partial_stat_index(b, h, s);
    const float l = g_partial_l[stat];
    if (l > 0.0f) {
      const float beta = __expf(g_partial_m[stat] - global_m);
      global_l += l * beta;
      out = fmaf(
          g_partial_o[partial_o_index(b, h, s, d)],
          beta,
          out);
    }
  }

  output[(b * kNumHeads + h) * kHeadDim + d] =
      float_to_bf16(out / global_l);
}

inline int ceil_pow2_capped(int x, int cap) {
  int p = 1;
  while (p < x && p < cap) {
    p <<= 1;
  }
  return p > cap ? cap : p;
}

inline int choose_num_splits(
    int batch_size,
    int num_heads_k,
    int64_t seqlen_k) {

  if (batch_size > kMaxSplitBatch || seqlen_k < 8192) {
    return 1;
  }

  // C500 has 104 APs. Aim for roughly one or two resident work units/AP
  // without making a split so short that launch/combine overhead dominates.
  const int groups = batch_size * num_heads_k;
  int desired = (104 + groups - 1) / groups;
  desired = ceil_pow2_capped(desired, kMaxSplits);

  const int pages = static_cast<int>((seqlen_k + kPageSize - 1) / kPageSize);
  int page_limited = pages / 8;  // at least 128 KV tokens per split
  if (page_limited < 1) {
    page_limited = 1;
  }
  while (desired > page_limited && desired > 1) {
    desired >>= 1;
  }

  return desired;
}

}  // namespace c500_paged_gqa

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
    int64_t causal) {

  using namespace c500_paged_gqa;

  // The challenge fixes these values. Refuse unsupported shapes rather than
  // silently using incompatible indexing.
  if (batch_size <= 0 ||
      seqlen_q != 1 ||
      num_heads != kNumHeads ||
      headdim != kHeadDim ||
      page_block_size != kPageSize ||
      causal != 0 ||
      (num_heads_k != 4 && num_heads_k != 8) ||
      num_blocks % batch_size != 0) {
    return;
  }

  const int b = static_cast<int>(batch_size);
  const int hk = static_cast<int>(num_heads_k);
  const int blocks_per_batch =
      static_cast<int>(num_blocks / batch_size);
  const int splits = choose_num_splits(b, hk, seqlen_k);

  if (splits == 1) {
    dim3 grid(static_cast<unsigned>(hk),
              static_cast<unsigned>(b),
              1u);

    if (hk == 4) {
      paged_gqa_decode_kernel<8, false>
          <<<grid, kThreads>>>(
              q, k_cache_paged, v_cache_paged, output,
              cache_seqlens, block_table,
              b, hk, blocks_per_batch, 1);
    } else {
      paged_gqa_decode_kernel<4, false>
          <<<grid, kThreads>>>(
              q, k_cache_paged, v_cache_paged, output,
              cache_seqlens, block_table,
              b, hk, blocks_per_batch, 1);
    }
    return;
  }

  dim3 split_grid(static_cast<unsigned>(hk),
                  static_cast<unsigned>(b),
                  static_cast<unsigned>(splits));

  if (hk == 4) {
    paged_gqa_decode_kernel<8, true>
        <<<split_grid, kThreads>>>(
            q, k_cache_paged, v_cache_paged, output,
            cache_seqlens, block_table,
            b, hk, blocks_per_batch, splits);
  } else {
    paged_gqa_decode_kernel<4, true>
        <<<split_grid, kThreads>>>(
            q, k_cache_paged, v_cache_paged, output,
            cache_seqlens, block_table,
            b, hk, blocks_per_batch, splits);
  }

  combine_splits_kernel
      <<<static_cast<unsigned>(b * kNumHeads), kHeadDim>>>(
          output, b, splits);
}

