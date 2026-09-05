#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

#include <mctlass/numeric_types.h>
#include <cute/arch/mma_sm80.hpp>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace my_optimized_gqa {

using MctBfloat16 = mctlass::bfloat16_t;
using MmaOp = cute::MACA_16x16x16_F32BF16BF16F32;

constexpr int kNumHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 16;
constexpr int kWaveSize = 64;
constexpr int kMmaDim = 16;
constexpr int kMmaKTiles = kHeadDim / kMmaDim;
constexpr int kMaxSplits = 128;
constexpr int kMaxPartialRows = 8192;
constexpr int kC500APs = 104;
constexpr int kResidentBlocksPerAP = 8;

constexpr float kSoftmaxScale = 0.08838834764831844055f;
constexpr float kNegInf = -1.0e30f;

static_assert(sizeof(MctBfloat16) == 2, "MCTLASS BF16 must be 16 bits");
static_assert(sizeof(__nv_bfloat16) == 2, "interface BF16 must be 16 bits");
static_assert(kMmaKTiles == 8, "fixed head-dimension specialization");

__device__ __align__(16) float g_partial_o[kMaxPartialRows * kHeadDim];
__device__ __align__(16) float g_partial_m[kMaxPartialRows];
__device__ __align__(16) float g_partial_l[kMaxPartialRows];

struct alignas(16) SharedStorage {
  uint16_t k[kPageSize * kHeadDim];
  uint16_t v[kPageSize * kHeadDim];
};

struct Mma16x16x16Bf16 {
  __device__ __forceinline__ static void accumulate(
      float (&c)[4],
      uint32_t a0, uint32_t a1,
      uint32_t b0, uint32_t b1) {
    float d0, d1, d2, d3;
    MmaOp::fma(d0, d1, d2, d3, a0, a1, b0, b1, c[0], c[1], c[2], c[3]);
    c[0] = d0; c[1] = d1; c[2] = d2; c[3] = d3;
  }
};

__device__ __forceinline__ float fast_exp(float x) {
  return __expf(x);
}

__device__ __forceinline__ uint32_t pack_u16(uint16_t lo, uint16_t hi) {
  return uint32_t(lo) | (uint32_t(hi) << 16);
}

__device__ __forceinline__ uint16_t bf16_bits(float x) {
  const __nv_bfloat16 value = __float2bfloat16_rn(x);
  return *reinterpret_cast<const uint16_t*>(&value);
}

__device__ __forceinline__ int partial_row(int batch, int head, int split, int num_splits) {
  return (batch * kNumHeads + head) * num_splits + split;
}

__device__ __forceinline__ int qk_smem_index(int row, int col) {
  const int tile = col >> 6;
  const int inner = row * 64 + (col & 63);
  const int swizzled = inner ^ (((inner >> 6) & 7) << 3);
  return tile * (kPageSize * 64) + swizzled;
}

__device__ __forceinline__ int v_smem_index(int token, int d) {
  return token * kHeadDim + (d ^ (token << 3));
}

__device__ __forceinline__ float reduce_same_row_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(uint64_t(-1), value, 16));
  value = fmaxf(value, __shfl_xor_sync(uint64_t(-1), value, 32));
  return value;
}

__device__ __forceinline__ float reduce_same_row_sum(float value) {
  value += __shfl_xor_sync(uint64_t(-1), value, 16);
  value += __shfl_xor_sync(uint64_t(-1), value, 32);
  return value;
}

__device__ __forceinline__ int resolve_physical_page(
    const int32_t* __restrict__ block_table,
    int batch, int blocks_per_batch, int logical_page, int lane) {
  int physical_page = lane == 0
      ? block_table[batch * blocks_per_batch + logical_page]
      : 0;
  return __shfl_sync(uint64_t(-1), physical_page, 0);
}

__device__ __forceinline__ void load_full_page_affine(
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    SharedStorage& shared,
    int physical_page, int lane, int kv_head, int num_heads_k) {

  const int token0 = lane >> 4;
  const int d0 = (lane & 15) << 3;
  const int token_stride = num_heads_k * kHeadDim;
  const int iteration_stride = 4 * token_stride;
  int64_t full_ptr_offset =
      (int64_t(physical_page) * kPageSize * num_heads_k + kv_head) * kHeadDim
      + token0 * token_stride + d0;
  int token = token0;

#pragma unroll
  for (int iteration = 0; iteration < 4; ++iteration) {
    const int4 k_vector = *reinterpret_cast<const int4*>(k_cache + full_ptr_offset);
    const int4 v_vector = *reinterpret_cast<const int4*>(v_cache + full_ptr_offset);

    *reinterpret_cast<int4*>(shared.k + qk_smem_index(token, d0)) = k_vector;
    *reinterpret_cast<int4*>(shared.v + v_smem_index(token, d0)) = v_vector;

    full_ptr_offset += iteration_stride;
    token += 4;
  }
}

__device__ __forceinline__ void load_tail_page_masked(
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    SharedStorage& shared,
    int physical_page, int valid_tokens, int lane, int kv_head, int num_heads_k) {

#pragma unroll
  for (int iteration = 0; iteration < 4; ++iteration) {
    const int vector_index = lane + iteration * kWaveSize;
    const int token = vector_index >> 4;
    const int d0 = (vector_index & 15) << 3;

    int4 k_vector, v_vector;

    if (token < valid_tokens) {
      const int64_t cache_offset =
          (int64_t(physical_page * kPageSize + token) * num_heads_k + kv_head) * kHeadDim + d0;
      k_vector = *reinterpret_cast<const int4*>(k_cache + cache_offset);
      v_vector = *reinterpret_cast<const int4*>(v_cache + cache_offset);
    } else {
      k_vector = {0, 0, 0, 0};
      v_vector = {0, 0, 0, 0};
    }

    *reinterpret_cast<int4*>(shared.k + qk_smem_index(token, d0)) = k_vector;
    *reinterpret_cast<int4*>(shared.v + v_smem_index(token, d0)) = v_vector;
  }
}

template <bool FullPage>
__device__ __forceinline__ void process_page(
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    SharedStorage& shared,
    int physical_page, int valid_tokens, int lane, int kv_head, int num_heads_k,
    int fragment_base, int mma_row, bool valid_row,
    const uint32_t (&q_fragment)[kMmaKTiles][2],
    float (&output_acc)[kMmaKTiles][4],
    float& row_max, float& row_sum) {

  if (FullPage) {
    load_full_page_affine(k_cache, v_cache, shared, physical_page, lane, kv_head, num_heads_k);
  } else {
    load_tail_page_masked(k_cache, v_cache, shared, physical_page, valid_tokens, lane, kv_head, num_heads_k);
  }

  __syncthreads();

  float logits[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int tile = 0; tile < kMmaKTiles; ++tile) {
    const int k_col = tile * kMmaDim + fragment_base;
    const uint32_t* k_words = reinterpret_cast<const uint32_t*>(shared.k + qk_smem_index(mma_row, k_col));
    Mma16x16x16Bf16::accumulate(logits, q_fragment[tile][0], q_fragment[tile][1], k_words[0], k_words[1]);
  }

  float local_max = kNegInf;

#pragma unroll
  for (int value = 0; value < 4; ++value) {
    const int token = fragment_base + value;
    if (valid_row && (FullPage || token < valid_tokens)) {
      logits[value] *= kSoftmaxScale;
      local_max = fmaxf(local_max, logits[value]);
    } else {
      logits[value] = kNegInf;
    }
  }

  const float page_max = reduce_same_row_max(local_max);
  const float new_max = valid_row ? fmaxf(row_max, page_max) : kNegInf;
  const float alpha = valid_row && row_sum != 0.0f ? fast_exp(row_max - new_max) : 0.0f;

  float weights[4];
  float local_sum = 0.0f;

#pragma unroll
  for (int value = 0; value < 4; ++value) {
    const int token = fragment_base + value;
    float weight = 0.0f;
    if (valid_row && (FullPage || token < valid_tokens)) {
      weight = fast_exp(logits[value] - new_max);
    }
    weights[value] = weight;
    local_sum += weight;
  }

  const float page_sum = reduce_same_row_sum(local_sum);

  if (valid_row) {
    row_max = new_max;
    row_sum = row_sum * alpha + page_sum;
  }

#pragma unroll
  for (int tile = 0; tile < kMmaKTiles; ++tile) {
#pragma unroll
    for (int value = 0; value < 4; ++value) {
      output_acc[tile][value] *= alpha;
    }
  }

  const uint32_t p0 = pack_u16(bf16_bits(weights[0]), bf16_bits(weights[1]));
  const uint32_t p1 = pack_u16(bf16_bits(weights[2]), bf16_bits(weights[3]));

#pragma unroll
  for (int d_tile = 0; d_tile < kMmaKTiles; ++d_tile) {
    const int d = d_tile * kMmaDim + mma_row;
    const uint16_t v0 = shared.v[v_smem_index(fragment_base + 0, d)];
    const uint16_t v1 = shared.v[v_smem_index(fragment_base + 1, d)];
    const uint16_t v2 = shared.v[v_smem_index(fragment_base + 2, d)];
    const uint16_t v3 = shared.v[v_smem_index(fragment_base + 3, d)];
    Mma16x16x16Bf16::accumulate(output_acc[d_tile], p0, p1, pack_u16(v0, v1), pack_u16(v2, v3));
  }

  __syncthreads();
}

__global__ __launch_bounds__(kWaveSize)
void paged_gqa_decode_mma(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int batch_size, int num_heads_k, int blocks_per_batch, int num_splits) {

  __shared__ SharedStorage shared;

  const int lane = int(threadIdx.x);
  const int kv_head = int(blockIdx.x);
  const int batch = int(blockIdx.y);
  const int split = int(blockIdx.z);

  if (lane >= kWaveSize || batch >= batch_size || kv_head >= num_heads_k) {
    return;
  }

  const int group = kNumHeads / num_heads_k;
  const int mma_row = lane & 15;
  const int lane_quarter = lane >> 4;
  const int fragment_base = lane_quarter * 4;
  const bool valid_row = mma_row < group;
  const int q_head = kv_head * group + mma_row;

  uint32_t q_fragment[kMmaKTiles][2];

#pragma unroll
  for (int tile = 0; tile < kMmaKTiles; ++tile) {
    uint32_t q0 = 0, q1 = 0;
    if (valid_row) {
      const int d0 = tile * kMmaDim + fragment_base;
      const int64_t q_offset = (int64_t(batch) * kNumHeads + q_head) * kHeadDim + d0;
      const uint32_t* q_words = reinterpret_cast<const uint32_t*>(q + q_offset);
      q0 = q_words[0];
      q1 = q_words[1];
    }
    q_fragment[tile][0] = q0;
    q_fragment[tile][1] = q1;
  }

  float output_acc[kMmaKTiles][4];

#pragma unroll
  for (int tile = 0; tile < kMmaKTiles; ++tile) {
#pragma unroll
    for (int value = 0; value < 4; ++value) {
      output_acc[tile][value] = 0.0f;
    }
  }

  float row_max = kNegInf;
  float row_sum = 0.0f;

  const int seq_len = cache_seqlens[batch];
  const int valid_pages = (seq_len + kPageSize - 1) / kPageSize;
  const int page_begin = (valid_pages * split) / num_splits;
  const int page_end = (valid_pages * (split + 1)) / num_splits;

  const int tail_tokens = seq_len & (kPageSize - 1);
  const bool has_tail = tail_tokens != 0;
  const int tail_page = valid_pages - 1;
  const int full_page_end = has_tail && page_end > tail_page ? tail_page : page_end;

  for (int logical_page = page_begin; logical_page < full_page_end; ++logical_page) {
    const int physical_page = resolve_physical_page(block_table, batch, blocks_per_batch, logical_page, lane);
    process_page<true>(k_cache, v_cache, shared, physical_page, kPageSize, lane, kv_head, num_heads_k,
        fragment_base, mma_row, valid_row, q_fragment, output_acc, row_max, row_sum);
  }

  if (has_tail && page_begin <= tail_page && tail_page < page_end) {
    const int physical_page = resolve_physical_page(block_table, batch, blocks_per_batch, tail_page, lane);
    process_page<false>(k_cache, v_cache, shared, physical_page, tail_tokens, lane, kv_head, num_heads_k,
        fragment_base, mma_row, valid_row, q_fragment, output_acc, row_max, row_sum);
  }

  if (!valid_row) {
    return;
  }

  if (num_splits > 1) {
    const int row = partial_row(batch, q_head, split, num_splits);

#pragma unroll
    for (int d_tile = 0; d_tile < kMmaKTiles; ++d_tile) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        const int d = d_tile * kMmaDim + fragment_base + value;
        g_partial_o[row * kHeadDim + d] = output_acc[d_tile][value];
      }
    }

    if (lane_quarter == 0) {
      g_partial_m[row] = row_max;
      g_partial_l[row] = row_sum;
    }
    return;
  }

  const float inverse_sum = 1.0f / row_sum;

#pragma unroll
  for (int d_tile = 0; d_tile < kMmaKTiles; ++d_tile) {
#pragma unroll
    for (int value = 0; value < 4; ++value) {
      const int d = d_tile * kMmaDim + fragment_base + value;
      output[(int64_t(batch) * kNumHeads + q_head) * kHeadDim + d] =
          __float2bfloat16_rn(output_acc[d_tile][value] * inverse_sum);
    }
  }
}

__global__ __launch_bounds__(kHeadDim)
void combine_splits(
    __nv_bfloat16* __restrict__ output,
    int batch_size, int num_splits) {

  const int row = int(blockIdx.x);
  const int d = int(threadIdx.x);
  const int batch = row / kNumHeads;
  const int head = row - batch * kNumHeads;

  if (batch >= batch_size || d >= kHeadDim) {
    return;
  }

  float global_max = kNegInf;

#pragma unroll 1
  for (int split = 0; split < num_splits; ++split) {
    const int partial = partial_row(batch, head, split, num_splits);
    if (g_partial_l[partial] > 0.0f) {
      global_max = fmaxf(global_max, g_partial_m[partial]);
    }
  }

  float denominator = 0.0f;
  float numerator = 0.0f;

#pragma unroll 1
  for (int split = 0; split < num_splits; ++split) {
    const int partial = partial_row(batch, head, split, num_splits);
    const float split_sum = g_partial_l[partial];
    if (split_sum > 0.0f) {
      const float beta = fast_exp(g_partial_m[partial] - global_max);
      denominator += split_sum * beta;
      numerator = fmaf(g_partial_o[partial * kHeadDim + d], beta, numerator);
    }
  }

  output[(int64_t(batch) * kNumHeads + head) * kHeadDim + d] =
      __float2bfloat16_rn(numerator / denominator);
}

inline int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

inline bool split_eligible(int num_blocks, int num_splits) {
  return num_splits == 1 || ceil_div(num_blocks, num_splits) != ceil_div(num_blocks, num_splits - 1);
}

inline int choose_num_splits(int batch_size, int num_heads_k, int seqlen_k) {
  if (seqlen_k <= 32) {
    return 1;
  }

  const int work = batch_size * num_heads_k;
  const int target = kC500APs * kResidentBlocksPerAP;
  const int num_blocks = ceil_div(seqlen_k, kPageSize);

  if (float(work) >= 0.8f * float(target)) {
    return 1;
  }

  int max_splits = kMaxSplits;
  if (max_splits > target) max_splits = target;
  if (max_splits > num_blocks) max_splits = num_blocks;

  const int capacity = kMaxPartialRows / (batch_size * kNumHeads);
  if (max_splits > capacity) max_splits = capacity;
  if (max_splits < 1) return 1;

  if (work * max_splits < target) {
    const int minimum_pages_per_split = 32 / num_heads_k;
    const int traffic_limited_splits = num_blocks / minimum_pages_per_split;
    const int minimum_parallel_splits = ceil_div(target, 4 * work);

    if (traffic_limited_splits >= minimum_parallel_splits && max_splits > traffic_limited_splits) {
      max_splits = traffic_limited_splits;
    }
  }

  float best_efficiency = 0.0f;
  float efficiency[kMaxSplits];

  for (int splits = 1; splits <= max_splits; ++splits) {
    float current = 0.0f;
    if (split_eligible(num_blocks, splits)) {
      const float waves = float(work * splits) / float(target);
      current = waves / ceilf(waves);
      if (current > best_efficiency) {
        best_efficiency = current;
      }
    }
    efficiency[splits - 1] = current;
  }

  for (int splits = 1; splits <= max_splits; ++splits) {
    if (split_eligible(num_blocks, splits) && efficiency[splits - 1] >= 0.85f * best_efficiency) {
      return splits;
    }
  }

  return 1;
}

} // namespace my_optimized_gqa

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

  using namespace my_optimized_gqa;

  if (batch_size <= 0 || batch_size > 64 || seqlen_q != 1 ||
      num_heads != kNumHeads || headdim != kHeadDim ||
      page_block_size != kPageSize || causal != 0 ||
      (num_heads_k != 4 && num_heads_k != 8) ||
      num_blocks % batch_size != 0) {
    return;
  }

  const int batch = int(batch_size);
  const int kv_heads = int(num_heads_k);
  const int blocks_per_batch = int(num_blocks / batch_size);
  const int splits = choose_num_splits(batch, kv_heads, int(seqlen_k));

  const dim3 grid{
      static_cast<unsigned>(kv_heads),
      static_cast<unsigned>(batch),
      static_cast<unsigned>(splits)};

  paged_gqa_decode_mma<<<grid, kWaveSize>>>(
      q, k_cache_paged, v_cache_paged, output,
      cache_seqlens, block_table,
      batch, kv_heads, blocks_per_batch, splits);

  if (splits > 1) {
    combine_splits<<<static_cast<unsigned>(batch * kNumHeads), kHeadDim>>>(
        output, batch, splits);
  }
}