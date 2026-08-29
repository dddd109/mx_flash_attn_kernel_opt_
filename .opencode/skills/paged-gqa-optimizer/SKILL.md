---
name: paged-gqa-optimizer
description: Optimize paged GQA attention kernel for MetaX C500 GPU
---

# Paged GQA Attention Optimizer

## Target
Optimize paged grouped-query-attention (GQA) decode kernel for MetaX C500 GPU.
This skill guides an agent to iteratively optimize the kernel using profiling data.

## Code Context
- CUDA version: `metax_c500_paged_gqa_decode_attempt.cu`
- Triton version: `submission_c500_regions.py`
- Both implement the same paged GQA decode algorithm
- Group (GQA ratio) can be 4 or 8
- Key parameters: kPageSize=16, kHeadDim=128, kNumHeads=32

## MetaX C500 Architecture Notes
- **Architecture**: SM80 (Ampere-generation, compatible with NVIDIA Ampere)
- **GPU**: MetaX C500 (104 APs, 64GB)
- **Warp size**: 32 threads

### CRITICAL: No Native BF16 MMA
Unlike NVIDIA Ampere which has `mma.sync.aligned.m16n8k8.f32.bf16.bf16.f32`, **C500 does NOT have native BF16 matrix multiplication**!

All BF16 dot products must be computed as:
1. Convert bf16 → float
2. Compute float multiply-add
3. Accumulate in float
4. Convert back to bf16 (if needed)

```cuda
// This pattern is REQUIRED for C500:
float sum = 0.0f;
for (int d = 0; d < 128; ++d) {
  sum += __bfloat162float(q_bf16[d]) * __bfloat162float(k_bf16[d]);
}
```

### INT8 MMA is Available
```cuda
// Only for INT8, C500 only (__MACA_ARCH__ == 1000)
__builtin_mxc_mma_16x16x16i8(a, b, c);
```

## MetaX-Specific Intrinsics

### Available
| Intrinsic | Description |
|-----------|-------------|
| `__lane_id()` | Get lane ID within warp (NVIDIA uses `laneId()`) |
| `__builtin_mxc_ldg_b32(ptr, ...)` | Read-only global memory cache load (32-bit) |
| `__builtin_mxc_ldg_b64(...)` | Read-only global memory cache load (64-bit) |
| `__builtin_mxc_mma_16x16x16i8(a,b,c)` | INT8 MMA (C500 only) |
| `__builtin_mxc_mma_16x16x4f32(a,b,c)` | FP32 MMA |
| `__syncwave()` | Warp synchronization (NVIDIA uses `__syncwarp()`) |

### NOT Available (DO NOT USE)
| NVIDIA Intrinsic | MetaX Alternative |
|-----------------|-------------------|
| `__shfl_xor_sync` | **NOT AVAILABLE** |
| `__reduce_add_sync` | **NOT AVAILABLE** |
| `__ldg` (CUDA) | Use `__builtin_mxc_ldg_*` |
| `cp.async` | **DISABLED** (`MACA_CP_ASYNC_ACTIVATED=0`) |

### Standard CUDA Types (Work on MetaX)
```cuda
#include <cuda_bf16.h>
__nv_bfloat16, __bfloat162float(), __float2bfloat16()
__half, __half2float(), __float2half()
```

## Optimization Implications for Paged GQA

### Memory Access (PRIORITY: HIGH)
- Use `__builtin_mxc_ldg_b32` for Q, K, V loads (read-only cache)
- Vectorized loads via `int4` are still beneficial
- Ensure coalesced access patterns

### Dot Product (PRIORITY: HIGH)
- **MUST use scalar loop** with `bfloat16 → float → fmaf → float` pattern
- Unroll inner loop for better ILP
- No MMA optimization possible for BF16

### Reduction (PRIORITY: MEDIUM)
- **Warp shuffle NOT available** - must use shared memory reduction
- Pattern: write to shared memory → syncthreads → parallel reduction
- Consider tree-structured reduction

### Software Pipelining (PRIORITY: MEDIUM)
- Since cp_async disabled, manual double buffering required
- Load next page while computing current page

## Environment Setup
```bash
export MACA_PATH=/opt/maca/
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export CUDA_PATH=$MACA_PATH/tools/cu-bridge
```

## Profiler Usage
```bash
mcProfiler perf_exec \
  --cmdline "python3 benchmark_paged_gqa.py" \
  --kernelname "paged_gqa_decode_kernel" \
  --casename "decode_bs4_seq16k" \
  --cwd /root/code \
  --metrics "sm_efficiency,achieved_occupancy,dram_utilization,l2_utilization" \
  --per-kernel --single-pass
```

Profile key metrics:
- `sm_efficiency`: SM utilization (target > 80%)
- `achieved_occupancy`: Warp occupancy (target > 60%)
- `dram_utilization`: Memory bandwidth (target > 70%)
- `l2_utilization`: L2 cache hit rate (target > 40%)

## Key Metrics to Profile
- `sm_efficiency`: SM utilization (target > 80%)
- `achieved_occupancy`: Warp occupancy (target > 60%)
- `dram_utilization`: Memory bandwidth (target > 70%)
- `shared_efficiency`: Shared memory efficiency

## Code Structure

### CUDA Version (metax_c500_paged_gqa_decode_attempt.cu)
- `paged_gqa_decode_kernel<Group, WritePartial>`: Main kernel
  - Group: 4 or 8 (GQA ratio)
  - WritePartial: true for multi-split, false for single-split
- `combine_splits_kernel`: Merges partial results from splits
- `choose_num_splits()`: Decides split count based on workload

### Triton Version (submission_c500_regions.py)
- `_paged_gqa_masked_kernel`: Basic masked attention
- `_paged_gqa_locality_kernel`: Batched with locality optimization
- `_paged_gqa_fulltile_kernel`: For long sequences
- `_paged_gqa_packed_kernel`: Packed KV format
- Key params: BLOCK_N, D_TILE, NUM_SPLITS

## Identified Optimization Points

### 1. Dot Product Reduction - Warp-level Reduce (REQUIRES ANALYSIS)
**CUDA kernel lines 214-218:**
```cuda
for (int lane = 0; lane < 16; ++lane) {
  dot += s.dot[t][g][lane];
}
```
- **Analysis**: `tid < Group` means threads 0,1,2,3 (for Group=4) participate
- **Problem**: These threads have different g values (g=tid), so they CANNOT directly use warp shuffle to reduce the same g's values
- **Verdict**: Warp shuffle reduction does NOT apply here without restructuring the code
- **Alternative**: Keep the shared memory reduction but try other optimizations

### 2. bf16 to Float Conversion (MEDIUM Priority)
**CUDA kernel lines 195-198:**
```cuda
sum = fmaf(bf16_to_float(s.q[g][d]),
           bf16_to_float(s.k[dot_tok][d]), sum);
```
- **Issue**: Separate bf16→float conversion for each operand in tight loop
- **Try**: Prefetch Q values into registers before the K/V loop
  ```cuda
  // Before page loop: convert all Q values for this head group
  __nv_bfloat16 q_reg[8];  // For Group=8
  for (int d = 0; d < kHeadDim; ++d) {
    q_reg[d] = s.q[g][d];
  }
  ```
- **Expected gain**: Reduce shared memory reads and conversions

### 3. Register Prefetch for Q (MEDIUM Priority)
**CUDA kernel lines 108-113:**
- Q values are read from global memory into shared memory each iteration
- **Try**: Keep Q in registers across iterations if registers allow

### 4. Shared Memory Bank Conflict Avoidance (LOW Priority)
**CUDA SharedStorage struct:**
```cuda
float dot[kPageSize][Group][16];  // May have bank conflicts
```
- **Issue**: Strided access pattern may cause bank conflicts
- **Try**: Pad the array or change layout: `float dot[16][Group][kPageSize]`

### 5. K/V Prefetch with Software Pipelining (LOW Priority)
**CUDA kernel lines 134-182:**
- Current: Load K/V for current page, then compute
- **Try**: Double buffer - compute on page N while loading page N+1

## Safer Optimization Candidates

### A. Triton: BLOCK_N Tuning
- BLOCK_N parameter (16, 32, 64) affects occupancy and memory access
- Profile different values for different seq_len ranges

### B. Triton: NUM_SPLITS Selection
- The `_pick_natural_splits` function determines split count
- Fine-tune thresholds in this function for C500 architecture

## Quick Wins Checklist
1. [ ] Prefetch Q values into registers (safe, may improve register usage)
2. [ ] Adjust Triton BLOCK_N for different seq_len ranges
3. [ ] Profile memory access patterns with mcProfiler
4. [ ] Analyze if MACA supports bf16 intrinsics like `__hfma`

## Iteration Workflow
1. Commit current state: `git add -A && git commit -m "checkpoint"`
2. Make single change
3. Test correctness: run benchmark with reference comparison
4. Profile and compare metrics
5. If better: continue; If worse: `git restore`

## Safety Rules
- ONE change at a time
- Always have a working restore point
- Test correctness before profiling
- Keep detailed notes of each change and result

## Triton Optimization Parameters

The Triton version has tunable parameters in `submission_c500_regions.py`:

| Parameter | Current Values | Impact |
|-----------|----------------|--------|
| `BLOCK_N` | 16, 32, 64 | Memory access granularity vs occupancy |
| `D_TILE` | 64 | Head dimension tiling |
| `NUM_SPLITS` | Auto (via `_pick_natural_splits`) | Parallelism for long sequences |
| `num_stages` | 1 or 2 | Software pipeline depth |

### BLOCK_N Selection Heuristic (lines 882-887)
```python
def _pick_block_n(seqlen_k):
    if seqlen_k <= 16:
        return 16   # Fine granularity
    if seqlen_k <= 256:
        return 32   # Balanced
    return 64        # Coarse granularity for long sequences
```
- **Try**: Adjust thresholds based on C500 memory hierarchy
- Small BLOCK_N: Better cache utilization, lower occupancy
- Large BLOCK_N: Worse cache utilization, higher occupancy

### NUM_SPLITS Selection (lines 830-864)
Controls how many parallel work units process a single decode query.
- More splits: Better parallelism for long sequences, more overhead
- Fewer splits: Less overhead, less parallelism

## CUDA Optimization Parameters

| Constant | Current Value | Impact |
|----------|---------------|--------|
| `kThreads` | 256 | Threads per block |
| `kPageSize` | 16 | Tokens per page |
| `kMaxSplits` | 32 | Maximum split count |
| `kMaxSplitBatch` | 16 | Max batch for split |

### Key Hot Loops to Optimize

**1. K/V Load (lines 163-178):** Memory-bound, vectorized load
**2. Dot Product (lines 193-199):** Compute-bound, bf16→float conversion
**3. Online Softmax (lines 211-240):** Mixed, reduction operations

## Profiling Workflow

1. Run baseline benchmark:
   ```bash
   python3 benchmark_paged_gqa.py > baseline_results.csv
   ```

2. Profile specific kernel:
   ```bash
   mcProfiler perf_exec --cmdline "python3 profile_run.py" ...
   ```

3. Compare metrics before/after each change

4. Key ratios to check:
   - If `sm_efficiency < 50%`: Likely memory-bound, optimize memory access
   - If `achieved_occupancy < 40%`: Resource contention, adjust block size
   - If `dram_utilization < 30%`: Poor memory access pattern