---
name: paged-gqa-optimizer
description: Optimize paged GQA attention kernel for MetaX C500 GPU
---

# Paged GQA Attention Optimizer

## Target
Optimize paged grouped-query-attention (GQA) decode kernel for MetaX C500 GPU.

## Profiler Usage
```bash
mcProfiler perf_exec \
  --cmdline "python3 benchmark.py" \
  --kernelname "paged_gqa_decode_kernel" \
  --casename "decode_bs4_seq16k" \
  --cwd /root/code \
  --metrics "sm_efficiency,achieved_occupancy,dram_utilization" \
  --per-kernel --single-pass
```

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