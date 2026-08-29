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

### 1. Dot Product Reduction - Warp-level Reduce (HIGH PRIORITY)
**CUDA kernel lines 214-218:**
```cuda
for (int lane = 0; lane < 16; ++lane) {
  dot += s.dot[t][g][lane];
}
```
- **Issue**: Reads from shared memory 16 times for simple reduction
- **Try**: Use warp-level shuffle reduction before syncthreads
  ```cuda
  // Use __shfl_xor_sync to reduce within warp
  float dot = s.dot[t][g][dot_lane];
  dot += __shfl_xor_sync(0xffffffff, dot, 16);
  dot += __shfl_xor_sync(0xffffffff, dot, 8);
  dot += __shfl_xor_sync(0xffffffff, dot, 4);
  dot += __shfl_xor_sync(0xffffffff, dot, 2);
  dot += __shfl_xor_sync(0xffffffff, dot, 1);
  ```
- **Expected gain**: ~30-50% faster reduction phase

### 2. bf16 to Float Conversion (MEDIUM Priority)
**CUDA kernel lines 195-198:**
```cuda
sum = fmaf(bf16_to_float(s.q[g][d]),
           bf16_to_float(s.k[dot_tok][d]), sum);
```
- **Issue**: Separate bf16→float conversion for each operand
- **Try**: Check if MACA supports `__hfma2` or similar bf16 intrinsics
- **Alternative**: Batch convert Q values before the inner loop

### 3. Exp Approximation (LOW-MEDIUM Priority)
**CUDA kernel line 236:**
```cuda
w = __expf(s.logits[g][t] - new_m);
```
- **Issue**: `__expf` is expensive
- **Try**: Use `__expf` approximation with limited precision if accuracy allows
- Or use table lookup for common values

### 4. Combine Kernel Optimization (LOW Priority)
**CUDA kernel `combine_splits_kernel` lines 322-345:**
- Two separate loops over splits
- **Try**: Fuse into single pass using warp reductions

## Quick Wins Checklist
1. [ ] Warp-level reduction for dot product accumulation
2. [ ] Prefetch Q values before K/V loop (reduce registers)
3. [ ] Check MACA-specific intrinsics for bf16
4. [ ] Profile memory access patterns

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