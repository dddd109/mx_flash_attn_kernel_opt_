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

### 1. Memory Access (High Priority)
**CUDA kernel lines 166-181:**
```cuda
// Current: scalar loads for kval, vval
int4 kval = zero4;
if (tok < valid_tokens) {
  kval = *reinterpret_cast<const int4*>(k_cache_paged + cache_offset);
}
```
- **Issue**: Indirect access through pointer arithmetic may not be optimal
- **Try**: Explicit prefetch, Software pipelining

### 2. Dot Product Unroll (High Priority)
**CUDA kernel lines 189-201:**
```cuda
#pragma unroll
for (int k = 0; k < 8; ++k) {
  sum = fmaf(bf16_to_float(s.q[g][d]),
             bf16_to_float(s.k[dot_tok][d]), sum);
}
```
- **Issue**: 8 separate bf16→float conversions per iteration
- **Try**: Use `__hfma` intrinsic for bf16 directly if available

### 3. Online Softmax Reduction (Medium Priority)
**CUDA kernel lines 206-246:**
- Thread divergence in `tid < Group` branch
- **Try**: Use warp-level reductions before shared memory sync

### 4. Split Overhead (Medium Priority)
**CUDA kernel `combine_splits_kernel`:**
- Sequential reduce over splits
- **Try**: Tree-structured reduce or keep more computation in main kernel

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