---
name: paged-gqa-cuda-optimizer-v15
description: Guide agent to optimize paged GQA attention using CUDA for MetaX C500 (verified benchmark)
---

# Paged GQA Attention Optimization Guide

## ⚠️ CRITICAL: Independent Implementation Required ⚠️
You MUST write your OWN kernel. Do NOT copy, paste, or structurally mirror any existing
CUDA implementation. The final kernel must be authored by you.

## How to use this guide
This document has three parts:
1. **METHODOLOGY** (the teaching): how to measure, classify bottlenecks, experiment.
2. **THE BENCHMARK** (authoritative): the 14 real test cases and their flash baselines.
3. **PRIOR STUDENTS' NOTES**: handoff hypotheses - VERIFY before trusting.

## The Benchmark (authoritative - 14 cases, user-verified)
warmup=3 for all. flash baseline measured with num_splits=0 via cuda events.
NOTE: 8 of these cases have num_heads_k=8 (gqa_ratio=4); 6 have num_heads_k=4 (gqa=8).

| case | batch | seqlen_k | nkv | gqa | type | iters | flash(us) |
|------|-------|----------|-----|-----|------|-------|-----------|
| 1 | 4 | 2 | 4 | 8 | edge | 100 | 31.1 |
| 2 | 4 | 2 | 8 | 4 | edge | 100 | 31.1 |
| 3 | 16 | 17 | 4 | 8 | perf | 100 | 38.3 |
| 4 | 16 | 64 | 8 | 4 | perf | 50 | 38.1 |
| 5 | 16 | 141 | 4 | 8 | perf | 50 | 38.3 |
| 6 | 16 | 362 | 8 | 4 | perf | 50 | 38.9 |
| 7 | 64 | 2048 | 4 | 8 | perf | 12 | 162.8 |
| 8 | 16 | 4096 | 4 | 8 | perf | 25 | 92.4 |
| 9 | 32 | 4096 | 8 | 4 | perf | 12 | 277.4 |
| 10 | 1 | 8192 | 4 | 8 | perf | 25 | 59.1 |
| 11 | 16 | 12251 | 8 | 4 | perf | 12 | 383.0 |
| 12 | 8 | 32768 | 8 | 4 | perf | 12 | 553.3 |
| 13 | 1 | 58966 | 4 | 8 | perf | 25 | 153.6 |
| 14 | 1 | 61519 | 4 | 8 | perf | 25 | 160.5 |

Score model: S=100/(1+(Tk-Th)/(Tb-Th)), Tk=your total over 14 cases, Tb=flash total,
Th fixed such that a strong kernel ~62-63. flash=50, target 62+.

## Task Specification
Read `/root/code/task.md` for the full interface (run_kernel signature, semantics).

## Key Constraints
- num_heads=32, headdim=128, page_block_size=16, seqlen_q=1, causal=0
- num_heads_k ∈ {4, 8} (gqa = 32/num_heads_k = 8 or 4)
- cache_seqlens[b]: per-batch actual length; only trust this for validity
- block_table: (batch, num_blocks/batch); only first ceil(cache_seqlens[b]/16) entries valid
- num_blocks = batch * ceil(seqlen_k/16). Case 1: all sequences length 1 (edge).
- Every case: >=1 seq at capacity; batch>1 also >=1 seq at length 1.

## METHODOLOGY

### Measure correctly
torch.cuda.Event timing; NOT torch.profiler (underreports on this GPU).
Take best-of-N for tiny cases (launch floor is noisy).

### Classify each case (launch/memory/compute/occupancy-bound)
Launch-bound: tiny work, time ~ launch cost. Memory-bound: bandwidth-limited.
Compute-bound: MMA/FMA limited. Occupancy-bound: too few warps resident.

### The regression rule (MANDATORY)
After EVERY change run ALL 14 cases. If any case regresses >5%, revert.
Backup before each change. Prior student lost 40 points fixing one case and
breaking others - do not repeat this.

## PRIOR STUDENTS' NOTES (verify yourself)

### Gen2/Gen6 kernels (~51-53) - the real bottleneck (VERIFY THIS YOURSELF)
A prior kernel (gen6, ~53) is slow on num_heads_k=8 large cases (9/11/12). Teacher
experiments showed:
- Forcing FEWER splits (to mimic the 62-kernel's choice) made it WORSE (case 12:
  574 -> 852us). So "restrain splits" is NOT the fix.
- The 62-kernel reaches case 12 at 494us using only ~13 splits. gen6 needs ~170
  splits to get 574us. Same data, same GPU.

DIAGNOSIS QUESTION for you: why can gen6's SINGLE block not sweep a long page run
efficiently? What does a block do per page? If it is:
  [stage page into smem] -> [__syncthreads] -> [compute] -> [next page]
then the __syncthreads + load latency is serialized per page (~2048 pages in case 12).
Investigate whether the per-page serial chain (load latency exposed at every barrier)
is the stall. Ideas to test WITHOUT being told the answer:
- Can the next page's loads be issued BEFORE the current page's compute, using
  SEPARATE shared-memory buffers that alternate (double-buffer), so load of page i+1
  overlaps compute of page i? (Prior student tried this with a big register buffer and
  LOST occupancy - the tradeoff is real. Test small.)
- Can the block keep MORE pages resident to hide latency?
- What is the block's achieved memory throughput vs the theoretical? If far below,
  it's latency-bound in the sweep, not bandwidth-bound.
Measure, don't guess. If double-buffering drops occupancy too far, try a shallower
pipeline (e.g. prefetch just the block_table page ids, or half a page ahead).

### gen6's strengths (keep)
MMA raw intrinsic, register-cached Q, warp-shuffle softmax, fused ns==1 short path,
adaptive split that helped cases 9/11.

### What a 62-scoring kernel does differently
- Its SINGLE block sweeps long page runs near bandwidth limit (that's why few splits
  suffice). It beats flash on short cases (~0.4x) and ~matches on long (0.85-1.1x).
- Even the 62-kernel is SLOWER than flash on case 10/13/14 (batch=1 long) - hard for all.

### Case 1 special
All cache_seqlens = 1 (single token). Edge case, exact required.

## Newest diagnosis (gen7 measured - verify and ACT on it)

A prior kernel reaches only **3 blocks/SM (~9% occupancy)** on long cases. Its shared
memory footprint (~8.7KB/block) is what limits how many blocks fit per SM. With so few
resident warps, a long page sweep is LATENCY-bound: each page's DRAM load (~800ns) has
almost no co-resident warps to hide it behind the per-page barrier.

Experiments already done (do not repeat):
- Double-buffer prefetch: hides sweep-length sensitivity but DROPS bandwidth to
  0.82 TB/s (worse). Not the answer alone.
- Removing smem padding: catastrophic bank conflicts. Padding is load-bearing.
- __syncwarp instead of __syncthreads: neutral.
- launch_bounds / register caps: neutral (smem binds, not registers).
- Finer splits: helps cases 9/11 a little; case 12 is flat beyond ~170.

So the lever is **occupancy via per-block footprint**. Test these (measure case 12,
and re-check all 14):
- If one block currently stages ONE page and serves ONE kv-head's group of query heads,
  could it stage ONE page and serve TWO kv-head groups? That halves the smem per
  query-head group and doubles useful work per block load.
- Or: shrink the staged tile so 6-8 blocks fit per SM.
- Target: get case 12 from ~573us toward the ~494us a strong kernel achieves, and
  raise blocks/SM from 3 toward 6+.
- Use the occupancy API (or compute smem*blocks<=64KB) to check blocks/SM after each change.

## Environment
export MACA_PATH=/opt/maca/
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH

## Compile
/opt/maca/mxgpu_llvm/bin/mxcc -std=c++17 -shared -fPIC your_kernel.cu -o your_kernel.so \
    -I/opt/maca/include -I/opt/maca/tools/cu-bridge/include
