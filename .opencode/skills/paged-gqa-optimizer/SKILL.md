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

### Gen2 kernel (~51/63) - what held it back
A prior student kernel (call it "gen2", score ~51) was tuned on a WRONG case table
(nkv=4 everywhere). On the REAL cases it fails specifically on num_heads_k=8 cases
(9, 11, 12): it OVER-SPLITS them, launching 5000-7000 blocks (49-66 waves) instead of
~1000-2000. It produces ns*batch*nkv*gqa total blocks but its per-block work is small.
The 62-scoring kernel caps splits: batch32→8, batch16→16, batch8→32 splits max.
LESSON: total blocks should be roughly 1-4 waves of SMs (104 SMs), NOT 50+ waves.
Sweep split count; find where time stops improving. More splits ≠ faster after a point.

### gen2's strengths (keep these)
MMA via raw intrinsic, register-cached Q, warp-shuffle softmax, fused ns==1 path,
lean host logic (no getenv/atoi per call).

### What a 62-scoring kernel does differently
- Restrained split counts (1-4 waves of blocks total).
- On nkv=8 cases (gqa=4), it still fills MMA rows efficiently.
- Never reads block_table padding; respects cache_seqlens exactly.
- Beats flash on short cases (0.4x) and matches on long (0.85-1.1x).
  Note: even the 62-kernel is SLOWER than flash on case 10/13/14 (batch=1, long) -
  those are hard for everyone.

### Case 1 special
All cache_seqlens = 1 (every sequence is a single token). Edge case, must be exact.

## Environment
export MACA_PATH=/opt/maca/
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH

## Compile
/opt/maca/mxgpu_llvm/bin/mxcc -std=c++17 -shared -fPIC your_kernel.cu -o your_kernel.so \
    -I/opt/maca/include -I/opt/maca/tools/cu-bridge/include
