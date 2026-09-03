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

## Newest diagnosis (gen8 measured - verify and ACT on it)

Prior attempts (all measured, all failed to break 53 on the nkv=8 long cases):
- Packing different kv_heads into one MMA: STRUCTURALLY impossible. The 16x16x16 MMA
  has ONE shared B operand (K column-block); all 16 M-rows share one kv_head's K.
- Two kv_heads per block: their K slices differ, nothing shareable; smem doubled -> 3
  blocks/SM -> worse.
- Removing the K smem tile to raise occupancy: per-thread 8B fragment loads break
  coalescing -> ~80% worse.
- Split tuning: cases 9/11 gained a little; case 12 is flat beyond ns~170.
- Double-buffer: hides sweep sensitivity but caps bandwidth below the fine-split case.

Measured facts for case 12 (batch=8, seq=32768, nkv=8):
- gen6 achieves ~0.94 TB/s; the reference kernel ~1.09 TB/s; device copy peak ~1.18 TB/s.
- So the gap is LATENCY (memory-level parallelism), not peak bandwidth, not occupancy.
- gen6 runs 7 blocks/SM already. More blocks didn't help.

## The open question for you (long-sweep MLP)

The reference streams a long page run at ~1.09 TB/s with fewer resident warps than
gen6. That means its THREADS have more independent loads in flight (memory-level
parallelism), OR it issues loads far ahead of use. gen6's per-page chain is:
  [issue page-i loads] -> [wait all] -> [compute] -> [issue page i+1] ...
which serializes: while computing page i, NO page i+1 loads are in flight.

Test hypotheses (measure case 12, re-check all 14):
1. Start issuing page i+1's K/V loads BEFORE computing page i, WITHOUT waiting, into a
   second small buffer. Keep buffers SMALL (see occupancy). This is classic
   double-buffering BUT the key is to keep per-buffer smem small enough that blocks/SM
   doesn't collapse - previous double-buffer attempts used big register buffers and lost.
2. Or: within one page, can each thread issue its K-loads for MULTIPLE pages at once
   (deeper per-thread MLP) rather than one page at a time?
3. Measure achieved TB/s on case 12 as you go; the target is ~1.09 TB/s.
This likely needs a careful rewrite of the inner loop. Budget generously.

## HINTS (teacher-allowed explicit directions - still YOUR job to implement)

Read your kernel's hot page loop and remove work that repeats on EVERY page when it
only matters for the LAST page.

### H5 (TEACHER PRIORITY #1): eliminate the per-page V transpose
Your kernel transposes V into dim-major shared memory AT LOAD TIME:
for each 16-byte vector loaded you scatter 8 SCALAR stores (`Vt[dim+e][tok] = vv[e]`).
That is 32 scalar shared-memory scatter-stores per thread per page, paid on EVERY page
of a long sweep (2048 pages in case 12 => 65536 scatter stores/thread just to transpose).

Why is it transposed? So the PV MMA can read 4 consecutive tokens of a fixed dim as one
vectorized word. But transposing on every store is expensive.

The alternative (what a strong kernel does): **keep V in its natural token-major layout**
(store the loaded 16-byte vector with ONE vector store, like you already do for K), and
make the PV read use an INDEX TRANSFORMATION so that reading 4 tokens of a fixed dim
does not bank-conflict. Concretely: if V stays `V_b[tok][dim]` (no +2 pad needed if you
swizzle), then the PV B-fragment for dim d and tokens t0..t3 is a gather of 4 values
`V_b[t0][d], V_b[t1][d], V_b[t2][d], V_b[t3][d]` (they are strided, NOT contiguous) -
pack them yourself into the 2 operand words. To avoid bank conflicts on that gather,
store V under a swizzled index so that (token,d) and (token+1,d) map to different banks.
A classic choice: XOR some token bits into the dim index (e.g. bank = ((d ^ (t<<k))/2)%32).
The KEY win: you remove the per-page transpose entirely; the only added cost is address
arithmetic (an XOR) on the PV read, which happens far fewer times than the transpose stores.

Measure the impact: your achieved TB/s on case 12 (currently ~0.94) should move toward
~1.09.

### H6 (TEACHER PRIORITY #2): simplify the softmax rescale
Your per-page online-softmax computes TWO exponentials (alpha AND beta) and multiplies
the P fragment by beta before PV. A numerically equivalent form needs only ONE rescale:
compute the merged max first (new_max = max(row_max, page_max)), then express every
weight relative to new_max directly. Then the running accumulator rescale uses one
factor, and P needs no beta multiplication. This removes one __expf and one multiply
per element per page. On 2048 pages that is real savings.

### H7 (lower priority): tail page load
Your tail path loads the full 16-token page then masks. A strong kernel predicates the
tail load so it never reads tokens beyond the sequence. Since the tail is 1 page per
sequence this is minor, but keeps reads from going out of bounds.

Prior student results (don't repeat): pure block_table prefetch neutral; removing K smem
broke coalescing; H1 (full/tail split of the loop) was worth only ~1% because case 12 is
DRAM-latency bound, not mask bound. NONE of them tried H5 (eliminating the V transpose),
which attacks the load path directly.

## Environment
export MACA_PATH=/opt/maca/
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH

## Compile
/opt/maca/mxgpu_llvm/bin/mxcc -std=c++17 -shared -fPIC your_kernel.cu -o your_kernel.so \
    -I/opt/maca/include -I/opt/maca/tools/cu-bridge/include
