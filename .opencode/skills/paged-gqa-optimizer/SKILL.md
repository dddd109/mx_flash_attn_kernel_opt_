---
name: paged-gqa-optimizer-v20
description: Guide agent to optimize paged GQA attention on MetaX C500 - experience-based
---

# Paged GQA Attention Optimization Guide (Experience-Backed)

## How to use this guide
This skill contains (1) the authoritative benchmark, (2) methodology, and (3) a set of
OPTIMIZATION EXPERIENCES distilled from studying a high-scoring kernel and from many
student generations. These are experience notes - applying them well is the task.
The skill NEVER contains the reference implementation's code; it describes what to do
and WHY, so you can design your own.

## The Benchmark (authoritative - SPJ-CONFIRMED; the 3 corrections below are vital)
14 cases. num_heads=32, headdim=128, page_block_size=16, seqlen_q=1, causal=0.
Key corrections vs earlier drafts: case7 nkv=**8**, case11 nkv=**4**, case13 nkv=**8**.
| case | batch | seqlen_k | nkv | type | iters |
|------|-------|----------|-----|------|-------|
| 1 | 1 | 1 | 4 | edge | 100 |
| 2 | 4 | 2 | 8 | edge | 100 |
| 3 | 16 | 17 | 4 | edge | 100 |
| 4 | 64 | 64 | 8 | perf | 50 |
| 5 | 16 | 141 | 4 | perf | 50 |
| 6 | 16 | 362 | 8 | perf | 50 |
| 7 | 64 | 2048 | **8** | perf | 12 |
| 8 | 16 | 4096 | 4 | perf | 25 |
| 9 | 32 | 4096 | 8 | perf | 12 |
| 10 | 1 | 8192 | 4 | perf | 25 |
| 11 | 16 | 12251 | **4** | perf | 12 |
| 12 | 8 | 32768 | 8 | perf | 12 |
| 13 | 1 | 58966 | **8** | perf | 25 |
| 14 | 1 | 61519 | 4 | perf | 25 |

Key facts: case 1 all sequences length 1. Every case >=1 seq at capacity; batch>1 also
>=1 seq at length 1. num_blocks = batch*ceil(seqlen_k/16). Never read block_table padding.

## Methodology (mandatory)
- Measure with torch.cuda.Event, best-of-N for tiny cases. torch.profiler underreports.
- Regression rule: ALL 14 after every change; revert if any >5% worse. Backup first.
- Verify correctness vs flash_attn (match>=0.99, no outliers) after each change.

## OPTIMIZATION EXPERIENCES (the core teaching content)

### E1. Know your binding constraint per case class
Classify each case: launch-bound (tiny work), DRAM-latency-bound (long sweep, big data),
bandwidth-bound, compute-bound, occupancy-bound. Different cases bind differently.
case 12 (batch=8, seq=32768, nkv=8) is DRAM-LATENCY bound: 0.94 TB/s achieved vs
1.18 peak. Compute is idle (SFU/exp free), occupancy is fine (7 blocks/SM), splits don't
matter beyond ~96. The fix must raise memory-level parallelism per thread.

### E2. Software pipelining is THE lever for long sweeps, but watch the register wall
A per-page serial chain [load page -> barrier -> compute page -> next] exposes DRAM
latency at every barrier. The high-scoring kernel keeps loads in flight across the
barrier. CAUTION: on this toolchain (64 threads/block), adding a pipeline that grows
registers past ~152 spills catastrophically (measured: any +32 regs -> +400us). If you
pipeline, do it with SMALL register cost: keep buffers in shared memory (double-buffer
2 small smem tiles), NOT registers. Two smem tiles of ~4KB each keep occupancy ~4+.

### E3. Vectorized, affine loads preserve transactions
Load K/V as 16-byte vectors. Compute a base pointer ONCE per page and advance by a
constant stride per iteration (affine addressing) rather than re-deriving
(tok*stride + dim) each time. A thread that loads 4 tokens of a fixed 8-dim slice with
a precomputed offset + fixed increments issues clean 16-byte transactions with no
per-iteration address math.

### E4. Shared-memory bank conflicts: pad OR swizzle - choose deliberately
Two families of layouts avoid bank conflicts:
(a) Padded row-major: row stride made odd (e.g. +2 bf16 = +1 word) so consecutive rows
    land on different banks. Cheap, works, but each access is to a padded layout.
(b) Swizzled atom layout: XOR some bits of the linear index into other bits (e.g.
    bits 6-8 into 3-5). This is what makes a 16x16 read spread across all 32 banks.

### E4b. THE #1 HIDDEN COST: transposing V (or K) at load time with SCALAR stores
BIGGEST measured win (score 54 -> 74). If your load path transposes a matrix into
shared memory by scattering SCALAR elements (e.g. 8 separate stores per 16-byte
vector), you pay ~8x the shared-memory store instructions of a vectorized store, on
EVERY page of a long sweep (case 12: 2048 pages -> ~65k extra scalar stores/thread).

Fix: keep the matrix in its NATURAL (global-memory) layout and write each 16-byte
vector with ONE store (same as the other matrix). Make the MMA read work on the
natural layout by:
- reading the few elements it needs (the PV read needs 4 tokens at one dim; read them
  as separate uint16 and pack into the operand words in the SAME order the transposed
  layout produced), AND
- choosing row padding so those strided reads do NOT bank-conflict. Concretely for a
  [token][dim] array with head=128: row stride in words must make 16 consecutive
  tokens land on 16 different banks. stride_words ≡ 4 mod 32 works (VPAD=8 halves =
  68 words). VERIFY by matching output to the reference bit-for-bit on a tiny case.

CRITICAL LESSON from earlier failure: removing the transpose but reading with an
unpadded layout (bank conflicts) OR getting the operand packing order wrong makes it
60% SLOWER or incorrect. The win requires BOTH vectorized store AND conflict-free
packed read. A plain "gather" without the padding/swizzle is not enough.

### E5. Softmax: express everything relative to the merged max (one rescale)
Online softmax can be written two ways:
(a) weights relative to page max, then rescale accumulator by alpha AND scale P by beta
    before PV (2 exponentials + a multiply per element).
(b) compute merged max = max(running, page) FIRST, express weights directly relative to
    merged max, rescale accumulator by ONE alpha. No beta, no per-element beta multiply.
(b) removes one expf and one multiply per element per page. Small but free. On 2048
pages it adds up. MEASURED ~1%.

### E6. Full pages vs tail pages: don't let the rare case tax the common case
In a 2048-page sweep, 2047 pages are FULL (16 valid tokens). Only the last page per
sequence is partial. Structure the sweep so the steady state does ZERO boundary work:
no per-page recompute of "how many tokens are valid", no per-page masking. Handle the
one tail page separately. (MEASURED: ~1% here because this kernel is latency bound, but
it is free and removes branch pressure.) More importantly the tail load should be
predicated so you never read past cache_seqlens (correctness + no wasted bandwidth).

### E7. Compile-time shape specialization helps the compiler
If the harness always passes num_heads=32, headdim=128, page=16, nkv in {4,8}, then
hard-coding those as compile-time constants (with static_assert guards) lets the
compiler fold all indices and unroll the 8 MMA K-tiles fully. Passing them as runtime
scalars forces runtime index math. Favor constexpr for everything that is actually fixed.

### E8. Grid layout: keep dimensions meaningful and page-balance splits per batch
Use grid dims that map 1:1 to (kv_head, batch, split) with device-side guard clauses.
Compute split boundaries PER BATCH from its own valid page count
(valid_pages * split / num_splits, rounded) so every split of a short batch is balanced
and you never launch work on empty page ranges. Avoid host-side "fixed token budget"
splitting that leaves trailing splits empty for short batches.

### E9. Restrained split count driven by a CTA-population model
Target total blocks ~= (SMs x resident blocks/SM) i.e. ~832 for 104 SMs x 8. Cap splits
by: max splits constant, page count, and partial-buffer capacity (rows for
batch*heads). More splits than needed only adds combine traffic. But do NOT under-split
DRAM-heavy small-batch cases: verify by sweeping. (Student kernels found finer splits
helped nkv=8 cases 9/11; case 12 is flat.)

### E10. Combine kernel: minimize passes and dead-split work
When multiple splits per (batch,head), the combine reads all partials. Skip splits whose
row-sum <= 0 (empty/fully-masked). A two-pass (global max, then weighted sum) or a
single online pass both work - the point is to touch each partial once and skip dead
ones. Store partials in FP32.

### E11. Fused single-split path for short work
When num_splits==1 (short seq or huge batch), write the final bf16 output directly from
the main kernel - no separate combine launch. This avoids a second launch on tiny cases
where launch overhead dominates (launch floor ~4-12us).

## What NOT to repeat (student/teacher measured dead ends)
- Block-table row prefetch into smem: neutral (broadcast + HW prefetch already cover it).
- Removing the K smem tile to raise occupancy: breaks load coalescing -> ~80% worse.
- Register double-buffer (big register ring): register spills -> much worse.
- Pure split-count restraint on DRAM-bound case 12: does not reach the reference; the
  gap is per-thread memory-level parallelism, not partitioning.
- Forcing the reference's exact split constant into a different kernel structure: no.

## Handoff
The best student kernel so far scores ~54 (agent_gen10_kernel.cu). You may start from it
or from flash_attn. Applying E1-E11 well is how you go further.

## VERIFIED CONFIGURATION HISTORY (real OJ scores - the climb to beat)
Each row is a REAL OJ submission; the lessons are cumulative and all verified on this HW.

| version | OJ score | what changed / the lesson it taught |
|---------|----------|-------------------------------------|
| optimized_c500_flash_attn.cu | 62.21 | Original. **TRAP: it ships `EXP25_FORCED_SPLITS=128` default** which bypasses its own tuned split heuristic and forces ns=128 on every long case. Its real tuned heuristic (guarded by `#if EXP25_FORCED_SPLITS>0`) is ~5% better. Do NOT start from this kernel's default build. |
| gen11b | 64.57 | V token-major VPAD=8, single vectorized store, no transpose (E4b) + merged-max softmax (E5b). |
| merged | 64.93 | Per-(batch,kv,wu)-class split tuning + refined split arithmetic (E8/E9). |
| **agentG_v2 (submission_agentG_v2.cu)** | **65.14** | **CURRENT BEST. START HERE.** V token-major VPAD=8, merged-max, split policy tuned per (batch,kv,work_units) class, bare `__builtin_mxc_mma_16x16x16bf16`, 64 thr/block, grid=(ns,batch*nkv). Has NO forced-split bug (clean arithmetic policy). |

Per-case OJ profile of the 65.14 kernel (tk_ms, score) - targets to beat:
1:(0.007,84) 2:(0.007,84) 3:(0.009,83) 4:(0.021,74) 5:(0.016,74) 6:(0.027,64)
7:(0.238,54) 8:(0.079,58) 9:(0.237,57) 10:(0.048,57) 11:(0.203,55) 12:(0.397,59)
13:(0.194,55) 14:(0.142,54). First place = 72.71.
Weakest cases (biggest headroom, lowest speedup vs flash): 7/10/11/12/13/14 (~1.1-1.3x).

Verified local sums (local bench_all.py, lower is better): agentG_v2=1610us, orig62=1960us.
Rough OJ-score proxy: score ~ +1pt per ~50us of local-SUM reduction from 65.14.

## CONFIRMED DEAD ENDS on this HW (all tested 2026-08~09, do not repeat blindly)
- smem double-buffer software pipeline (2 K/V tiles): 152->176 regs (spill wall) OR
  8->4 CTAs/SM if held in smem. NET REGRESSION on latency-bound cases. MLP via this
  route fails; per-thread MLP must not grow regs past ~152.
- Multi-kv-head per CTA (2-4 warps each staging own page): __syncthreads couples the
  warps -> they can't overlap; neutral at best. Multi-warp CTAs lose independent
  progress.
- Forcing ns=128 (or any fixed large split) on small-batch/long-KV cases: oversubscribes,
  split-0 page ranges become dead work, partial+combine traffic swamps gains (case 13
  -25% vs tuned heuristic). Split choice MUST be the tuned arithmetic policy.
- 2-kv-heads-per-block to "fill 16 MMA rows": impossible for GQA (different kv heads
  need different K/V); rows waste is NOT the bottleneck (DRAM-latency bound).
- Split micro-tuning to local random cache_seqlens: noise; OJ distribution differs.

## What the current best (65.14) does - reproduce these exactly
1. 64 threads/block = one MMA wave. Grid = (ns, batch*num_heads_k). Each block handles
   one (batch, kv_head) split over `gqa` query heads (gqa = 32/num_heads_k).
2. Q: 8 tiles x 2 regs cached once (tile t reads dims t*16 + grp*4 .. +3, grp=lane>>4).
3. K smem: K_b[16][HEAD+2] padded row-major (row stride 130 halves = 65 words, odd).
4. V smem: **token-major V_b[16][HEAD+VPAD] with VPAD=8** (stride 136 halves = 68 words,
   ≡4 mod 32 so 16 consecutive tokens hit 16 different banks). Stored with ONE uint4
   store per 8 elements - NO transpose. THIS IS THE KEY WIN (see E4b).
5. PV read: for output dim d, gather 4 tokens t0..t3 (t0=grp*4) as 2x uint16 LDS each
   and pack: word0 = V[t0][d] | V[t0+1][d]<<16, word1 = V[t2][d] | V[t3][d]<<16.
   (Must match the transposed layout's packing exactly.)
6. Online softmax: merged-max, single alpha rescale (E5b). Tail page masked.
7. Split policy (arithmetic, no env): the per-(batch,kv,work_units) tuned policy in
   run_kernel - pages<=4->1; pages<=16&&wu>=32->3; kv8 mid pages->sqrt rule;
   wu<=8 (batch1)->64/90/148 by kv&pages; kv4 batch16 wu=64->22; kv8 wu=256->11;
   kv8 wu=128->5; kv8 wu=512/256/64-> mult 10/30/10.5 sqrt rule. THEN tokens_per_split
   = ceil(pages/ns)*16, recompute ns (balances per-batch short rows, never empty).
8. Fused single path when ns==1 (write output directly). Combine kernel otherwise,
   one thread per 2 dims, online pass, skip empty splits.
9. Compile: plain mxcc, NO mctlass/cute include needed (bare intrinsic works).
10. Tail page per split range: only the final partial page is masked; steady state does
    zero boundary work (E6). Never read block_table padding.

## Where the headroom is (for beating 65.14)
- All weak cases (7/10/12/13/14 and 9/11) sit at ~1.1-1.35x flash on OJ. flash itself is
  ~0.7 of DRAM peak on batch1 long-context cases -> the GPU is NOT saturated; the kernel
  is latency/occupancy limited, not bandwidth-limited.
- Occupancy ceiling: 64-thr/8KB-smem CTAs are smem-capped at 8 CTAs/SM (512 thr). Raising
  resident threads needs cutting smem/CTA (K in smem 4KB, V sourced differently) WITHOUT
  breaking the E4b vectorized-store/conflict-free-read win and WITHOUT crossing ~152 regs.
- Per-thread MLP across the page barrier without a register-expensive pipeline: issue the
  NEXT page's K/V vector loads into smem BEFORE the current page's __syncthreads barrier
  using a SMALL double-buffer (2 small tiles) that keeps 8+ CTAs/SM (E2 caution). This is
  the untested lever that the reference (~72) likely uses on the long sweeps.

Priority improvement directions (in order):
- Cut smem/CTA to raise occupancy for cases 10/13/14 (batch1, underfilled).
- Real MLP across the page barrier for cases 7/12/13/14 (long sweeps, latency-bound),
  respecting the register wall.
- Fix case 9's split (kv8 batch32 wu=256 wants ~ns 6-8 for 2 clean waves; current policy
  picks 11 -> verify against local bench case 9 only, don't disturb 6/12).

