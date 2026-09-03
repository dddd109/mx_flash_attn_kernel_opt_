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

## The Benchmark (authoritative)
14 cases. warmup=3. Score model: flash=50, strong kernel ~62.
| case | batch | seqlen_k | nkv | type | iters | flash(us) |
|------|-------|----------|-----|------|-------|-----------|
| 1 | 4 | 2 | 4 | edge | 100 | 31.1 |
| 2 | 4 | 2 | 8 | edge | 100 | 31.1 |
| 3 | 16 | 17 | 4 | perf | 100 | 38.3 |
| 4 | 16 | 64 | 8 | perf | 50 | 38.1 |
| 5 | 16 | 141 | 4 | perf | 50 | 38.3 |
| 6 | 16 | 362 | 8 | perf | 50 | 38.9 |
| 7 | 64 | 2048 | 4 | perf | 12 | 162.8 |
| 8 | 16 | 4096 | 4 | perf | 25 | 92.4 |
| 9 | 32 | 4096 | 8 | perf | 12 | 277.4 |
| 10 | 1 | 8192 | 4 | perf | 25 | 59.1 |
| 11 | 16 | 12251 | 8 | perf | 12 | 383.0 |
| 12 | 8 | 32768 | 8 | perf | 12 | 553.3 |
| 13 | 1 | 58966 | 4 | perf | 25 | 153.6 |
| 14 | 1 | 61519 | 4 | perf | 25 | 160.5 |

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

## VERIFIED 64.57-POINT CONFIGURATION (a student reached this on the real OJ)
This is a working configuration distilled from a real OJ submission (64.57 pts, beats
the 62.21 reference). Treat it as a validated starting point to IMPROVE, not copy code.

Architecture that scores ~64 on the real OJ:
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
6. Online softmax: merged-max, single alpha rescale (E5). Tail page masked.
7. Split policy (arithmetic, no env): pages<=4 ->1; pages<=16 && wu>=32 ->3;
   wu<=8 -> 64 if pages<=1024 else (batch==1 && kv==8 && pages>2000 ? 64 : 128);
   else mult=(kv==8&&wu>=64&&pages>=200)?30:20, ns=round(mult*sqrt(pages*wu)/wu).
   THEN tokens_per_split = ceil(pages/ns)*16, recompute ns.
8. Fused single path when ns==1 (write output directly). Combine kernel otherwise,
   one thread per 2 dims, online pass, skip empty splits.
9. Compile: plain mxcc, NO mctlass/cute include needed (bare intrinsic works).

Real-OJ per-case profile of this config (targets to beat):
case13 (batch1 kv8 seq58966) is the weakest: ~1.03x flash. The kv8 large cases
(7:1.25x, 9:1.37x, 12:1.40x) and case 6 (1.61x) have headroom. Edge 1-3 ~5.4x are
near launch floor.

Priority improvement directions (untested by this config):
- case 13 (batch=1, kv=8): only 8 (batch,kv) units -> underfilled SMs. Try ns sweep
  again near 64, or 2 kv_heads per block (gqa=8 total, 8 MMA rows used of 16) to
  halve block count and double rows per block. smem stays ~same if you load one page
  and read both kv heads' slices (they are different 128-col slices of the page).
- Larger blocks (128 threads, 2 waves) sharing one staged page for kv8 cases.
- Tune the mult/nv heuristics per (batch, kv) class with the real case table.
