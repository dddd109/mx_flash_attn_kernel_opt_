# Agent Generations Feedback Integration

Consolidated findings from 5 generations of independent student agents optimizing
paged GQA attention on MetaX C500. Each generation had NO answer access and worked
only from the skill + task spec.

## Generation progression

| Gen | Skill | Score* | Key outcome |
|-----|-------|--------|-------------|
| 1 | v5 (leaky) | 51 | Overcame baseline (50); used MMA, split-KV, fused short path |
| 2 | v9-v10 | 61 | Removed host overhead (getenv/atoi); all short cases at launch floor |
| 3 | v11 | 18.7 | CATASTROPHIC regression: fixed case 9, broke long-seq pipeline |
| 4 | v13 | 61 | Held line; verified skill restructure; reverted risky change |
| 5 | v14 | 61 | Two clean negative results; identified real bottleneck (case 13 DRAM) |

*Scores are ESTIMATES from an AUTHORITATIVE-BUT-UNVERIFIED local benchmark. See
benchmark caveat below - they may not match the official OJ.

## Consolidated technical findings (across generations)

### What works (reliably reproduced)
1. **MMA via raw intrinsic** (`__builtin_mxc_mma_16x16x16bf16`) - NOT `load_matrix_sync`
   which crashes on C500 (toolchain bug, wrong address pattern in emitted IR).
   Fragment layout reverse-engineered empirically: A[t&0xf][(t>>4)*4 + r], B similar,
   C = A·B^T.
2. **Register-cached Q fragments** (8 MMA steps × 2 regs) - avoids re-reading Q per page.
3. **Warp-shuffle softmax stats** (`__shfl_xor_sync` width 16/32) instead of shared-mem
   roundtrips + barriers.
4. **Fused single-split path** (ns==1 writes output directly, no combine launch) -
   essential for short/tiny cases.
5. **Adaptive page-aligned split count** - pure arithmetic policy, NO getenv/atoi on
   hot path (host overhead cost ~7us on tiny cases).
6. **High occupancy** (~7 blocks/SM, lean shared memory ~8.6KB/block) hides DRAM latency.
7. **Cached partial buffers** (static, grow-only) - no cudaMalloc per call.

### What was DISPROVEN (clean negative results - save future time)
1. **Partial-page specialization** (gen5): not a win. Short cases are launch-floor
   bound; long cases waste <=0.5% on ragged tail.
2. **block_table prefetch to smem** (gen5): regressed 1-3%. pid load is a broadcast,
   HW prefetch already covers the address stream, occupancy covers latency.
3. **Register-level double-buffer prefetch** (gen4): regressed long-seq 30-40% because
   doubling smem crushed occupancy (17KB → 3 blocks/SM).
4. **num_splits=1 for long seq** ("long-seq trap", gen5): wrong for this kernel
   structure. Case 7 optimum ns≈13, case 8 optimum ns≈43, case 13 optimum ns≈128.
   Split-count is structure-dependent - sweep it.
5. **Warp shuffle absent on C500**: WRONG (early false belief). It works fine.

### Real bottleneck (gen5 diagnosis, still open)
- **Case 13 (batch=1, seq=58966) tops at ~0.7 TB/s** while case 11 (batch=16, seq=12251)
  reaches ~1.8 TB/s. Same kernel. Difference = per-SM sequential-page locality.
  512 blocks / 104 SMs ≈ 5 blocks/SM. Hypothesis: transpose grid so adjacent SMs work
  on interleaved pages of one sequence → better DRAM page hits. UNTESTED.
- **Combine-phase cost**: long cases write ~32KB×ns partials to global + re-read in
  combine kernel. An in-kernel second-level reduction (cross-block atomics) might close
  5-10%. UNTESTED, risky.

### The regression catastrophe (gen3) - permanent lesson
Fixing ONE launch-floor case (case 9, worth ~10us) by removing the long-seq pipeline
lost ~500us across long cases → 61 → 18.7. RULE: after every change, run ALL cases;
any long-seq regression >5% ⇒ revert. This is now a mandatory skill rule.

## Benchmark CAVEAT (IMPORTANT - read before trusting any score)

The score estimates above come from a LOCAL benchmark reconstructed from task.md's
test-case table. The table in task.md has corrupted/misaligned columns (markdown tab
rendering), so:
- 14 cases exist but the local benchmark used 12 (missed cases 6, 14)
- Some (batch, seqlen_k, num_heads_k) triples may be misread (e.g. num_heads_k vs
  gqa_ratio confusion)
- cache_seqlens distribution was random (uniform in [1, seqlen_k]) with [0]=max,
  [1]=1 pinned - the official generator's exact distribution is unknown ("fixed seed",
  at least one at capacity, batch>1 also at least one at length 1)

Therefore: the RELATIVE ordering of generations (gen1<gen2≈gen4≈gen5, gen3 disaster)
is likely robust, but the ABSOLUTE score estimates (50/61/63) may not match OJ.
The real score requires official benchmark. Recommend: reconstruct a cleaner benchmark
from the verifiable parts of task.md before making further claims.
