# Teacher analysis after Gen7 (score ~53, no gain but deep diagnosis)

## Gen7 findings (high value)
1. **Occupancy is only 3 blocks/SM (9.4%)** on gen6 kernel. Limiter = shared memory
   8768B/block (65536B/SM -> 7 blocks by smem, but 154 regs/thread -> more; min binds at 3?
   Actually 7 by smem; gen7 measured 3. Need to reconcile). Low occupancy = few warps
   to hide DRAM latency in long sweeps.
2. Removing the smem +2 padding -> 3x WORSE (bank conflicts; padding is load-bearing).
3. __syncthreads->__syncwarp: neutral.
4. launch_bounds/reg caps: neutral (smem binds, not regs).
5. Double-buffer prefetch: hides sweep-length sensitivity (flat 711us across ns) BUT
   caps at 0.82 TB/s < fine-split 0.99 TB/s. 25% worse overall.
6. Not bandwidth saturated (86% of peak), not compute bound (removing MMA made it worse).
7. ns sweep confirms auto (~170 for case 12) is locally optimal.

## The real unlock (gen7's hypothesis, teacher agrees directionally)
gen6's block is too heavy (smem 8.7KB) -> 3 blocks/SM -> latency-bound long sweeps.
The 62-scorer sustains peak BW with fewer resident warps -> structurally different
per-block work (e.g. 2 kv-head groups share one staged page; or much smaller footprint
so 6-8 warps/SM).

## Status
Scores: flash=50, gen2=50.9, gen6=53.1, gen7~53. Target 62.21.
The path to 62 requires a kernel whose blocks are SMALL enough for high occupancy AND
stream pages efficiently. Neither split tuning nor double-buffering alone achieves it.

## Teacher decision
Do NOT leak "the answer". Skill v17 should pose: "your block uses 8.7KB smem giving
only 3 blocks/SM. What if one block served TWO kv-head groups from ONE staged page
(sharing the load, halving smem per head-group)? Or reduce per-page smem so 6-8 blocks
fit? Measure occupancy and case 12 time for each."
