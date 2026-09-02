# Teacher analysis after Gen8 - structural MMA limit reached

## Gen8 decisive findings
1. gen6 is actually 7 blocks/SM (gen7's "3" was its double-buffer variant). smem 8768B binds.
2. MMA 16x16x16 has ONE shared B operand (K) for all 16 M-rows. M-rows MUST share one
   kv_head's K. CANNOT pack different kv_heads or batches into one MMA instruction.
   => Option B (fill 16 rows across work units) is structurally impossible.
3. Option A (2 kv-heads per block): kv heads read different K slices, no smem shareable.
   Doubling smem -> 3 blocks/SM, worse.
4. Removing K smem -> 8 blocks/SM but per-thread 8B loads break coalescing -> -80%.
5. gen6 case 12 = 0.94 TB/s; ref = 1.09; device copy peak = 1.18. Gap is LATENCY.
6. Case 12 ref 494us = 1.09 TB/s with FEWER resident warps -> ref kernel streams with
   better memory-level parallelism per thread (deeper outstanding loads), not more blocks.

## Conclusion
Within the gen6 architecture (one kv_head group per 64-thread block, page-staged),
62 is NOT reachable by split/packing/occupancy tweaks. The reference likely:
- issues many more independent loads per thread (register-level MLP), or
- uses a wider MMA (M32) / different fragment mapping, or
- streams K and V in separate phases.

## Status
flash=50, best student kernel (gen6)=53.1. Target orig=62.21.
Scores progressed 50->53.1 across gen2..gen8 with correct benchmark.
Remaining gap is architectural, beyond quick iteration.

## Teacher recommendation
Options: (a) accept 53 as student ceiling from flash baseline, document thoroughly;
(b) give gen9 a MUCH longer budget (hours) to attempt the architectural rewrite;
(c) teacher provides ONE high-level architectural hint (e.g. "issue many independent
page loads before computing") - risks approaching answer leakage.
