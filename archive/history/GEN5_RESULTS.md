# Generation 5 Results - Held 61, clean negative results

## Behavior
- Followed no-regress directive; preserved baseline (kernel byte-identical to gen2/61).
- Used the v14 HW-agnostic diagnosis framework to form and TEST two hypotheses.
- Both were clean negative results (tested, measured, reverted):

### Negative result 1: partial-page optimization
Hypothesis: loading full 16-token pages wastes bandwidth on partial pages.
Finding: short cases are launch-floor-bound (invisible there); long cases waste <=0.5%
on ragged tail. NOT a win. Confirmed by measurement.

### Negative result 2: block_table prefetch
Hypothesis: per-page block_table dependent load serializes memory stream.
Finding: pid load is a broadcast (~1 transaction), pages physically consecutive
(HW prefetch hides address stream), occupancy already covers latency. Adding
shared-mem round-trip + barrier REGRESSED cases 11/12/13 by 1-3%. Reverted.

## Valuable forward directions (gen5's diagnosis, recorded for next student)
1. **Case 13 bandwidth puzzle**: batch=1, seq=58966 tops at 0.7 TB/s while case 11
   (batch=16) reaches 1.8 TB/s. Same kernel; difference is per-SM sequential-page
   locality. 512 blocks / 104 SMs ≈ 5 blocks/SM. Hypothesis: transpose grid so
   adjacent SMs work on interleaved pages of the same sequence for better DRAM
   page hits.
2. **Combine-phase cost**: cases 11/12 write ~32KB×ns partials + re-read in combine.
   A split within one kernel (cross-block atomics / second-level reduction grid,
   no full re-read) might close 5-10%. Risky - guard with regression rule.
3. Split-count for case 13 swept 64-1024: 128 is optimum. Not a lever.

## Status
- Final kernel = gen2 kernel (61/63). Correctness 100% all cases.
- Score ~61 (short/medium banked at 0.31-0.65x flash; residual is cases 11/12/13 at 1.05-1.11x).
