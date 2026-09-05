# Generation 3 Regression Analysis

## Summary
Gen3 (independent agent) regressed score from 61.0 to 18.7 by breaking long-seq pipeline
while trying to fix case 9.

## Per-case: gen2 vs gen3 (best_of_3, us)
| case | gen2 (61pt) | gen3 | delta | impact |
|------|------------|------|-------|--------|
| 1 | 11.79 | 15.74 | +4.0 | regression |
| 2 | 11.75 | 17.18 | +5.4 | regression |
| 3 | 13.99 | 14.96 | +1.0 | regression |
| 4 | 20.74 | 27.74 | +7.0 | regression |
| 5 | 24.81 | 30.66 | +5.9 | regression |
| 7 | 158.26 | 247.46 | +89.2 | regression |
| 8 | 92.44 | 135.43 | +43.0 | regression |
| 9 | 22.60 | 16.23 | -6.4 | FIXED |
| 10 | 55.59 | 74.59 | +19.0 | regression |
| 11 | 218.18 | 334.15 | +116.0 | regression |
| 12 | 332.94 | 508.16 | +175.2 | regression |
| 13 | 173.15 | 256.04 | +82.9 | regression |
| TOTAL | 1136 | 1678 | +542 | -42 points |

## Root cause
Gen3's "partial-page specialization" for case 9 replaced the register-prefetch software
pipeline with a simple break-based stage_page loop. This removed latency hiding that long
sequences critically depend on. Long-seq regressed ~500us total to gain ~6us on case 9.

## Lessons for skill (CRITICAL)
1. **Never sacrifice the latency-hiding pipeline** for a launch-floor case.
   Long sequences are the bulk of the workload; short cases are launch-bound already.
2. **ALWAYS run full 12-case regression** after ANY change. A single-case win is
   worthless if it regresses others.
3. Case 9 was already near its launch floor (~16us). Squeezing to 13us = ~10us total.
   Regressing long seq by 500us is catastrophic.
4. **Best result remains gen2 kernel at 61.0/63.** Saved as agent_v8_gen3_best_gen2_kernel.cu.
