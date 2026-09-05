# Teacher experiment: H6 (merged-max softmax, drop beta) - ~1% gain

gen9 (page-max + alpha + beta) -> merged-max (single alpha, P relative to merged max):
| case | gen9 | H6 | delta |
|---|---|---|---|
| 9 | 255.6 | 254.3 | -1.3us |
| 12 | 572.6 | 568.9 | -3.7us |
| 8 | 92.0 | 91.5 | -0.5us |
| 7 | 158.6 | 157.5 | -1.1us |

Correctness passed. ~1% gain - confirms case 12 is DRAM-bound (SFU/exp not bottleneck).
Small but safe; could be shipped.

## Cumulative teacher experiments on gen6 structure
- H1 (full/tail loop split): ~1% (gen9)
- H5 (remove V transpose): FAILED -60% (transpose enables vector PV reads)
- H6 (merged-max softmax): ~1%
- split sweep: no effect beyond gen6's chosen ~96-170
=> gen6 structure is near its compiler-limited optimum on DRAM-bound cases.

## Why reference (494us) beats gen6 (572us) on case 12
Reference uses a COHERENT set of choices: affine lane mapping (token0=lane>>4, d0=(lane&15)*8,
iterate token+=4), Swizzle<3,3,3> K atom layout, XOR V swizzle, canonical MMA fragment
reads, merged-max softmax, page-balanced device-side splits. Individually porting any ONE
of these into gen6's layout fails or barely helps (proven H1/H5/H6). Reaching 494 needs
the whole coherent rewrite, which is beyond incremental teaching.
