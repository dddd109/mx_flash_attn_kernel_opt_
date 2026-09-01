# Generation 4 Results - Held the line at 61/63

## Behavior (the key lesson this generation demonstrated)
- Strictly followed the "DO NOT REGRESS" prime directive.
- Independently verified the skill's prior-student notes (confirmed: on this kernel
  structure, occupancy > smem prefetch).
- Attempted double-buffered smem pipeline -> long-seq regressed +30-40% -> REVERTED.
- Final kernel is byte-identical to gen2 (61/63 preserved).
- Completed within 20-minute time budget.

## Confirmed performance (gen2/4 kernel, best_of_3)
| case | shape | opt(us) | flash(us) | ratio |
|------|-------|---------|-----------|-------|
| 1 | 4/8/4 | 12.4 | 29.9 | 0.41 |
| 2 | 4/2/8 | 12.1 | 30.5 | 0.40 |
| 3 | 16/17/8 | 13.0 | 37.2 | 0.35 |
| 4 | 64/64/4 | 19.4 | 37.7 | 0.52 |
| 5 | 16/141/8 | 23.8 | 37.0 | 0.64 |
| 7 | 64/2048/4 | 159.3 | 162.0 | 0.98 |
| 8 | 16/4096/4 | 92.1 | 90.8 | 1.01 |
| 9 | 32/8/4 | 12.7 | 30.3 | 0.42 |
| 10 | 1/8192/4 | 57.0 | 56.9 | 1.00 |
| 11 | 16/12251/4 | 218.5 | 205.4 | 1.06 |
| 12 | 8/32768/4 | 334.5 | 320.2 | 1.04 |
| 13 | 1/58966/4 | 174.1 | 157.2 | 1.11 |

Score estimate: 61/63 (unchanged, preserved).

## What this proves about the skill
- The v13 restructure (methodology + prior-student notes) is working:
  - Student verifies notes independently, doesn't blindly trust them.
  - Mandatory regression rule prevented a repeat of gen3's catastrophe.
  - Student correctly identified why their own improvement failed (occupancy).
- Remaining honest gap to 63: gen2 kernel is at its local optimum; needs a
  fundamentally different idea (e.g. register-based K/V pipeline without smem
  growth) to break past 61.

## Next ideas for gen5 (recorded for the next student, not answers)
- Register-resident K/V staging that hides latency WITHOUT doubling shared memory
  (occupancy-preserving pipeline). Previous attempts grew smem and crushed occupancy.
- Persistent-CTA or cooperative-groups approach to reduce per-launch overhead.
- Revisit case 12/13 (batch=1 long) with a sweep of ns in a NARROW range around the
  current optimum to find any remaining few-%.
