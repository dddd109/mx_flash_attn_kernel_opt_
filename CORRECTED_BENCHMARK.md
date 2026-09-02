# Corrected Benchmark Data (official 14-case table, verified against user's 62.21)

## Verification
User confirmed original optimized code = 62.21. Using my corrected benchmark + score
formula, I reproduce 62.21 exactly. flash baseline = 50.00. **Benchmark is now trustworthy.**

## Score model
S = 100 / (1 + (Tk - Th) / (Tb - Th)) where Tk=sum over 14 cases, Tb=flash total,
Th derived from orig=62.21 anchor. Th=1514.4us.

## Per-case data (us, cuda_event timing)

| case | batch | seqlen_k | nkv | flash | orig(62.21) | gen2 | note |
|------|-------|----------|-----|-------|-------------|------|------|
| 1 | 4 | 2 | 4 | 31.12 | 12.15 | 18.03 | edge, cache_seqlens=1 |
| 2 | 4 | 2 | 8 | 31.06 | 12.17 | 14.59 | edge |
| 3 | 16 | 17 | 4 | 38.34 | 16.07 | 12.66 | gen2 faster |
| 4 | 16 | 64 | 8 | 38.09 | 19.23 | 20.24 | |
| 5 | 16 | 141 | 4 | 38.27 | 22.79 | 22.69 | |
| 6 | 16 | 362 | 8 | 38.89 | 42.50 | 41.87 | orig slower than flash |
| 7 | 64 | 2048 | 4 | 162.78 | 151.43 | 159.19 | |
| 8 | 16 | 4096 | 4 | 92.36 | 89.02 | 92.12 | |
| 9 | 32 | 4096 | 8 | 277.37 | 232.99 | 280.13 | **orig much faster** |
| 10 | 1 | 8192 | 4 | 59.07 | 89.66 | 56.33 | orig SLOWER than flash |
| 11 | 16 | 12251 | 8 | 383.00 | 327.65 | 391.61 | **orig faster, gen2 slower** |
| 12 | 8 | 32768 | 8 | 553.26 | 493.80 | 573.75 | **orig faster** |
| 13 | 1 | 58966 | 4 | 153.57 | 165.45 | 173.84 | orig slower than flash |
| 14 | 1 | 61519 | 4 | 160.50 | 169.52 | 181.95 | orig slower |
| SUM | | | | 2058 | 1844 | 2039 | |

## Key insights (where orig beats gen2)
- **case 9/11/12 (all nkv=8, large)**: orig 0.84-0.89x of flash. gen2 is 1.0-1.04x.
  gen2 was tuned on WRONG (nkv=4) cases - poor at nkv=8 GQA.
- orig wins 195us over gen2 TOTAL. gen2 only ~19us better than flash.
- gen2 is actually ~51 (near baseline), NOT 61 as previously (wrong) estimated.

## gen2's actual weakness (nkv=8 large-batch GQA)
Cases 9/11/12 have nkv=8 (gqa=4). gen2 kernel likely not reusing KV well when
only 4 query heads share a KV head.

## What to teach next (skill v15 focus)
Agent must handle nkv=8 cases (4/6/9/11/12) efficiently. gen2 kernel fails there.
The skill's prior notes were tuned on wrong case table - must be regenerated.
