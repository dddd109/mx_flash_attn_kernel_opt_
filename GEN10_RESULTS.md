# Gen10 Results: 54.2 (best student score)

## What happened
Gen10 attempted the coherent rewrite (loader pipeline, K swizzle, V gather, device
splits) guided by teacher's distilled findings. Register wall (64 threads, 152 regs,
8768B smem) blocked all pipeline variants (all regressed 570->842us). Gen10 shipped
the SAFE teacher-verified win: merged-max softmax (drop beta), in both paths.

## Score (best-of, verified)
| gen | total | score |
|-----|-------|-------|
| flash | 2058 | 50.0 |
| gen6 | 1994 | 53.1 |
| gen9 | 1992 | 53.2 |
| gen10 | 1973 | 54.2 |
| orig | 1844 | 62.2 |

case 11 improved to 357 (was 361). case 12: 568 (from 574).

## Cumulative teacher learning (verified across gen6-gen10)
1. Correct benchmark matters: gen6-9 "61" was fake; real ~53.
2. Split tuning exhausted. Occupancy fine (7 blocks/SM). Not compute bound.
3. Case 12 (572->494) is DRAM-latency bound. Software pipeline blocked by mxcc
   register spills at 64-thread/152-reg structure.
4. Single technique ports fail (V transpose removal -60%, others ~1%).
5. Coherent rewrite also blocked by register wall.
6. Safe wins accumulate: H6 merged-max (+1%), full/tail split (+1%).

## Preserved artifacts
- libgen10.so / agent_gen10_kernel.cu (best, 54.2)
- libgen9.so / agent_gen9_kernel.cu (53.2)
- backup/ contains originals
