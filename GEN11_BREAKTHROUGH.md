# GEN11 BREAKTHROUGH: 73.9 score (from 54.2) - V transpose removal

## The winning change (teacher-identified, student-implemented)
gen10 transposed V into dim-major shared memory at load time using EIGHT scalar
stores per 16-byte vector (32 scalar scatter-stores per thread per page). A
high-scoring kernel keeps V token-major and stores it with ONE vectorized store.

gen11 stores V token-major: `V_b[token][HEAD+VPAD]` with VPAD=8 (row stride 136
halves = 68 words, ≡4 mod 32), single uint4 store per 8 elements. PV read gathers
2 uint16 LDS loads per operand word (V[t0][d] low, V[t0+1][d] high) reproducing the
exact operand packing. Correctness bit-exact on all 14 cases.

## Results (conservative measured)
| metric | value |
|---|---|
| case 12 | 574 -> 487us (ref 494) |
| case 9 | 256 -> 217us (ref 233) |
| case 10 | 62 -> 52us (ref 90) |
| case 13 | 173 -> 147us (ref 165) |
| TOTAL | 1973 -> 1706us |
| SCORE | 54.2 -> 73.9 (ref 62.21) |

## Why it worked
8x fewer shared-memory store instructions on the per-page hot path. V transpose was
the hidden per-page cost that split-tuning, occupancy, softmax, and MLP experiments
never addressed (they all kept the transpose).

## Remaining gap (for next gen)
Short/edge cases 1-3 (batch=4, seq<=2): gen11 18-21us vs ref 12-16us. Launch
overhead dominated. ~30us total available.
