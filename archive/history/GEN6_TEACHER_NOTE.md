# Teacher analysis after Gen6 (score 53.11)

## Key experiment (teacher, not leaked to students)
Forced case 12 (batch=8, seq=32768, nkv=8) to use restrained split ns=13
(orig's choice): result 852us - WORSE than gen6's 574us (ns~170).

## What this proves
gen2/gen6 kernel CANNOT process long page sequences efficiently in a single block.
It MUST over-split to compensate. orig achieves 494us at ns=13, meaning orig's
single block IS efficient at long sweeps.

Root cause hypothesis: gen6 does per-page [stage->__syncthreads->compute] serially.
orig (62.21) has an efficient long-sweep block (affine addressing, template
specialized full pages, likely no per-page barrier stall).

## Conclusion for teaching
Patching split counts will NEVER reach 62. The agent needs a kernel whose SINGLE
block efficiently streams many pages (hides load latency WITHIN the block) while
keeping occupancy. This is the structural skill to teach.

## Not leaked to skill
This analysis is teacher-only. The skill should pose it as a diagnosis question:
"On a long page sweep in one block, is the per-page [load->barrier->compute] serial
chain stalling? What happens to time if you process pages with the load for page i+1
issued before computing page i, WITHOUT doubling shared memory (registers or
load-ahead)?"

## Gen6 result
Score 53.11 (from 50.87). Case 9/11 improved via finer split. Case 12 unchanged (573),
structural. Saved as agent_gen6_kernel.cu.
