# Teacher experiment: H5 (eliminate V transpose) - FAILED on gen6 structure

## Experiment
Modified gen9 kernel: keep V token-major (single vector store, like K), change PV read
from transposed-contiguous (2 uint32 loads) to token-major gather (4 scalar uint16 reads
across token rows, packed into operand words). Correctness passed (match 1.0).

## Result
| case | gen9 (transposed V) | H5 (gather V) |
|---|---|---|
| 9 | 252us | 398us (-37%) |
| 12 | 572us | 916us (-60%) |

## Why it failed
Transposing V trades store cost for FAST PV reads (2 contiguous uint32 smem loads/tile).
The gather version needs 4 STRIDED scalar uint16 reads/tile. On this compiler the scalar
gather is far more expensive than the transpose stores it saves. The reference kernel can
use a gather because its XOR swizzle makes the gather bank-conflict-free AND its codegen
handles it well - it's a full-layout synergy, not an isolated win.

## Teacher conclusion
H5 as an isolated change HURTS the gen6-style kernel. Do NOT teach it as a standalone
optimization. The transpose in gen6 is load-bearing for its PV read efficiency.

## Revised view of the 53->62 gap
gen6's structure (transposed V + vector PV read) is internally consistent and near its
compiler-limited optimum (~53). Reaching 62 likely needs the reference's WHOLE layout
(Swizzle<3,3,3> K atoms + XOR V + canonical atom index math + merged-max softmax) - a
coherent rewrite, not incremental patches. Incremental teaching has hit diminishing
returns at ~53.
