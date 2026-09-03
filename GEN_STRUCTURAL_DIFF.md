# Structural diff: Reference (62.21) vs student kernel (~53) - teacher analysis

A complete line-by-line diff was produced (see task output). The DECISIVE differences,
with teacher judgment on which are worth teaching:

## 1. Split strategy is NOT the gap (PROVEN)
Forced-split sweep on student kernel case 12: ns=32..256 all give 568-660us.
Best ~96 = 568us. Reference achieves 494us. Reference's split is fixed ~32 (forced
const 128 clamped by capacity). Student kernel NEVER reaches 494 at any split count.
=> The gap is per-block/per-thread efficiency, not work partitioning.

## 2. V shared-memory layout & transpose cost (STRONG candidate)
- Student: transposes V at load time: for each uint4 (8 bf16), 8 SCALAR smem stores
  `Vt[dim+e][tok]=vv[e]`. Per thread per page: 4 uint4 x 8 = 32 scalar scatter stores.
  2048 pages => 65536 scalar stores/thread. Padded layout, no swizzle.
- Reference: keeps V token-major (as in gmem), NO transpose, reads with a d_xor swizzle
  index `v_smem_index(token,d) = token*128 + (d ^ (token<<3))` that spreads PV reads
  over banks WITHOUT transposing.
- Student pays transpose on EVERY page store; reference pays only an index XOR on read.
  On a 2048-page DRAM-heavy sweep this is per-page smem store overhead + extra latency.

## 3. Softmax bookkeeping (minor)
- Reference: merged max first, weights relative to merged max, single alpha rescale,
  no beta. 1 exp for alpha + 4 exp for weights.
- Student: page max, alpha=exp(m-mnew), beta=exp(mx-mnew), P*beta into PV. 2 exp
  (alpha,beta) + 4 exp. Extra __expf per page + explicit beta multiply per element.
  On 2048 pages this is 2048 extra expf + multiplies per thread. Minor but real.

## 4. MMA operand-slot order (both call same HW op)
- Reference calls MmaOp::fma(Q_as_A, K_as_B). Student calls builtin(K_as_arg1, Q_as_arg2).
  Since C500 MMA may be symmetric in operand order for this shape, probably equivalent.
  Not prioritized.

## 5. Tail-page loading (minor)
- Student tail: unconditional 16-token load + runtime mask. Reference: predicated
  zero-filled loader. On mostly-full cases tail is 1 page of 2048. Minor.

## 6. QK smem swizzle vs padding (secondary)
- Reference: K in arch atom layout Swizzle<3,3,3> (2x 16x64 atoms). Student: padded
  row-major. Both avoid bank conflicts. Reference's is the "canonical" atom layout.

## Teacher recommendation for next teaching round
The #1 candidate to teach: ELIMINATE THE PER-PAGE V TRANSPOSE. Instead of transposing
V into dim-major on store, keep it as-loaded (token-major) and make the PV MMA read
use a swizzled index (XOR token bits into dim). This removes ~32 scalar smem stores per
thread per page from the critical load path. Requires rethinking the PV B-fragment
gather but that's the reference's approach.
