# Benchmark Verification Status (honest assessment)

## The problem
The official OJ benchmark is opaque. My local benchmark was reconstructed from task.md's
case table, but that table's markdown rendering CORRUPTED the columns (empty cells ate
their tab separators). Attempts to reverse-engineer it hit fundamental ambiguity.

## What task.md reliably says
- 14 test cases exist. My local benchmark used 12 (missed cases 6 and 14).
- Global constants: num_heads=32, headdim=128, page_block_size=16, seqlen_q=1, causal=0.
- num_blocks = batch_size * ceil(seqlen_k / page_block_size). NO 3x redundancy.
- Every case: >=1 sequence pinned to capacity; if batch>1, >=1 sequence at length 1.
- cache_seqlens distribution: "fixed seed, synthesized" - exact distribution unknown.
- Correctness: >=99% within tol=1.6e-2*(1+|ref|), no >8x outliers. Edge cases (1/2/3):
  100% within tol.

## The corrupted case table (each row = tab-separated, empty cells removed)
Header (9 cols): case | batch | seqlen_k | cache_seqlens_range | num_heads_k | gqa_ratio | type | warmup | iters

case 1:  4 | 8 | edge | 3 | 100                  <- nkv/gqa missing (edge case)
case 2:  4 | 2 | 1~2 | 8 | 4                     <- batch=4 seq=2 nkv=8 gqa=4
case 3: 16 | 17 | 1~17 | 4 | 8                   <- batch=16 seq=17 nkv=4 gqa=8
case 4: 64 | 1~64 | 8 | 4 | perf | 50            <- batch=64, seqlen_k?? (likely 64), nkv=8
case 5: 16 | 141 | 1~141 | 4 | 8                 <- batch=16 seq=141 nkv=4 gqa=8
case 6: 362 | 1~362 | 8 | 4                      <- batch MISSING, seq=362, nkv=8 gqa=4
case 7: 64 | 2048 | 1~2048 | 12                  <- batch=64 seq=2048, nkv/gqa missing
case 8: 16 | 4096 | 1~4096 | 4 | 8 | 25          <- batch=16 seq=4096 nkv=4 gqa=8 warmup=25
case 9: 32 | 8 | 4 | 12                          <- ambiguous
case10:  1 | 8192 | 4 | 8 | 25                   <- batch=1 seq=8192 nkv=4 gqa=8
case11: 16 | 12251 | 1~12251 | 12                <- batch=16 seq=12251, nkv/gqa missing
case12:  8 | 32768 | 1~32768 | 8 | 4             <- batch=8 seq=32768 nkv=8 gqa=4
case13:  1 | 58966 | 25                          <- batch=1 seq=58966
case14: 61519 | 4 | 8                            <- batch MISSING, seq=61519 nkv=4 gqa=8

## Verifiable vs my local benchmark
CONFIRMED matching: cases 2,3,5,8,10,12 have same (batch,seqlen,nkv) as my benchmark.
AMBIGUOUS/UNKNOWN: case 1 (nkv?), 4 (seqlen_k?), 6 (batch?), 7 (nkv?), 9 (seqlen/nkv?),
11 (nkv?), 13 (nkv?), 14 (batch missing entirely).

## Where my benchmark may be wrong vs OJ
1. nkv for cases 1,7,11,13: I assumed 4. Could be 4 or 8 - GQA ratio differs.
2. Missed cases 6 and 14 entirely (they have batch values I couldn't recover).
3. cache_seqlens distribution: I used uniform random + pin max/min. Official generator's
   exact distribution is unknown. This affects whether cases are mostly-full (bandwidth)
   or mostly-empty (launch bound) - CRITICAL for performance character.

## Recommendation
The relative results (gen3 disaster, gen2/4/5 held) are likely robust because they
depend on coarse behaviors. But ABSOLUTE scores (50/61/63) are NOT trustworthy against
OJ. To validate, we need either: (a) the official benchmark harness, or (b) a
reconstructed table agreed upon from a cleaner source.
