# CONTEXT.md - MX FlashAttention paged GQA decode optimization

Goal: Optimize a paged-GQA flash-attention decode kernel for **MetaX C500** to beat
OJ score 65.14 (current best `submission_agentG_v2.cu`). Top of leaderboard 72.71.

## Situation (2026-09-04, NEW: GPU now available!)
- Prior sessions had NO GPU and optimized blind via OJ submissions. That changed:
  a **local MetaX C500 GPU is now available** (mx-smi works). Local loop = closed.
- Repo: /root/workspace/mx_flash_attn_kernel_opt (github dddd109, colleagues may push SOTA).
- Local loop verified: compile with mxcc -> .so, `full_verify.py` (14-case correctness),
  `bench_all.py` (our kernel vs flash_attn speedup, SPJ-confirmed case table).

## Case table (SPJ-confirmed - DO NOT use the stale one in SKILL.md/full_verify.py)
case: (batch, seqlen_cap, nkv)
1:(1,1,4) 2:(4,2,8) 3:(16,17,4) 4:(64,64,8) 5:(16,141,4) 6:(16,362,8)
7:(64,2048,8) 8:(16,4096,4) 9:(32,4096,8) 10:(1,8192,4) 11:(16,12251,4)
12:(8,32768,8) 13:(1,58966,8) 14:(1,61519,4)

## Current best local baseline (agentG_v2) vs flash_attn (us, speedup)
case 1:11.2/32.0=2.86 2:12.9/28.6=2.22 3:11.8/36.1=3.06 4:22.6/44.3=1.96
5:19.2/35.6=1.85 6:28.8/39.3=1.36 7:224.5/257.1=1.15 8:72.7/92.5=1.27
9:208.9/279.1=1.34 10:52.8/59.0=1.12 11:164.4/207.7=1.26 12:444.0/554.2=1.25
13:190.9/210.0=1.10 14:145.9/158.8=1.09  SUM ours=1610us
(Matches OJ profile pattern: cases 10/13/14 ~1.1x and 7/12 ~1.2x are the low-score cases.)

## Current step
Round-1 parallel subagent dispatch. Parent (me) benches contenders serially on the
shared GPU (only one device), merges winners. Each round uses FRESH agents with
limited context (task+skill+baseline+scripts only, no SOTA/teacher docs) so the runs
double as "skill teaching verification" for the contest.

## Local-sum to OJ-score proxy (measured)
- optimized_c500_flash_attn.cu (orig 62.21 OJ): local SUM = 1959.8us
- submission_agentG_v2.cu (65.14 OJ): local SUM = 1610.5us
=> In the useful range, OJ score rises ~ 1pt per ~50us of local SUM reduction
   (the two anchors: 350us <=> ~3pts). Judge candidates by SUM (mult=1).
- Baselines compiled: build/agentG_v2.so (1610us), build/orig62.so (1960us).
  (agentG_v2 is the 65.14 kernel; also present in worktrees as build/baseline.so)

## Constraints / gotchas
- Correctness must hold on all 14 cases: match>=0.99 (edge cases 1.0), no 8x outliers
  vs flash_attn. TOL = 1.6e-2*(1+|ref|).
- Kernel launched via `run_kernel` C symbol; blocks_per_batch = num_blocks/batch_size.
  Never read block_table padding; valid pages from cache_seqlens[b] only.
- Local tiny-case timing has a ~10-20us launch floor (higher than OJ's ~5us). Judge by
  speedup-vs-flash, not absolute us. Judge SUM for whole-kernel changes.
- Case 7 & 11 nkv differ between stale docs (kv4) and SPJ truth (kv8/7, kv4/11).
- Combine launches add ~launch floor; ns==1 fused path avoids it.
- Big case loads: q per (b,h); K/V per (page,kv_head). Long sweeps are DRAM-latency
  bound; batch1 cases are SM-occupancy/split bound.

## Open questions
- Best way to raise per-thread memory-level parallelism / pipeline long sweeps without
  register spills (mxcc spills past ~152 regs; 64 threads/block).
- Occupancy / larger CTA with 2 kv_heads per block (gqa rows 16 used fully) for
  kv8 case13-style underfill.
- Exact OJ cache_seqlens distribution (locally uniform currently); don't over-tune splits
  to local distribution.

## Recent decisions
- Use SPJ-confirmed case table in all local scripts (bench_all.py).
- Local bench = bench_all.py (single process, official iters, flash vs ours).
- Baseline = submission_agentG_v2.cu compiled to build/agentG_v2.so (ALL PASS, 1610us sum).
- Bench/full-verify python files were single-process-corrected; bench_cuda_event.py fine.
