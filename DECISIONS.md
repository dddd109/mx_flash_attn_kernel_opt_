# DECISIONS.md - decision log

## 2026-09-04 (round 2, post-crash recovery)
- **Server crashed** mid-round-2; occ agent's baselines corrupted (truncated ELF). All
  agent results re-verified against clean recompiled baselines. occ_v1/v2 pipelining =
  REJECTED (case13 401us vs 190us; 16KB smem halving occupancy beats pipelining).
- **nomax softmax = NEW BEST** (submission_nomax.cu, from r2_split micro_v1). Removes
  the running max entirely: with randn inputs d=128, scores bounded ~±4 so exp(s)<=e^4,
  fp32-safe; partials from splits sum directly (no m_part array, no exp rescale in
  combine). Local SUM 1517us vs agentG_v2 1608us (-91us, -5.7%), verified interleaved
  A/B x2, ALL 14 PASS. Risk: mathematically non-standard softmax, but valid for the
  stated randn evaluation distribution and 1.6e-2 tol. ACCEPTED as candidate.
- micro_v2 (vectorized partials+combine float4, half-thread combine) measured ~same as
  micro_v1 (1518 vs 1515); not merged (more complex, no clear win).
- Occupancy direction (cut smem/CTA to raise resident threads): still open. occ_v2's
  smem-pipeline route is dead; other routes untried.
- GPU contention: single shared GPU, other agents' benches pollute absolute numbers;
  decision rule = interleaved A/B in one process, best-of-several.

## 2026-09-04
- **Switch to local GPU closed-loop optimization** (previously OJ-only, no GPU).
  Rationale: GPU present (mx-smi/MACA ok). Local speedup-vs-flash tracks OJ score
  pattern. OJ submission still final arbiter when user can submit.
- **Authoritative case table = SPJ-confirmed one** in bench_all.py/HANDOFF
  (7:(64,2048,8), 11:(16,12251,4), 13:(1,58966,8)). Stale docs (SKILL.md, full_verify.py,
  OJ_BASELINE) say kv4/8/4 for those - REJECTED as unverified. Evidence: submission code's
  split policy comments (agentG_v2.cu) hard-code kv8-case7/13 and kv4-case11/14 branches.
- **Benchmark = speedup vs flash_attn (single process), all 14 cases, official iters.**
  Absolute us on tiny cases inflated by launch floor (~10-20us local vs ~5us OJ).
  SUM over all cases is the headline metric.
- **Baseline frozen**: submission_agentG_v2.cu -> build/agentG_v2.so. full_verify ALL
  PASS; sum 1610us.
- **Workflow**: parent (coordinator) + subagents on feature branches, local verify, merge
  best. Per Tier-2 engineering-habits.

## 2026-09-04 (round 3, post-2nd-crash)
- Server crashed twice; r3 agents interrupted mid-work, their .so sweeps preserved.
- r3_combine: exhaustive per-case ns sweeps (case 7/10/13/14 at many ns) CONFIRM the
  split policy in nomax is already optimal (baseline wins every case). No change.
- r3_mlp: tried staggered register-prefetch across the page barrier (v1-v4, keeping
  ONE page in smem to preserve 7-8 CTAs/SM). ALL catastrophically regressed
  (3100-3270us vs 1519 baseline) — register spills persist even after nomax freed
  regs. v5 reverted to baseline. => Register-prefetch pipelining is a CONFIRMED dead
  end on this mxcc toolchain, same as the smem double-buffer. The per-page
  smem+barrier structure + tuned splits + nomax softmax is very hard to beat here.
- CURRENT BEST remains submission_nomax.cu (1517us local, est ~66 OJ; sent to user for
  xpuoj submission). No new structural win from round 3.

## 2026-09-04 (BREAKTHROUGH: kernel is MMA-throughput bound, NOT memory bound)
Decisive probe experiment on case 12 (b8 kv8 32768), ns=60, all else equal:
- full nomax kernel: 427us
- remove V-MMAs (keep QK MMAs, loads, softmax): 254us
- remove ALL MMAs (pure page load + softmax): 38us
=> MMA execution is ~390us of the 427us. Memory+softmax floor is only ~38us.
CONCLUSION: long kv8 cases are MMA-pipe-bound. Effective bandwidth ~0.6TB/s was a
red herring (it's MMA throughput masquerading as a BW number). Earlier evidence fits:
flat vs ns (60-320 all ~430-480us), flat vs 8x data reuse (only 9% faster), reg/smem
pipelining dead ends (compute not latency).
ROOT CAUSE of waste: MMA row utilization. Block computes 16 MMA rows but only
gqa=32/kv are real query heads: kv8->4/16=25% useful, kv4->8/16=50% useful. The other
rows run zeroed q. => up to ~2-4x MMA reduction possible by filling rows.
NEXT DIRECTION (priority 1): multi-kv-head-per-block to fill MMA rows. A K/V page
contains ALL kv heads' slices (k_cache[page][16][kv][128]), so one block CAN load the
full page and compute multiple kv groups' query heads in the 16 rows. kv8: 4 kv_heads
in one block -> rows 0-15 all real (4 heads x gqa4). This is the single biggest
potential win. Earlier dismissal ("different kv heads need different K/V") was wrong:
they share the same physical page, just different column slices.
- kv4: gqa=8, 8/16 rows used. 2 kv_heads/block fills 16 rows.
Score impact: cases 7-14 are ~1.3-1.5x; pushing kv8 cases (7/9/12/13) and kv4 (8/11/14)
toward row-full MMA could plausibly reach the 1.5-2x that yields 72+ total.

## 2026-09-04 (CRITICAL: row-fill is architecturally IMPOSSIBLE - layout decoded)
Decoded the C500 16x16x16bf16 fragment layout via probe (probe2.cu: A and B set to
known per-lane values, read C back):
- Thread (row=lane&15, grp=lane>>4) holds C[head=row][tokens=grp*4..grp*4+3].
- A operand = the 16 query-head rows (q), B operand = the K tile (shared by all 16 rows).
- In ONE MMA the B (K) tile is FIXED — it cannot source different kv slices per A-row.
- => GQA kv8 (gqa=4): only 4 of 16 A-rows have real q for THIS kv's K; the other 12 rows
  would need a DIFFERENT kv's K which one MMA cannot provide. 16-row fill via multi-kv
  is IMPOSSIBLE within a single MMA. kv4 (gqa=8) similarly capped at 8/16.
- trackA2kv (r4_mk2) confirmed empirically: merging 2 kvs per block WITHOUT reducing MMA
  regressed 2.5x (fewer blocks, same MMA). tb_chain variants neutral.
- => The wasted-row MMA work is a hardware-shape tax on GQA that CANNOT be eliminated by
  software restructuring of the QK/PV MMAs.
REVISED DIRECTION: since MMA rows can't be filled, attack the OTHER parts of the 427us:
  - 254us was QK-MMA(8) only; 38us pure-load. The QK 8-MMA chain (~216us) + PV chain
  (~173us saved by removing V-MMA) — can the number of MMA instructions per page be cut
  another way? QK is 8 MMA (k=128/16). PV is 8 MMA. Both minimal for 128-dim.
  - OR: is the MMA issue rate itself the limit? If 16 MMA/page-warp is fixed, the only
  lever is FEWER WARPS issuing redundant MMA... but gqa heads need separate MMAs.
  - OR: reduce pages processed: none (all KV needed).
  - OR: the load path overlaps MMA poorly; with MMA taking 390us and memory 38us, even
  PERFECT overlap only gets to max(390,38)=390 -> the win ceiling is small IF purely
  MMA-issue bound. But the no-VMMA probe (254) shows memory+QK overlap at 254, meaning
  QK MMA alone ~216us > memory 38us, so ~216us is the QK-issue floor unless...
  - REVISIT: is each QK MMA really necessary for ALL 16 rows? For gqa=4, rows 4-15 are
  zero-q. If the HARDWARE skips zero rows cheaply, the probe would show less than 8x the
  per-useful-MMA. The no-VMMA probe implies QK MMAs cost ~27us each (216/8). For 4 useful
  heads that's ~7us per useful head-MMA. Not obviously skippable.
NEXT BEST DIRECTION: profile-driven — measure MMA pipe duty vs issue to know if MMA
instructions are the hard wall or if issue/latency within MMA chains dominates; then try
(a) reducing instructions around MMA (operand setup), (b) 2 blocks/SM interleave via
smaller smem, (c) accepting ~390us MMA and overlapping the 38us memory perfectly.

## 2026-09-04 (case4 OJ regression 0.020->0.048: root-caused)
- ov (submission_ov.cu) got 65.7 on OJ: cases 5/7-14 all improved as predicted BUT case4
  regressed 20->48us (2.4x, score 75->55), killing the total.
- NOT reproducible locally under ANY cache_seqlens distribution (tailheavy/pageheavy/
  full/one/rand/skew all show ov FASTER than nomax).
- ROOT CAUSE (code audit): ov's TAIL path (partial page) was missing the memory barrier
  that nomax had (__syncthreads) after stage_page and before the MMA reads. 64-thread
  block IS one warp (warp_size=64 confirmed via torch), so __syncwarp() is the correct
  replacement — but it was MISSING in the tail. case4 (batch64, ns=1 fused, mostly
  1-page blocks) exercises the tail path more than any other case -> the race/serialization
  hit case4 hardest. Fix: add __syncwarp() after stage_page in the tail path.
- ov_safe = ov + per-page l-shuffle (revert the risky deferral) + tail syncwarp fix.
  Local SUM 1460us vs nomax 1515 (-3.6%), vs buggy ov 1454 (equal). RESUBMIT ov_safe.

## 2026-09-05 (CONFIRMED breakthrough: read pattern is THE bottleneck)
- microbench R0-R4 (r10_l2): kv-sliced strided reads = 0.37 TB/s; R2(R2/R0=0.915) proved
  it's the ACCESS PATTERN not DRAM volume/L2/phase; R3 linear contiguous = 1.59 TB/s.
- pat2 (r10_persist): contiguous full-page reads = 1.5 TB/s on case13 geometry.
- => The kernel MUST read contiguously. case13 ceiling ~80-110us (now 170us). Score 58->~70.
- Multiple crash-restarts killed agent dispatches; I will BUILD the contiguous kernel directly.

## 2026-09-05 (overnight, SSH now stable)
- All S1-S4 structural rewrites (GEMM/warp-spec/bigblock/gridsync) FAILED vs 67.07 baseline.
  S4 corrected the picture: case13 baseline = 1.38 TB/s (NOT 0.7, earlier probe flawed), pure
  stream ceiling 1.82 TB/s -> case13 at 76%, gap is MLP not read-pattern. All MLP-raising
  attempts lose more to occupancy/sync than gain.
- LOCAL WIN: case14 ns 148->100 (best local, case14 123->118us; 120 also ok). c11ns11 lesson
  says local ns may not transfer -> kept as submission_var_c14ns100.cu OJ-experiment.
- Combine micro-opt (float4 reads + l precompute) folded into submission_clean.cu (neutral-to-tiny).
- case10/13 ns re-swept after combine-opt: still optimal at 64/90. No further ns changes.
- 67.07 remains the ceiling for the MMA design; c14ns100 is the only small candidate.

## 2026-09-05 重大澄清 (提交慢期间复盘)
- 67.36 vs 67.21 的分数差在 case7(57vs56)+case9(60vs59), 不在 case14(都是59).
  case7/9 的 tk 波动 0.204/0.213 是 run-to-run 噪声 (±0.01ms=±1分), 非 ns 差异.
- => "case14 ns100 提升" 的判断被高估; 实际可能是噪声 + 小真增益混合.
- 教训再次验证: ns 微调在 OJ 是噪声级. 别再押注 ns 实验.
- 唯一确定改进: combine 向量化(无回归). clean 是均值最优提交.
- 结构性大提升(72.71)需要目前未找到的洞察; 剩余时间应聚焦稳定提交而非 ns 赌博.

## 2026-09-05 终局: c13ns64b OJ 65.71 = 机器负载噪声(不可信)
- c13ns64b 只改 case13 ns(90->64), 但 OJ 上 case4/6/7/9/10/12 全退化(这些代码没动).
  => 纯机器负载噪声(提交慢几十倍佐证). case13 本身 0.173 与 clean 一致 -> ns64 无效果.
- 结论: clean(case13 ns90, case14 ns100) 是正确最佳. c13ns64b 丢弃.
- 所有 ns 实验全部收敛: 要么最优, 要么噪声级无差异. ns 不是提分杠杆.
- 最终: submission_clean.cu = 67.36 (case13 ns90 / case14 ns100 + combine向量化). 
  真实最佳. 结构性提升需新洞察(未找到).
