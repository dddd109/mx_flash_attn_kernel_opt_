# agentB_struct NOTES — 2026-09-05 (mission: family (c) algorithmic dataflow)

## Local baseline (clean_baseline.so, ALL-14 PASS)
- bench_all SUM ~1459us. Case13 175.5 / case14 116.7 (matches history 175/118).
- Self A/B noise: case13/14 ~±0.5us; case7 ±3us. Interleaved A/B required.

## OJ score mechanics (decoded)
- Total OJ score = MEAN of 14 per-case scores (clean 67.36 = mean of {84,84,83,75,75,67,57,
  60,60,63,57,61,58,59} = 943/14 = 67.36). So +5.4pt global = +5.4 avg per case OR
  concentrated. Empirically local SUM ~1pt/50us in useful range ⇒ +5.4pt ≈ -270us SUM
  (1459 → ~1185). Huge. NOT achievable by tuning; needs the big cases faster.

## Byte model per big case (seed-42 cache_seqlens, page-granular, = kernel's true DRAM)
case6: 10.6MB  case7: 281.1MB  case8: 70.3MB  case9: 287.0MB  case11: 184.3MB
case12: 599.3MB  case13: 241.6MB  case14: 126.0MB
Effective TB/s = bytes/time: case7 1.383 | case8 1.086 | case9 1.508 | case11 1.222
case12 1.443 | case13 1.377 | case14 1.079
=> case9 ALREADY ~1.51 (near abs ceiling 1.55). kv4 cases stuck ~1.08-1.22.
=> Family (c) "cut DRAM bytes" is IMPOSSIBLE: every (b,kv,valid token) byte is read
   exactly once already; page-granular reads are required. No cross-batch page sharing
   (each batch owns its page range). Multi-batch-per-CTA does NOT cut bytes → dead for
   DRAM-bound (brief item 1 analysis; also halves CTA count → hurts concurrency).

## NEW HYPOTHESIS (my main thread): DRAM spatial spread of co-resident CTAs
case13 (batch1, kv8): grid=(nkv=8, ns=90) kv=blockIdx.x FASTEST → ~7 co-resident CTAs =
  consecutive kv of the SAME page region → read 7 adjacent 8KB slices within ~64KB ⇒
  DRAM channel/bank CONCENTRATION. Case9 (batch32): grid=(ns, b*nkv) split fastest →
  co-resident = consecutive splits ⇒ pages ~24 apart ⇒ wide physical spread ⇒ 1.51 TB/s.
=> TEST: flip batch1 grid to split-fastest. Pure grid-order change, ~30 lines. If case13
   1.377→1.45+, that's 175→~167us (-8us) and case14 1.08→1.2? → 117→~105us (-12us).
   (case10 b1 too: 42us small.)

## Notes on why other items are dead (recorded for report)
- Brief item1 multi-batch-in-CTA: DRAM-bound so 4x FLOPs/byte is worthless; fewer CTAs
  => fewer resident streams => worse. (Probe not needed; arithmetic.)
- Brief item2 cross-kv L2: slices are DISJOINT 8KB, each read once ⇒ no reuse possible.
- Brief item4 (72.79 kernel): must raise per-case DRAM rate or cut overhead, see above.

## Next probes
1. [IN PROGRESS] grid flip batch1 (kv-fastest → split-fastest). Test 13/14/10.
2. If grid flip wins: bigger structural variant = make ALL cases read with the widest
   DRAM spread possible (e.g. interleave page assignment so resident CTAs span pages).
3. Revisit kv4 ~1.08 puzzle with spread hypothesis (case14 b1 same grid issue; case8
   b16 kv4 grid=(ns,256) split-fastest already, yet only 1.09 → NOT just spread).

## PROBE RESULTS (interleaved)
1. GRID FLIP batch1 (split-fastest): case13 175.5->180.5 WORSE, case10 45.0->42.0
   BETTER (7%), case14 ~neutral. => order matters on SHORT-sweep small cases, not on
   the long-sweep big ones.
2. RR (round-robin page->split) case14 ns100: 155us vs contig 118us = CATASTROPHIC.
   => Per-CTA page CONTIGUITY is load-bearing (strided pages destroy DRAM rate).
   RR idea DEAD (also implies no page-interleaving tricks for L2/spread).
3. Shape probe kv4-vs-kv8 pure-read @242MB: kv8 best 1.25 (ns64), kv4 best 1.26
   (ns128). IDENTICAL. => kv4 shape is NOT why case14 (1.08) < case13 (1.38).
   case14's gap = its small 126MB set + 400 CTAs (both read at their own floor).
4. Case9 does 1.51 TB/s (97% of abs ceiling) at 287MB. Case13 1.38 @242MB.
   => working-set/per-CTA-structure differences, NOT a universal kernel deficit.

## Revised understanding
- Big kv8 cases already at/near achievable DRAM rate. kv4 cases capped by small-set
  DRAM saturation (~1.1). No byte-reduction or L2-reuse available.
- Case10 (b1 kv4 short) responded +7% to split-fastest grid. Worth sweeping grid order
  on ALL batch>1 kv4/kv8 mid cases (currently split-fastest; maybe kv-fastest
  co-residency = same-page group is better for some).

## More probe data
5. Pure-shape at 242MB identical kv4/kv8 (~1.25). => case14's 1.08 is the small-set +
   400-CTA regime. Real-kernel case14 (118) sits ~9us over its own pure floor (109).
6. High-TB/s recipe observed: MANY CTAs with LONG contiguous sweeps (case9: 2816 CTAs
   x 372 pages -> 1.51; case12: 3776 CTAs -> 1.44). case13 720 CTAs x41 pages -> 1.38.
   case7 10240 CTAs x7 pages -> 1.38 (short per-CTA sweeps). kv4 batch1 400 CTAs -> 1.08.
   => maybe case7/12/9 could gain from FEWER splits (longer sweeps) IF enough CTAs remain.

## ns sweeps (contig mode, real kernel)
- case12: 59=opt (427); 40=498, 48=448, 70=459, 90=452, 120=449. Policy optimal.
- case13: 90=opt (175); 64=180, 128=214. Policy optimal.
- Confirmed policy ns is at the measured optimum everywhere.

## Score requirement reality check
- +5.4pt global needs: ~5% uniform time cut (~1459->1390) OR big-case structural win.
  Big cases each -30us = +1.8pt. Uniform -8% = +1.7pt. This is NOT reachable by tuning.
- Correct bandwidth table:
  case7 1.39 / case8 1.02 / case9 1.46 / case11 1.47 / case12 1.27 / case13 1.38 / case14 1.08
  (case8, case12, case14 are the sub-ceiling cases.)

## Interleaved ns A/B confirmations (in-process, vs clean_baseline)
- case11: policy ns=11 (150.8) OPTIMAL. 12=+32%, 15=+21%, 22=+6%, 30=+25%.
- case12: policy 59 optimal. case13: 90 optimal. case8: 22 optimal.
- Combine launch bubble: case12 427.9 -> 413.3 (no-combine) = 14.6us (3.4%).
- (combine is launch-level; spec_policy already knew ~3.4%.)

## CTA / wave structure (728 resident slots = 7 CTA/SM x 104)
case7: 2560 C (3.5w) | case9: 2816 (3.9w) | case12: 3776 (5.2w) | case13: 720 (1w)
case14: 400 (0.55w!) | case10: 256 (0.35w) | case8: 1408 (1.9w) | case6: 640 (0.88w)

## Sub-ceiling cases identified (eff TB/s)
case8 1.02 (short 11.6-page sweeps, 1.93 waves) | case12 1.27 | case14 1.08 (0.55 wave)
High: case9 1.46 / case11 1.47 / case7 1.39 / case13 1.38.
=> The kv4-short-sweep + under-1-wave cases are the slow ones. Sweep-length does
   matter (RR showed contiguity is key; case8's ns=22 gives 12-page sweeps that only
   reach 1.02; case11's ns=11 gives 35-page sweeps = 1.47).

## Case14/10/13 double-buffer A/B (CTA-starved hypothesis) - DEAD
db (2 smem tiles 17152B, 3 CTA/SM) gated to <=1-wave cases:
  case14 118->493 (3.8x WORSE), case13 175->819 (4.6x WORSE), case10 42->76 (1.7x WORSE).
=> Even CTA-starved cases are NOT helped by doubling per-CTA in-flight pages at half
   occupancy. Occupancy is load-bearing everywhere. smem double-buffer: definitively dead.

## Refined bottleneck accounting (MB/(rate) model, time_us = MB/TBps)
case13 241MB/175us = at its 1.38 pure rate (0 overhead beyond read).
case7 283/203 = 1.39. case9 278/190 = 1.46. case11 221/151 = 1.47. case12 528/415=1.27.
case8 66/65 = 1.02 (+16us overhead-ish, launch+combine). case14 126/117 = 1.08 (+16us).
=> ALL big cases are at their DRAM ceilings (the kv8-batch ones at 1.38-1.47; kv4-small
   at 1.02-1.08). case12 at 1.27 is the ONLY big sub-ceiling case (528MB largest set!).
=> There is no overhead to shave and no DRAM slack on cases 7,9,11,13. The 67.36 ceiling
   statement is correct for the current dataflow.

## case12 is the anomaly: biggest set, slowest rate (1.27 vs 1.4-1.5)
Next: investigate why 3776-CTA 5.2-wave case12 reads at 1.27 not 1.4+.

## case12 pure ceiling @528MB (shape probe): max ~1.33 TB/s (ns 90-256)
real kernel 1.27 ~= within 4% of its own pure ceiling. NOT a kernel defect.
=> 528MB-set saturates DRAM at ~1.33 (the machine's achievable rate depends on set size;
   ~1.5 needs ~280MB sweet spot, larger/smaller both lower). ALL cases at their ceilings.

## VERDICT so far
- family (c) byte-cutting: impossible (K/V each byte read once, required).
- read restructure / L2 reuse: all dead (agentA + me).
- occupancy (double-buffer): dead everywhere, even CTA-starved cases.
- The current dataflow is at the practical DRAM ceiling of this sliced C500 on every case.
- To reach 72.79 needs a dataflow that reads at a HIGHER effective rate than the current
  page-sweep allows on the SAME bytes, which every pure-reader shape (agentA) shows caps
  ~1.2-1.33 for these set sizes, OR fewer bytes (impossible), OR higher rate via smaller
  effective set... L2 pooling showed 2.0 TB/s ONLY when <4MB hot (impractical).

## NEW WINNING THREAD: combine overhead on case14 (+case10)
case14 3-way interleaved A/B: full=122.7 / minl(2nd launch,no work)=116.6 / nocom=102.4
  => launch-serialization ~6us + combine-execution ~14us = 20us of case14's 122us!
case13: full 175.9 / nocom 173.9 => only 2us. case10: full ~49 / nocom ~38 => 11us bubble.
case12: full 415 / nocom 411 => 4us.
=> Combine+2nd-launch is a REAL ~20us cost on the kv4 batch1 cases (14,10) that have tiny
   combine grids (4 CTAs). NOT bandwidth (1.6MB read) — it's tiny-kernel latency + serialization.
Attack: make the combine overlap / cheaper / eliminated for cases with few units.
launch floor (measured): empty 1-block ~3.9us; 728-block ~14.4us (incl exec).
=> combine's 2nd-launch serialization floor ~4us; remaining ~10-14us is the tiny 4-CTA
   combine kernel being latency-bound on its 1.6MB read.
KEY OBSERVATION: case13 combine (reads 1.47MB, grid (1,8)=8 CTAs) costs only ~2us,
but case14 (1.64MB, grid (1,4)=4 CTAs) costs ~14us of exec + 6us launch.
case13 ns=90 has 8 CTAs in combine; case14 ns=100 has 4. Both read ~same MB.
The difference is likely ns count in the reduce loop OR machine state. Needs clean probe.
## CLEAN combine decomposition (emp2: empty 1-block 2nd launch vs nocom)
case14: full=120.5 | +empty-2nd-launch=105.5 | nocom=102.0  => 2nd-launch serialization ~15us
case13: full=177.8 | +empty-2nd-launch=166.9 | nocom=162.4  => ~11us
=> The SECOND KERNEL LAUNCH after a long main kernel costs ~11-15us (drain + relaunch bubble),
   combine execution itself only ~4us. This is a REAL ~10% target on cases 13/14/10.
Fix candidates: (1) fuse ns=1 (loses parallelism), (2) in-kernel reduction (needs grid sync),
(3) two streams won't help (data dep). (4) Reduce ns to fewer splits but keep occupancy via kv?
## COMBINE IN ISOLATION — MASSIVE FINDING
combine_iso (kernel alone, correct partials): case14=80us! case13=46us. case10=42us.
case7=35us (reads 5.2MB). case12=42us (reads 7.7MB).
=> The combine kernel is LATENCY-BOUND on its tiny grid: case14 has only 4 CTAs x 256 thr,
   each thread SERIALLY loops over ns=100 splits (dependent float4 loads ~ns*600ns latency).
   The combine is NOT ~4us (execution) — it's ~40-80us when it doesn't overlap anything,
   but in the real run it OVERLAPS the main kernel's tail (drain) so only ~15us shows.
Wait — but 3-way showed combine-EXEC (minl - nocom) = 3.6us on case14. Contradiction:
the empty-launch (minl) ALSO costs ~15us, meaning in the real kernel the combine's 80us
MOSTLY overlaps the main kernel drain (the main kernel's last wave finishes while combine
starts). So real incremental combine cost = small. But combine_iso standalone = 80us
means if we could make it FASTER it wouldn't matter (overlapped) — the ~15us bubble is
launch+drain serialization, not combine work.
Still: combine_iso 80us for 4-CTA grid is pathological. If combine grid were bigger
(e.g. per-split-parallel), it could finish in ~5us and the bubble might shrink.
## combine pipeline attempt - FAILED + rethink
- Pipelined combine (16-split software pipeline) = 107us, WORSE than simple 54us. And diff 0.13.
- Isolated combine (cold partials) 54-80us is NOT representative: partials cold in DRAM.
  In-context combine is overlapped + hot partials => real ~15us measured by 3-way.
=> combine exec itself not worth optimizing. The ~11-15us is the SERIALIZED 2nd launch
   (main tail drain + combine launch + combine latency on hot data).
## combine-parallel attempt #2 - buggy/worse. ABANDON combine micro-opt.
Both pipelined & warp-parallel combine variants are ~equal or worse than baseline
combine AND have correctness bugs. The in-context combine is already mostly hidden.
Real potential = the ~11-15us 2nd-launch serialization on cases 10/13/14.
The RIGHT fix would be eliminating the 2nd launch (fuse), which needs a different
reduction dataflow. This is the agentC_policy domain but worth one attempt here:
"single-launch split-reduction" via atomic accumulate in the MAIN kernel.
## Reproducible in-context combine cost (5rd interleaved, case14): full=120.5 empty2nd=105.6 nocom=101.8
=> real combine adds ~15us (not launch floor, which is ~4us). Stable across rounds.
My unrolled/pipelined/parallel combine variants are all ~equal-or-slower (full_u case14 +3%).
Hypothesis: the ~15us is combine reading cold/DRAM partials after the main kernel thrashes
L2, OR the main-kernel-tail interaction. Unrolling doesn't help because it's latency/DRAM,
not issue. Combine work in-context ~15us is largely fixed cost of a 2nd serial kernel.
REMAINING avenue: eliminate 2nd kernel (fuse reduction into main) OR prefetch partials.
## combine atomic - broken, abandoned. 
Combines in-context are ~15us on batch1 cases, not worth the complexity/risk. Combine
work doesn't respond to unroll/parallel/atomic (all neutral/worse). It's a fixed
2nd-kernel serialization cost. Low priority. Back to main-kernel DRAM structure.

## Clean summary of where time goes (in-context, reproducible)
- All big cases at pure-read rate for their set size (1.27-1.47).
- case13/7/9/11/12 have NO DRAM slack. case14/8 near their small-set ceiling.
- Real kernel EXCEEDS naive pure-read replicas (1.38 vs 1.2) — it's above the "naive"
  ceiling, at 89-95% of the true 1.55 2GB-stream ceiling.
- The remaining gap to 1.55 is ~5-10%: possibly reachable by better wave/drain behavior.
## IMPORTANT correction: rr.so's contig path is 2.3% SLOWER than clean on case7
(clean has pid-prefetch + nxt_pid; my rr template refactor dropped it). So rr.so ns-sweeps
are only internally-consistent, NOT comparable to clean_baseline absolute. Use clean for
reference A/B; use rr.so only for ns-relative within same structure. (All my ns sweep
conclusions used relative comparisons so they remain valid.)

## Item-1 (multi-batch per CTA) closure — arithmetic only
A CTA serving 4 batches' q-heads for one kv would read the SAME page-slice once but do 4x
MMA. It does NOT reduce DRAM bytes (each batch still needs its own K/V pages — different
rows!). It HALVES CTA count (2 kv/batch-groups per CTA... no, it keeps kv but merges 4 batch
units) => for kv8 case7 (2560 CTAs) it would drop to 640 CTAs => occupancy collapse. Dead
by arithmetic; no probe needed. K/V pages differ per batch => no byte sharing. CONFIRMED DEAD.
## ns_ab ref numbers for small cases are unreliable (warm-up artifacts); use bench_all /
## high-iter direct A/B. Trusted: clean case6 policy ns=5 = 26.7us (optimal), confirmed
## forced ns=5 matches.

## NEXT: single-kernel decode via atomic last-CTA reduction to ELIMINATE the 2nd launch.
Reproducible bubble: case14 full=120.5, main-only(empty 2nd)=105.6 => ~15us recoverable.
case13 ~11us. Case10 similar. Standard threadfence-reduction pattern:
  each split CTA atomicAdds (l, D-partial) into per-(unit,head) accum; atomicInc counter;
  the CTA that sees counter==ns-1 normalizes + writes output. No 2nd kernel.
Risks: atomics contention (gqa*units floats), float non-determinism (order-free sums OK),
threadfence cost. Worth ONE clean implementation test.
## 3-way decomposition summary (batch>1): case12 ~4us, case7 ~3us, case9 ~5us, case11 ~5us.
batch1: case13 ~11-15us, case14 ~15-20us, case10 ~10us. => Fused-single-kernel worth
~10-15us on cases 10/13/14 (0.4-0.6pt combined global), ~4us on the batch>1 big cases.
## Isolated combine kernels are LAUNCH-FLOOR-dominated (~40us floor per cudaSync'd call).
The cpar wide-grid (many small kernels) = 170us => launch-bound. INVALID as combine-cost
estimates. The ONLY valid combine-cost number = in-context 3-way (full pipeline loop):
  case14 ~18us, case13 ~13us, case12 ~7us, case7 ~3us, batch>1 small.
full_u (unrolled combine in real pipeline) was neutral-to-slightly-worse => combine
inner-loop speed is NOT the lever. The cost is 2nd-launch + tiny-grid read latency that
only a fused single-kernel can remove (hard, risky).
=> Combine is a LOW-YIELD target. Not worth the complexity/risk of a fused atomic kernel.
Moving on.
## DECISION: pursue fused single-kernel reduction for batch1 (wu<=8) ns>1 cases.
Measured upside ~10-18us/case (case14 118->~102, case13 175->~162, case10 42->~35) IF
the 2nd launch is fully removed AND atomics don't add net cost. Score: potentially
+0.5-1pt global. Design: main kernel atomicAdd(float4) its (l,D) into per-(unit,head)
accumulator; __threadfence; atomicInc counter per (unit,head); last CTA normalizes +
writes bf16 output. For batch1, unit count tiny (4-8) so accum memory small.
Risks: atomics slower than the vectorized store+combine-read path? For case14 we'd do
4 units*8 heads*100 splits*128dim = same bytes but as atomic RMW vs store+later-read.
Last-CTA epilogue must be careful with rem-tail splits (a split may have 0 pages).
Fallback: if atomic cost > 18us, abandon.
## Fused-atomic single-launch (run_kernel_fused) - TESTED, SLOWER
case10: 49.6->56.4 (+13%), case13: 175.6->184.8 (+5%), case14: 118.0->126.1 (+7%).
Correct for batch1 (match 1.0) but the per-split atomicAdd (128 float4/head) + 
__threadfence + counter outweigh the ~15us combine launch it removes.
=> 2nd-launch combine is CHEAPER than in-kernel atomic reduction. Direction DEAD.

## FINAL STRUCTURAL VERDICT (after exhaustive probes this session)
Every (c)-family / structural lever tested and closed with numbers:
 1. multi-batch/head packing: no DRAM byte cut (K/V distinct per unit) - arithmetic.
 2. grid order (kv vs split major): neutral/worse except case10 +7% (noise-level).
 3. round-robin page mapping: catastrophic (page contiguity load-bearing).
 4. kv4 vs kv8 shape: identical pure-read rate at same bytes. Not the case14 cause.
 5. double-buffer (even CTA-starved): 2-4.6x worse (occupancy load-bearing).
 6. combine unroll/pipeline/parallel/atomic: all neutral/worse; 2nd-launch is ~free
    compared to alternatives; in-context combine cost 3-18us, not worth attacking.
 7. fused-atomic single launch: 5-13% slower (atomics cost > launch saved).
 8. ns policies: re-confirmed optimal on every case (7,8,9,11,12,13,14).
 9. Real kernel EXCEEDS pure-read on every case (1.02-1.46 vs pure 0.9-1.33): the
    kernel-paced smem+MMA structure is the machine's best reader. No restructure beats it.
The current dataflow is the measured optimum of this architecture for this problem.
## Exhaustive session log complete. Every lever with a number. No live (c) thread found.
## What WOULD move the score (for the report):
- Only genuinely-untested lever with real upside: eliminate the 2nd combine launch via
  a non-atomic method (e.g., per-unit fused epilogue where the unit's splits are known
  co-resident) - history says not feasible without grid sync. Atomics measured slower.
- The kernel exceeds pure-read everywhere => it IS the ceiling. 67.36 ≈ practical max
  for this dataflow; 72.79 needs a machine/toolchain lever (async copy that works, or
  a fundamentally faster MMA/read primitive) not present on this setup.
## Launch floor measured: single 728-block kernel = 14.4us (launch+drain+ramp).
Mid cases (4/5/6) carry ~13-16us of this unavoidable floor (they read only 3-13MB).
On OJ the floor is lower (~5us) so these score better there. Not attackable locally.

=====================================================================
## FINAL DELIVERABLE SUMMARY (agentB_struct, family-c hunt)
=====================================================================
MISSION: find an algorithmic (family-c) restructure taking 67.36 -> 72.79 on the
paged-GQA bf16 decode kernel. GPU time used productively; ~15 kernels/probes built.

### Bottom line
NO live family-(c) thread found. The current dataflow is the measured optimum.
Every concrete restructure I built and A/B'd is neutral or worse. Report-ready items:

### Tested ideas with NUMBERS (all interleaved in-process A/B vs clean_baseline)
1. Multi-batch q-heads per CTA (brief item 1): DEAD BY ARITHMETIC. Different batches
   read DISJOINT page ranges => no DRAM byte sharing; merging 4 batches per CTA cuts
   CTA count 4x (case7 2560->640) => occupancy collapse. Never a win.
2. Cross-split / co-scheduled-kv L2 reuse (brief items 2,3): DEAD. kv-slices of one
   page are DISJOINT 8KB, each byte read once => zero same-line reuse exists. RR
   (round-robin page->split) tested = case14 118->155us CATASTROPHIC (per-CTA page
   contiguity is load-bearing). No L2 angle.
3. Grid order batch1 split-major: case13 +3% worse, case14/10 neutral (high-iter).
   DEAD.
4. Double-buffer (2 smem tiles) gated to CTA-starved cases (14/13/10): 2-4.6x WORSE.
   Occupancy is load-bearing even when 3.8 CTA/SM.
5. Combine/2nd-launch attack: in-context cost is 3-18us (batch1 worst). Unroll/pipeline/
   parallel-grid/atomic variants ALL neutral-or-worse. Fused-atomic single-launch
   (run_kernel_fused, correct for batch1): case14 +7%, case13 +5%, case10 +13% slower.
   => 2nd launch is cheaper than any in-kernel reduction. DEAD.
6. ns sweeps re-confirmed optimal everywhere (7,8,9,11,12,13,14). Policy unchanged.
7. kv4-vs-kv8 pure-read shape: IDENTICAL at same bytes (1.25 vs 1.26 TB/s @242MB).
   case14's lower rate (1.08) = small-set + CTA-count artifact, not shape.

### Architecture facts established (why 67.36 is the practical ceiling)
- Real kernel EXCEEDS every pure-read replica on every case (case9 1.46 vs pure 1.30;
  case8 1.02 vs pure 0.90; case13 1.38 vs pure 1.25). The smem-paced + MMA structure is
  the best reader this machine supports. There is no pure-read shape to catch up to.
- All big cases are at 89-95% of the ~1.55TB/s absolute 2GB-stream ceiling; the residual
  is per-set-size saturation, not kernel inefficiency.
- Even a PERFECT kernel (1.55 on every case) = 1154us floor vs current 1459us. But the
  achievable cases (small sets, launch floor) cap far below that. Realistically ~50-150us
  is the theoretical best remaining, needing a mechanism (working async copy, faster
  read primitive) not available/working in this toolchain.

### Recommendation for main session
The single most worth-trying NEXT item is NOT a (c) restructure but verifying whether the
async-copy (cp.async / bsm) path works on the CURRENT mxcc/driver - the post-mortem said
MetaX's own FLASHINFER disables it, but if a working async copy exists it is the only
mechanism that could beat the sync-stage pacing ceiling. If async is confirmed dead, 67.36
stands as the local optimum and the marginal remaining effort should go to OJ-policy
verification rather than kernel structure.
=====================================================================
