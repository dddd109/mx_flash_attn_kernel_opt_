# agentC_policy NOTES — 2026-09-05 (policy/launch/ns + OJ score calibration)

Role: last few % on policy layer + OJ score calibration. All findings quantified,
interleaved A/B where performance claims made.

## Baseline & machine
- baseline.cu (67.36 OJ). Local bench (interleaved, this session): SUM ~1455-1462us.
- case medians (local): 1:11.3 2:11.6 3:12.5 4:20.0 5:18.3 6:26.5 7:203 8:64.5
  9:189.6 10:42.1 11:149.1 12:415.6 13:175.5 14:117.0 us.

====================================================================
## TASK 1: SCORE FORMULA (derived, 97/98 exact on 7 OJ submissions)
====================================================================
The reported per-case `tk` (col1) = OUR kernel time; `tb` (col2) = flash reference.
Score does NOT use per-run tb. It uses a FIXED per-case reference R:

    score_c = round( 100 * R_c / (R_c + ours_c) )      [R_c in us]

Fitted R (us): c1:35.5 c2:35.5 c3:42.5 c4:58.5 c5:44.5 c6:48.0 c7:274 c8:107
c9:308 c10:62.5 c11:244.5 c12:560 c13:236 c14:166.5

Equivalently score = round(100*s/(s+1)) with s = R/ours. The brief's own anchors
confirm: case4 sp2.86->0.741 (100*2.86/3.86=74.1), case6 sp1.85->0.649. R is ~
5-10% below the reported (jittery) flash tk. 97/98 points reproduced exactly;
the single miss (nomax case8 59 vs 60) is OJ noise.

Total = mean of the 14 per-case scores (clean 943/14 = 67.36 ✓).

### Per-case OJ-point sensitivity at the clean operating point
us saved (OJ) to move +1 integer point (R-model, exact to the round boundary):

| case | ours | R    | score | raw   | us to +1 | us to +2 |
|------|------|------|-------|-------|----------|----------|
| 1    | 7    | 35.5 | 84    | 83.53 | 0.5      | 1.0      |
| 2    | 7    | 35.5 | 84    | 83.53 | 0.5      | 1.0      |
| 3    | 9    | 42.5 | 83    | 82.52 | 0.6      | 1.2      |
| 4    | 19   | 58.5 | 75    | 75.48 | 0.0 (*)  | 1.0      |
| 5    | 15   | 44.5 | 75    | 74.79 | 0.6      | 1.3      |
| 6    | 24   | 48.0 | 67    | 66.67 | 0.9      | 1.9      |
| 7    | 204  | 274  | 57    | 57.32 | 1.5      | 9.6      |
| 8    | 71   | 107  | 60    | 60.11 | 1.1      | 4.0      |
| 9    | 207  | 308  | 60    | 59.81 | 5.9      | 14.2     |
| 10   | 37   | 62.5 | 63    | 62.81 | 1.1      | 2.6      |
| 11   | 182  | 244.5| 57    | 57.33 | 1.3      | 8.6      |
| 12   | 352  | 560  | 61    | 61.40 | 1.4      | 16.0     |
| 13   | 169  | 236  | 58    | 58.27 | 1.6      | 8.4      |
| 14   | 115  | 166.5| 59    | 59.15 | 1.7      | 6.3      |

(*) case4 raw already 75.48 > 75.5 boundary: at +0us to score 76 threshold; its
score 75 may read 76 on a re-run. Empirically case4 already oscillates 74/75/76.

### Ranked return on effort (big-case points are cheap-ish)
Big-case +1pt costs: c7 ~1.5us, c11 ~1.3us, c13 ~1.6us, c14 ~1.7us, c12 ~1.4us,
c9 ~5.9us, c8 ~1.1us (all at the clean operating point). Because the big cases sit
JUST above a round boundary, a mere 1.3-1.7us saved on ANY of 7/8/11/12/13/14 = +1
OJ point. Small cases (1-6) need only 0.5-1us/pt but are launch-floor/noise-limited
on OJ (+-1pt run noise) so their "gains" are not reliable to bank.

Empirical ladder-slope fit (regression over 7 submissions) agrees in ranking:
c7 11.7us/pt, c9 9.8, c11 9.2, c13 8.5, c12 16, c14 5.7, c8 3.5 (avg slope over
the 182->203us ladder range, hence larger than the near-boundary 1.3-1.7us values).

====================================================================
## TASK 2: LOCAL vs OJ REPRESENTATIVENESS  (kernel time only)
====================================================================
Machine speed parity CONFIRMED by case7: local 203 vs OJ 204 (short rows, fill
insensitive). So the local/OJ gaps below are DISTRIBUTION mismatches, not clock.

| case | local | OJ(clean) | delta    | flag |
|------|-------|-----------|----------|------|
| 1    | 11.3  | 7   | +4.3us (+61%) | local launch floor > OJ floor |
| 2    | 11.6  | 7   | +4.6 (+66%)   | same |
| 3    | 12.5  | 9   | +3.5 (+39%)   | same |
| 4    | 20.0  | 19  | +1.0 (+5%)    | OK |
| 5    | 18.3  | 15  | +3.3 (+22%)   | floor + row-fill |
| 6    | 26.5  | 24  | +2.5 (+10%)   | borderline |
| 7    | 203   | 204 | -1.0 (-0.5%)  | REPRESENTATIVE |
| 8    | 64.5  | 71  | -6.5 (-9%)    | local FASTER (fill too low) |
| 9    | 189.6 | 207 | -17 (-8%)     | local FASTER (fill too low) |
| 10   | 42.1  | 37  | +5.1 (+14%)   | local floor |
| 11   | 149.1 | 182 | -33 (-18%)    | local MUCH FASTER (fill too low) |
| 12   | 415.6 | 352 | +64 (+18%)    | local SLOWER (fill/other) |
| 13   | 175.5 | 169 | +6.5 (+4%)    | ~OK |
| 14   | 117.0 | 115 | +2.0 (+2%)    | ~OK |

### Root cause: local seed-42 randint(1,seq) fill is ~42-45%; OJ fill is higher
Local uniform fill fractions: c7 46%, c8 42%, c9 45%, c11 43%, c12 43%.
OJ effective fills (matched by reproducing OJ reported times with ns=22):
- case11: OJ 182us @ ns22 == local fill 0.485 (sum_pages ~5868 vs local 6068? note
  exact pages differ by distribution shape; fill_fraction 0.485 vs local 0.43).
- case9: OJ 207 @ ns11 == local fill ~0.50 (210us) vs local-uniform 0.45 (190).
- case8: OJ 71 @ ns22 == local fill ~0.47 vs local-uniform 0.42.
=> For batch>1 mid/large cases OJ rows are ~10-20% longer than local-uniform.
- case12: reverse (local SLOWER). OJ tk flash = 572us is only +3% over local flash
  (554us), yet OJ ours 352 vs local 415 = OJ did 15% less work OR the OJ case12 row
  distribution has far less spread (balanced rows) than local seed-42. case12 is the
  ONE case where local tuning is misleading in the pessimistic direction (local looks
  slower than it will be on OJ). DO NOT over-optimize case12 on local; its OJ score
  (61) already reflects the easier OJ distribution.
- case13/14 (batch1, full rows): representative; local==OJ (small const offset).

### Distribution sensitivity of policy choices (checked per flagged case)
ns policy tested at OJ-like fill fr0.47-0.51: case8 ns22 still optimal; case9 ns11
optimal; case7 ns5 optimal; case12 not re-optimized (see flag). Case11 is the ONLY
case whose ns optimum MOVES with fill -> TASK: found the win (below).

====================================================================
## TASK 3: ns re-check on cases 5/6/8/10 (interleaved, sustained)
====================================================================
All re-confirmed OPTIMAL under local-uniform AND under OJ-like fill:
- case5: ns3 optimal {1:+18%worse, 2:+13%worse, 5:+3.3%worse}
- case6: ns5 optimal (ns6 +0.3% noise in one run, -1.9% in 12-round sustained ->
  ns5 strictly better; ns4 +6%, ns3 +13%, ns8 +22%)
- case8: ns22 optimal (ns16 +4.3%, ns19 -0.35% noise, ns24 +23%, ns26 +11%) and at
  fr0.47 fill ns22 still optimal (ns26 +9.5%, ns30 +13%)
- case10: ns64 optimal (ns52 +3%, ns74 +1.6%, ns43 +8%, ns86 +5%)

No policy change for 5/6/8/10.

====================================================================
## TASK 4: CASE 4 (score 75, ratio 2.2x) — launch-bound analysis
====================================================================
- ns=1 (fused, no combine) confirmed right: ns2 +41%, ns3 +40% worse (interleaved).
- case4 = 512 CTAs (0.7 wave), pages<=4/row. Time: local 19-20us vs OJ 19us.
- Local launch floor (empty-ish tiny kernels) is ~9-12us (case1/2/3). So case4's
  19us is ~half floor, half real work + epilogue. On OJ the floor is lower (~7us),
  so case4's 19us is nearly all real work there.
- Distribution check: uniform 19.3 / full 21.9 / half 15.0 / skew 22.0 us (ns=1
  robust across all; no ns>1 helps in any). ns=1 IS the right choice.
- Single-kernel "smarter handling of all 64 batches": the current ns=1 path already
  launches ONE kernel (grid 512) with no combine and direct output. Combining 64
  tiny batches into fewer, wider CTAs was tested historically (multi-unit/CTA) and
  is slower (occupancy loss). Nothing to win here: case4 is 0us to +1pt away (raw
  75.48) so it may already tick to 76 on an OJ re-run without any code change.

====================================================================
## TASK 5 (NEW FINDING): CASE 11 ns 22 -> 43 at OJ fill (real candidate win)
====================================================================
Under the OJ fill distribution for case11 (~0.485-0.51) the ns optimum moves UP:

  case11 @ fr0.485 (OJ-exact, ns22==182us==OJ): ns22 180.4 | ns30 +4.2% | ns34 -2.5%
  ns38 -6.5% | ns43 -9.4% | ns52 +10% | ns64 +4% | ns86 +9%
  case11 @ fr0.51: ns43 -14.3% vs ns22
  case11 @ fr0.55: ns43 -13.2% vs ns22
  Crossover: ns43 > ns22 only above fr~0.46. Below that ns43 is +2% (not harmful).

Explanation: kv4/wu64 case11 rows are long (~92 pages/unit at OJ fill); ns22 gives
~4.2 pages/CTA and 1408 CTAs (1.93 waves). ns43 gives ~2.1 pages/CTA and 2752 CTAs
(3.78 waves) -> better DRAM concurrency, like the kv8 big cases (case9 2816 CTAs,
case12 3776). case8 (short 30-page rows) already over-splits at ns22 so it does NOT
want more; it is why the rule must be pages-gated, not a blanket regime-E change.

### Candidate one-line policy change (baseline_pol_c11ns43.cu)
Regime E:  `ns = (pages >= 512) ? 43 : 22;`   (was `ns = 22;`)
Affects ONLY case11 (pages 766). Case8 (pages 256) unchanged. ALL 14 verify PASS.

### A/B (interleaved 8 rounds):  baseline vs c11ns43
- case11 fr0.485 (OJ fill): 180.7 -> 163.1 us = -9.7%
- case11 localseed42 (43% fill): 148.5 -> 151.8 = +2.2% (the only downside, and only
  if OJ fill were actually ~43%, contradicted by all evidence)
- case8 fr0.47: -0.4% (unchanged). All other cases bit-identical (regime not hit);
  residual +-2% on some small cases is binary-recompile noise, not behavior.

### Expected OJ score impact
case11 currently 182us/57. R=244.5. ns43 OJ estimate: 182*0.906 ~ 165us
-> score 60 (+3). Even at HALF the measured gain (173us) -> 58 (+1).
Historical OJ fill estimate (67%) and the flash-tk-ratio method (48-51%) both place
case11 above the ns crossover, so this is directionally safe. The prior OJ experiment
that "killed" ns changes on case11 tested ns 22->11 (the WRONG direction); ns 43 was
never tried and is predicted to win.

Recommendation: SUBMIT baseline_pol_c11ns43.cu to OJ. Risk is bounded (~+2% on case11
if OJ fill is unusually low = ~0 OJ points lost; upside +1 to +3 points).

====================================================================
## Combine / launch (context from prior agents, re-verified not the lever)
====================================================================
- In-context combine cost is 3-18us (batch1 worst: case13 ~11us, case14 ~15us).
- Fused-atomic single-launch was tested (agentB): 5-13% SLOWER. 2nd launch is
  cheaper than in-kernel atomic reduction. DEAD.
- combine kernel micro-opt (unroll/parallel/pipeline): neutral/worse. DEAD.
- => policy layer has no combine win; the launch serialization is ~the machine's
  price of a 2nd kernel and cannot be removed for less than it costs.

## Score model sanity notes for coordinator
- OJ per-run reported tb (flash) jitters run-to-run, but score is computed against a
  fixed per-case R. => judging local changes by "speedup vs local flash" is the right
  proxy only if local flash and OJ R move together; they do NOT per case (local flash
  is 15-27% faster than OJ flash on batch>1 due to lower fill). Judge instead by
  (new_ours/current_ours) ratio applied to the OJ-reported ours of the last clean
  submission, then feed into score=round(100*R/(R+ours)).

## Artifacts (this workspace)
force_ns.cu/.so (env FORCE_NS override), ns_ab.py, ns_ab_dist.py, ab_lib.py,
dist_sens.py, fill_scan.py, ns_oj11*.py, ns_knee11*.py, baseline_pol_c11ns43.cu/.so,
fit_refR.py, boundary_table*.py, conv_table.py, score_table.py, rep_table.py.
