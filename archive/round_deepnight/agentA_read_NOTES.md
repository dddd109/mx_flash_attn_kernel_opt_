# agentA_read NOTES — 2026-09-05

Mission: find a (b)-family wider/contiguous DRAM-transaction restructure that beats
the current kv-sliced page-sweep, WITHOUT losing smem-mediated MLP, on case13/14.
Ceiling doc said both big cases already read at/past achievable pattern rate and that
(b) is only worth it if a concrete design survives micro-probe.

## Workspace
/tmp/agent_ws/agentA_read/ — baseline.cu/.so (175/118us confirmed, ALL 14 PASS),
probe_wider.cu/.so, probe_pace.cu/.so, probe_pace2.cu/.so, probe_pref.cu/.so,
probe_stagger.cu/.so (+ stag_*.so), probe_sts.cu/.so, variants/{ns_100,ns_128,ns_180,
ns_256,strip_epi,no_mma}.so, plus *_ab.py logs.
All numbers are same-process interleaved CUDA-event bursts (best of 4-6 rounds).

## Confirmed baseline (clean, interleaved, machine shared)
case13 (b1 seq58966 kv8, 241.6MB): real 175us = 1.38TB/s
case14 (b1 seq61519 kv4, 126.0MB): real 118us = 1.06TB/s
verify_real ALL PASS. bench_all SUM ~295us.

## Probe suite (all read the SAME unique bytes once, exact kernel layout)

### 1. The mechanism: per-page __syncwarp pacing, NOT shape, NOT smem
Control (no barrier, xor-accumulate, exact kernel address pattern kvgrid):
  case13 1.17TB/s (205us), case14 0.72-0.74TB/s (170us)
+ __syncwarp after each page's 8 KB/V uint4 loads (kv_sync):
  case13 1.28-1.30TB/s (186-189us)  case14 1.08-1.17TB/s (107-116us)
=> The real kernel is faster than pure reads because the kernel stages each 8KB
page slice with a per-page barrier, forcing every co-resident CTA to issue its full
page burst densely. The barrier is the MLP enabler. (Reg-level xor lets the compiler
spread the 8 loads and stall.)

### 2. Wider/contiguous per-CTA shapes are DEAD (family b)
- strip4k (each CTA reads one CONTIGUOUS 4KB K + 4KB V strip/page; same #CTA & bytes
  as kv-sliced): case13 best 1.19-1.21TB/s @ ns128 (199us) — same as kv-sliced, both
  still << real kernel 1.38. case14 best 1.01 @ ns256 (125us) — NOT better than kv.
- slice512 (2 adjacent kv slices = 512B contiguous per token): 0.71-0.73TB/s (worse).
- whole-page contiguous (kv4 16KB / kv8 64KB per CTA): catastrophic 0.04-0.05TB/s.
Conclusion: contiguity-per-CTA is NOT the DRAM lever on this machine; slice width per
CTA is already fine; wider per-CTA reads lose (fewer CTAs, worse pacing, no density
gain). The 8KB/slice is already dense at the DRAM level when paced by the barrier.

### 3. LDG->STS staging is NOT the speedup; MMA is fully hidden
- sts_min (exact kernel stage_page LDG->STS + per-page sync + MINIMAL smem read):
  case13 0.35TB/s (683us), case14 0.32TB/s (397us) — 4x SLOWER than the real kernel.
  My earlier full-LDS-consume smem probe (smem_pace) also ~0.35TB/s. => STS or heavy
  smem reads are NOT what makes the kernel fast; a pure LDG->reg + sync cadence
  (kv_sync) already matches/beats the kernel on case14.
- no_mma variant (kernel with the 2x8 MMA+exp+softmax replaced by tiny LDS+xors):
  identical to real (case13 175 vs 175; case14 117 vs 118). Epilogue-stripped variant
  also identical. => all compute/epilogue is hidden behind DRAM; the kernel is purely
  read/pacing bound. Nothing on the compute side is on the critical path.

### 4. Deeper MLP doesn't help
- reg_pref (manual 2-page register prefetch, 16 uint4/thread in flight): identical to
  kv_sync (1 page in flight). Case13 188-189us both. => 2 pages in flight per CTA adds
  nothing; occupancy-limited cross-CTA MLP is already saturated.
- stagger probe (per-page sync + D dummy-ALU cycles to desync CTAs): monotonic LOSS
  (case13 189->224us @ d160, case14 118->172us @ d160). Adding compute between syncs
  reduces effective read rate; the kernel's own cadence is already optimal.

### 5. ns (split count) already optimal
Kernel ns variants (real kernel, full MMA): case13 ns90 best (175us), ns128 210, ns180
193. case14 ns100 best (118), ns128 121, ns180 129, ns256 157.
Pure-read likes ns180-256 (more CTAs) but the real kernel's overheads grow; the probe
headroom (case14 pure 107us @ ns180) is NOT realizable because a real kernel at ns180
costs +11us. Dead end confirmed.

## The "kernel exceeds pure-read" mystery — resolved
Pure-read replicas were the WRONG upper bound, as the ceiling doc suspected, but the
reason is now precise: without the per-page barrier, a register-xor reader issues each
CTA's 8-page-slice loads spread out; with the barrier (kv_sync), 1.28TB/s case13 /
1.13 case14. The REAL kernel (175us) still beats kv_sync (186us) on case13 — the only
remaining difference is the kernel's higher register count + smem + MMA cadence, which
evidently schedules the LDG stream better (150 regs = load-pacing optimum, per history;
no_mma==real shows it is NOT the MMA itself). On case14 the kernel (118) is at/above
its pure-read pacing bound (112-116). No structure I tried closes or beats it.

## VERDICT
Family (b) — wider/contiguous DRAM transaction restructure — is comprehensively dead.
Every concrete micro-probe (contig-per-CTA strips, 512B slices, whole-page, deeper reg
prefetch, desync stagger, ns changes, STS variants) either ties or loses to the current
kv-sliced page-sweep. The kernel's read path is at its structural ceiling: paced per-
page dense bursts at 7 CTAs/SM, all compute hidden, ns optimal. The only untapped
headroom to the ~1.55TB/s absolute ceiling requires either (c)-family algorithmic
change (fewer DRAM bytes or L2 reuse across heads/splits) or a machine-level (more
threads/CTAs resident) lever that history has closed. Do NOT pursue (b).

## Artifacts
probe_ab.log (contig vs kv vs smem_pace vs strip4k vs slice512/4, case13/14)
probe_pace_ab.log (kv vs +sync vs smem_pace2)
probe_pace2_ab.log (with-sync shape scan + ns scan)
pref_ab.log (kv_sync vs reg_pref)
stag_ab.log (sync + dummy delay sweep)
sts_ab.log (LDG->STS minimal vs kv xor)
epi_ab.log (real vs epilogue-stripped)
mma_ab.log (real vs no_mma)
ns_ab.log (kernel ns variants)
final_ab.log (real vs no_mma vs kv_sync pure, 3 rounds)
