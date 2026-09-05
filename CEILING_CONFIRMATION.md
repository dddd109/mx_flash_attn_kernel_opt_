# CEILING CONFIRMATION — 2026-09-05 (main session, decisive)

User asked: confirm we are truly at the ceiling BEFORE restructuring. First place
moved to 72.79 (we are 67.36, gap +5.4pt). This doc records the decisive evidence.

## Method
- Same-process, interleaved, CUDA-event burst timing (bench_all methodology),
  in a clean machine window (baseline case13 175us / case14 117us re-confirmed).
- Probes: ceiling_probe.cu (k_contig streaming read + k_kvgrid kv-sliced page sweep
  replicating the kernel's exact addressing). Working sets = real case13 (kv8,
  241.6MB) and case14 (kv4, 126MB) K+V.
- Workspace: /tmp/agent_ws/ceiling/  (ceiling_ab.py, contig_ceil.py, kv4_bandwidth.py,
  dram_ceil.py, ceiling_probe.cu/.so, baseline.cu/.so)

## Absolute hardware read ceiling (2GB contiguous stream)
  184K..1.47M threads: 1.50-1.55 TB/s plateau => ~1.55 TB/s is the DRAM read ceiling
  of this (shared, sliced) C500.

## Working-set ceilings vs the real kernel (clean state)
case13 (kv8, 242MB):  contig on 242MB max ~1.22 TB/s (>=184K thr)
                      kvgrid ns90-180 max ~1.23
                      REAL KERNEL 175us = 1.38 TB/s   <-- EXCEEDS every pure-read replica
case14 (kv4, 126MB):  contig on 126MB max ~0.99 (>=184K thr; 126MB too small to keep
                      all channels busy with few loaders)
                      kvgrid ns100-720 thr128-512 max ~0.98
                      REAL KERNEL 117us = 1.08 TB/s   <-- EXCEEDS every pure-read replica

## Interpretation (why kernel > pure-read replicas)
Pure-read replicas read each unique byte exactly once with NO smem reuse.  The real
kernel stages each page into smem; at 7 CTAs/SM a page's bytes are loaded once into
smem and the ~3.5 resident-split CTAs share *different slices*, so no L2 same-line
reuse (already proven).  The kernel beating pure-read on TB/s means the pure replicas
are NOT the right upper bound: the kernel's smem-mediated pacing + 8 CTAs/SM issue
concurrently delivers higher effective read rate than a thread doing raw LDG->reg.
=> Both big cases read at/past their achievable pattern rate. No read-side slack.

## Cross-checks that bound the remaining search space
- L2 (8MB): single-pass kv-sliced read has no same-line reuse (spec_read notes):
  L2 cannot help case13/14 in current structure; even fully-L2-hot caps ~2.0 TB/s.
- Case14's "low" 1.08 vs case13's 1.38 is a WORKING-SET / slice-width artifact
  (kv4 has half the slices; 126MB set saturates the memory system worse), NOT a
  kernel inefficiency. ns=100 is already the knee (ns<100 monotone worse).
- Occupancy A/B (session4): forcing fewer regs (-maxrregcount) = catastrophic
  (+82%); 150 regs is the load-pacing optimum. Register/smem 2-page prefetch:
  mechanically closed.

## VERDICT
- The current kernel's data-movement is at its architectural ceiling.
- 67.36 -> 72.79 (+5.4pt) cannot come from read-side or register-side or L2-side
  tuning of THIS structure.  It requires a structurally different dataflow that
  either (a) reads fewer DRAM bytes per useful FLOP (impossible: K/V read once),
  (b) converts the kv-sliced read into wider/contiguous DRAM transactions WITHOUT
  losing the smem-mediated MLP (all historical big-CTA/warp-spec attempts lost to
  occupancy/sync), or (c) changes the algorithm (e.g. cross-head Q sharing, or
  exploiting gqa: q heads within a kv group could share loaded K/V).
- Conclusion: reconstruction is warranted ONLY if a concrete (c)-family or
  (b)-family design survives micro-probe. Spawn specialist agents to hunt those.

## Artifacts
- ceiling_ab.log: interleaved real-vs-contig-vs-kvgrid (burst method)
- contig_ceil.py log: contig ceiling vs thread count on 242MB/126MB
- kv4_bandwidth.py log: kv4-sliced pure-read shape sweep (max ~0.98)
- dram_ceil.py log: 2GB absolute ceiling ~1.55 TB/s
