# DECISIONS.md - decision log

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
