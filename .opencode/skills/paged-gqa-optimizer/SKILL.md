---
name: paged-gqa-cuda-optimizer-v13
description: Guide agent to optimize paged GQA attention using CUDA for MetaX C500
---

# Paged GQA Attention Optimization Guide

## ⚠️ CRITICAL: Independent Implementation Required ⚠️

You MUST write your OWN kernel. Do NOT copy, paste, or structurally mirror any
existing CUDA implementation file. The evaluation checks that your implementation
is genuinely your own work. You may look at general reference materials, but the
final kernel structure and code must be authored by you.

## How to use this guide (IMPORTANT)

This document has two parts:
1. **METHODOLOGY** (the true teaching content): how to profile, experiment, verify.
2. **PREVIOUS STUDENTS' EXPLORATION NOTES**: findings that PRIOR students reached by
   their own experiments. These are NOT teacher answers - they are handoff notes from
   previous generations, like a team's engineering log. You should:
   - READ them as hypotheses worth testing, not as truths.
   - VERIFY them with your own measurements before trusting them.
   - Feel free to DISAGREE if your experiments show otherwise (one previous student
     found the "long-seq trap" advice wrong for their kernel structure and was right
     to trust their own measurements).
   - The final kernel must be YOUR work. Using prior students' notes to avoid repeating
     their mistakes is good engineering, not cheating.

## Your Task
Optimize paged GQA attention decode kernel for MetaX C500 GPU, starting from the
flash_attn_with_kvcache baseline. Your goal is to beat the baseline by a significant
margin across all OJ test cases.

## Baseline Reference Performance
Use flash_attn_with_kvcache with `num_splits=0` as your baseline. Measured with
`torch.cuda.Event` timing (NOT torch.profiler - it underreports on C500):

| case | batch | seqlen_k | num_heads_k | flash_time (us) |
|------|-------|----------|-------------|-----------------|
| 1 | 4 | 8 | 4 | 31.5 |
| 2 | 4 | 2 | 8 | 31.7 |
| 3 | 16 | 17 | 8 | 38.2 |
| 4 | 64 | 64 | 4 | 38.6 |
| 5 | 16 | 141 | 8 | 38.9 |
| 7 | 64 | 2048 | 4 | 162.0 |
| 8 | 16 | 4096 | 4 | 90.9 |
| 9 | 32 | 8 | 4 | 31.8 |
| 10 | 1 | 8192 | 4 | 57.4 |
| 11 | 16 | 12251 | 4 | 206.1 |
| 12 | 8 | 32768 | 4 | 320.7 |
| 13 | 1 | 58966 | 4 | 153.4 |

## Task Specification
Read `/root/code/task.md` for the complete OJ interface specification.

## Key Constraints (from task.md)
- num_heads = 32, headdim = 128, page_block_size = 16, causal = 0
- num_heads_k = 4 or 8 (GQA ratio = 8 or 4)
- cache_seqlens[b] varies per batch; only trust this for validity
- block_table padding slots may be valid page IDs - never read them
- num_blocks = batch_size * ceil(seqlen_k / page_block_size)
- Each batch has 1 query token (seqlen_q = 1)

## Data Layout
```
Q: (batch, 1, num_heads, headdim) - bf16
K/V cache: (num_blocks, page_block_size, num_heads_k, headdim) - bf16
block_table: (batch, blocks_per_batch) - int32
cache_seqlens: (batch,) - int32

Token t of batch b:
  page = block_table[b, t // 16]
  offset = t % 16
  kv[page, offset, kv_head, :] where kv_head = query_head // gqa_ratio
```

## Implementation Requirements

1. CUDA C++ with mctlass library
2. Export `run_kernel` exactly as specified in task.md (extern "C")
3. Compile: `/opt/maca/mxgpu_llvm/bin/mxcc -std=c++17 -shared -fPIC -c ...`
4. Verify correctness vs flash_attn (rtol=atol=1.6e-2, >=99% match)
5. **Measure with torch.cuda.Event, NOT torch.profiler**

## Optimization Diagnosis Framework (HW-agnostic thinking)

These are GENERAL questions any GPU kernel optimizer should ask, in priority order.
They do not assume this specific GPU or this specific operator's "answer" - they teach
you how to FIND the answer. Apply them to your own kernel and let the measurements lead.

### A. Where is the time actually going? (always start here)
Before changing anything, classify each case:
- **Launch-bound**: total time ≈ fixed cost of launching kernel(s), independent of work.
  Signature: time barely grows as (batch × heads × seqlen) grows. Optimization lever:
  reduce launch count, reduce host-side work, increase work per launch.
- **Memory-bound**: time grows with bytes moved; SM compute units idle waiting.
  Signature: your measured achieved bandwidth ≈ hardware limit; compute % low.
  Optimization lever: reduce bytes (reuse, avoid re-reads), improve coalescing,
  overlap loads with compute (pipeline), raise occupancy to hide latency.
- **Compute-bound**: time grows with FLOPs; memory is underutilized.
  Signature: high SM utilization, bandwidth far below limit.
  Optimization lever: use hardware math units (MMA/tensor cores) more, reduce
  redundant math, balance instruction mix.
- **Occupancy/latency-bound**: few warps resident, so stalls dominate.
  Signature: time doesn't improve with more parallelism in the same kernel.
  Optimization lever: shrink per-thread/block resources (shared mem, registers) so
  more blocks fit per SM.
Classify EACH case, because the same kernel can be launch-bound on tiny cases and
memory-bound on huge ones.

### B. For a memory-bound kernel: what is the minimum bytes that MUST be read/written?
- Each batch's query reads its own K/V pages once. In GQA, multiple query heads share
  one KV head. Question: does your kernel read each needed byte exactly once per
  query-head group, or once per query head (redundant)?
- Question: are you reading any bytes you don't need (padding, masked-out tokens)?
  Any read of an invalid token is wasted bandwidth.
- Question: are your loads the widest the hardware supports (128-bit)?
- Question: is the block_table indirection (a dependent load: load page id, then load
  that page) serializing your memory stream? How can you overlap the indirection with
  the data loads?

### C. For a compute-bound kernel: are you using the strongest math unit available?
- Does the target GPU expose a fused matrix-multiply unit (tensor core / MMA)? If so,
  does your hot loop use it, or scalar FMA?
- If you don't know the instruction set, read the vendor's headers/ISA docs and find the
  matrix multiply primitive. Its fragment layout must be learned from the docs/headers.
- Question: does your accumulator stay in the widest precision the hardware accumulates
  in, converting only at boundaries?

### D. For a latency/occupancy-bound kernel: what is eating the SM?
- Per-block shared memory, registers, and threads all limit how many blocks run per SM.
  Compute your theoretical max blocks/SM for each, and find which one binds you.
- If shared memory binds: shrink the tile, or avoid staging what can stay in registers.
- If registers bind: reduce per-thread state, or restructure to need less.
- Compare a lean config (more blocks) vs a rich config (bigger tile, fewer blocks) by
  experiment - the winner is hardware-specific and surprising (both directions have been
  observed on this project).

### E. General engineering habits that transfer anywhere
- Change ONE variable at a time; measure all cases before/after; keep or revert.
- Keep an engineering log: hypothesis, experiment, result. It makes you faster at
  knowing what NOT to try again.
- If a case is at the launch floor, do not waste time on its GPU kernel - spend effort
  where the time actually is.

## How to measure (IMPORTANT - torch.profiler underreports on this GPU)
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
for _ in range(reps): run()
end.record()
torch.cuda.synchronize()
per_call_us = start.elapsed_time(end) * 1000 / reps
# For launch-floor cases (very short), take best-of-N runs; single-run numbers are noisy.
```

## How to verify correctness
```python
out = your_run_kernel(...)
ref = flash_attn_with_kvcache(...)
diff = (out - ref).abs()
tol = 1.6e-2 + 1.6e-2 * ref.abs()
assert (diff <= tol).float().mean() >= 0.99   # >=99% match
assert not (diff > 8 * tol).any()              # no 8x outliers
```

## Scoring Context
- Baseline flash_attn = 50 points
- A good optimized kernel scores 60-70+

## How to find what to optimize (experiment-driven)
1. Measure ALL 12 cases first (baseline). Rank them by (flash_time - your_time) / flash_time.
2. The cases where you are FARTHEST from flash are your opportunities - but compute how
   many POINTS each case contributes before spending effort (a case at 30us that you can
   halve is worth ~15us; a case at 300us you can trim 10% is worth ~30us).
3. For the biggest opportunity: PROFILE the kernel's behavior (occupancy, memory throughput,
   launch count) with whatever tooling works on this platform, form a hypothesis, test it
   with ONE variable change, measure all 12 cases before/after, keep or revert.
4. Keep a log of every change: what you tried, the hypothesis, the result (all 12 cases),
   whether you kept it. This is your engineering record.

## PREVIOUS STUDENTS' EXPLORATION NOTES (handoff, NOT teacher answers)

These were discovered by earlier student kernels through their OWN experiments. Treat as
hypotheses: verify before trusting, and trust your measurements over these notes.

### Note set 1: where the score comes from
Prior students observed that TOTAL score depends on ALL cases, and that on SHORT
sequences (seqlen < 100) flash is inefficient (launch overhead + low parallelism),
so there is more relative headroom there. On LONG sequences flash is already strong.
They ALSO observed the opposite caution: over-splitting long sequences made them slower.
Your job is to CONFIRM which of these matter for YOUR kernel, by measuring.

### Note set 2: "long-sequence trap" - DISPUTED, verify yourself
A prior student theorized long sequences should NOT be split (a single pipelined sweep).
But a LATER student measured that for THEIR kernel structure, splitting long sequences
helped (case 7 optimal ~13 splits, case 8 ~43 splits). Both were right for their own
kernels. LESSON: split-count is kernel-structure-dependent. Measure your own optimum by
sweeping split count for each case class. Do not copy either student's conclusion.

### Note set 3: occupancy vs software prefetch (both observed)
One student found that high occupancy (many resident blocks/SM, ~7 blocks/SM) hid DRAM
latency better than intra-block register prefetch, and that a heavy prefetch buffer that
dropped occupancy to 3 blocks/SM was SLOWER. Another student found software pipelining
essential. These are consistent: you want BOTH enough occupancy AND (for long streams)
pipelining. Sweep your shared-memory budget vs occupancy and find the balance for each
case class. A later student who removed the pipeline entirely to gain occupancy on short
cases catastrophically regressed long cases - so the balance is case-dependent.

### Note set 4: launch overhead floor
Prior students measured a single kernel launch ~4us and a torch elementwise ~13us
(including dispatch). For tiny workloads, launch overhead can dominate the GPU work.
They found minimizing host-side work (no getenv/atoi per call, cached buffers, single
kernel launch when possible) moved tiny cases to the launch floor. VERIFY this matters
for YOUR kernel by measuring launch count and host time per call.

### Note set 5: MMA on C500
A student successfully used the raw `__builtin_mxc_mma_16x16x16bf16` intrinsic after
finding `load_matrix_sync` crashes on this hardware (toolchain bug). They reverse-engineered
the fragment layout empirically. This is a research finding worth verifying, not an answer:
the C500 MMA fragment layout must be derived from the headers and your own experiments.

## Handoff: the current best kernel (Generation 2, ~61/63)

The best student kernel so far is available as a starting point:
`previous_generation_kernel.cu` (in your workspace). You may use it as your baseline
and improve it. If you prefer to start from flash_attn again, that is also fine.

### Mistakes prior students made (learn from them, then verify)
1. Host-side overhead per call (getenv/atoi, complex branches, cudaMalloc each call)
   added measurable latency on tiny cases. Keep host logic lean; cache buffers.
2. Launching a separate combine kernel where a single fused launch suffices.
3. Trusting their own benchmark loop timing over the authoritative cuda_event timing.
4. The fatal one (see mandatory regression rule): fixing one case, breaking others.

### Known remaining opportunity areas (from gen2's OWN measurements - verify yourself)
- Small-batch partial pages: a case with batch=32, seqlen=8 has 128 (batch,kv) blocks,
  each processing 8 valid tokens of a 16-token page. The student loaded the full page
  and masked; a partial-page specialization (only load valid tokens) is a hypothesis
  worth testing. NOTE: a later student tried this and broke the long-seq pipeline - so
  if you test it, apply the regression rule strictly.
- batch=1 long sequences (seqlen 8192+): the student was 1.0-1.1x of flash there. See
  if you can tune closer to 1.0x without regressing short cases.

## ⚠️ MANDATORY REGRESSION RULE (methodology, non-negotiable) ⚠️

A prior student's kernel regressed catastrophically (61 -> ~19 score) by fixing ONE
case and breaking the others. This is the single worst mistake an optimizer can make:

1. **After EVERY code change, run ALL 12 cases.** A single-case improvement is
   worthless if it regresses others. Measure before AND after for every case.
2. **If a change improves one case but regresses any long-sequence case by >5%,
   REVERT it.** No single-case win is worth a global regression.
3. **Keep a backup** (e.g. cp your_kernel.cu backup.cu) before starting each change.

## Environment
```bash
export MACA_PATH=/opt/maca/
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH
```

## Compilation Reference
```bash
/opt/maca/mxgpu_llvm/bin/mxcc -std=c++17 -shared -fPIC your_kernel.cu -o your_kernel.so \
    -I/opt/maca/include -I/opt/maca/tools/cu-bridge/include
```