---
name: paged-gqa-cuda-optimizer-v6
description: Guide agent to optimize paged GQA attention using CUDA for MetaX C500
---

# Paged GQA Attention Optimization Guide v6

## ⚠️ CRITICAL: Independent Implementation Required ⚠️

You MUST write your OWN kernel. Do NOT copy, paste, or structurally mirror any
existing CUDA implementation file. The evaluation checks that your implementation
is genuinely your own work. You may look at general reference materials, but the
final kernel structure and code must be authored by you.

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

## Optimization Directions to Explore (research these independently)

### 1. Memory Access Efficiency
- Vectorized 16-byte loads (int4) for K/V pages
- Coalesced access patterns across threads
- Consider how to minimize address computation overhead
  (hint: incremental address arithmetic beats per-element indexing)

### 2. GQA Data Reuse
- 8 query heads share each KV head (ratio 8)
- Process a query-head group together to reuse loaded K/V

### 3. Parallelization Strategy
- Long sequences (4096+) need work splitting across SMs
- Consider splitting the KV dimension across parallel waves
- Small batch + long sequence is the hardest case (case 10, 13)
- Large batch + short sequence also matters (case 1-5, 9)

### 4. Shared Memory and Bank Conflicts
- 32 shared memory banks; linear access patterns can conflict
- Explore swizzle/index-transformation patterns
- Test different layouts empirically

### 5. Compute Kernel
- Explore mctlass BF16 MMA primitives (cute/MACA headers)
- Compare MMA vs scalar compute for your workload
- FP32 accumulation, bf16 only at boundaries

### 6. Online Softmax
- Numerically stable streaming softmax for long sequences
- FP32 state (running max, running sum) updated incrementally

## Benchmarking Methodology (IMPORTANT)
```python
# CORRECT: use cuda events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
for _ in range(reps): run()
end.record()
torch.cuda.synchronize()
per_call_us = start.elapsed_time(end) * 1000 / reps

# WRONG: torch.profiler (underreports ~400x on C500)
```

## Verification
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
- Focus on cases where flash is slow (small batch short seq, and batch=1 long seq)

## Empirical Performance Insights (learned from profiling, not answers)

Your kernel's TOTAL score depends on ALL cases. Profile shows:
- The highest-scoring kernels gain most from SHORT sequences (seqlen < 100): they
  achieve ~2.5x faster than flash there (flash is inefficient on tiny workloads due
  to launch overhead and low parallelism).
- For LONG sequences (seqlen 2048+), flash is already near-optimal. You should aim
  to match it there, not dramatically beat it. Do NOT spend too long chasing >1x on
  long sequences - the short-sequence wins matter more for the score.
- The single most impactful tuning lever is your SPLIT COUNT / work partition:
  - Too few splits: SMs idle on small-batch long-seq cases.
  - Too many splits: combine/partial-output traffic dominates on large-batch or
    short-seq cases (partial write+combine is ~32KiB per split per head).
  - Fixed split sizes waste performance across heterogeneous cases. Make the split
    choice ADAPTIVE based on (batch, num_heads_k, seqlen) at runtime.
  - Page-aligned splits (multiple of page_block_size) let you skip boundary checks.
  - On small-batch-long-seq, prefer enough splits to fill all SMs but no more.
  - On large-batch cases, more (batch, head) work already fills SMs -> use few splits.
- A combined single-kernel path (no separate combine kernel) for short sequences is a
  clear win: many short cases spend significant time in a second kernel launch.
- When your KV loads are already vectorized 16B and reuse is good, the next biggest
  win is usually NOT more compute but reducing redundant work: reading each K/V
  page exactly once per query-head group that needs it.

## The "Long Sequence Trap" (most important lesson)

Profiling the top-scoring kernels shows a CRITICAL pattern you must avoid:

**Do NOT aggressively split LONG sequences.** The reference high-scoring kernel runs
case 8 (seqlen=4096) with num_splits=1 - a single fused pass. It matches flash_attn on
long sequences (0.93-1.07x) WITHOUT splitting, by being efficient in a single sweep:

- A 64-thread wave that streams pages sequentially with software pipelining
  (prefetch next page while computing current) can already saturate DRAM bandwidth.
- Splitting adds: a second kernel launch, partial-output writes to global memory
  (~32KiB per split), and a combine pass that re-reads everything. For seqlen 4096+
  this overhead often EXCEEDS the parallelism benefit.
- Your current failure mode: over-splitting long sequences (tokens_per_split too
  small), so split/combine overhead dominates and you land at 1.8-1.9x of flash.

**The correct split strategy (empirically):**
- seqlen <= ~64: no split, fused single kernel (write output directly, no combine)
- medium seq (hundreds): modest splits ONLY if (batch x heads) doesn't fill SMs
- long seq (4096+): prefer num_splits=1 with a well-pipelined single sweep that
  streams pages; only split if profiling shows SMs are idle
- Split boundaries page-aligned (multiple of 16)
- When batch x num_heads_k >= ~80 (fills 104 SMs), ALWAYS use few splits - you
  already have enough parallelism from the batch dimension

## Occupancy is what hides latency, not intra-block prefetch

Another empirical lesson: on C500, DRAM latency is hidden by having MANY resident
blocks per SM (high occupancy), NOT by software double-buffering inside a single block.

- A single 64-thread block only reaches ~750 GB/s. You need ~1000+ blocks total
  (high occupancy across 104 SMs) to approach >4 TB/s.
- A register-level prefetch buffer that forces 3 blocks/SM can be SLOWER than a
  leaner shared-memory staging that allows 7 blocks/SM.
- Measure your achieved occupancy. If it's low (<4 blocks/SM), reduce per-block
  shared memory / registers to raise it before adding complex pipelining.

## Short-sequence launch overhead (another scoring lever)

For tiny workloads (seqlen < ~100, or few (batch x head) pairs), total time is
DOMINATED by kernel launch overhead (~10-30us on C500) and by a SECOND kernel launch
(combine). The reference achieves 0.40x of flash on these BECAUSE it minimizes:

- Number of kernel launches (avoid launching a separate combine kernel when possible)
- Grid/block setup cost (fixed grids, precomputed host-side configuration)
- Any per-call host-side work (allocations, copies) - do them outside the timed region

If your short-seq cases are 0.6-0.8x of flash and the reference is 0.40x, the gap is
most likely: you're launching a combine kernel you could fuse, or your grid is
oversized for the tiny workload.

### Launch overhead is the hard floor for tiny workloads
Measured on C500: a SINGLE kernel launch costs ~4us (empty kernel), and a torch
elementwise op costs ~13us (includes dispatch). For cases with seqlen < ~100 and
small (batch x heads), total GPU work is only 5-15us - the launch itself dominates.

Implications:
- On tiny cases you CANNOT go below ~4-12us. The reference hits 12.5us which is
  essentially launch-floor-limited.
- The biggest lever is MINIMIZING HOST-SIDE WORK inside run_kernel:
  - Avoid getenv() / atoi() / file I/O on the hot path (these add CPU us).
  - Precompute everything possible; a small branch table is faster than strcmp/getenv.
  - Avoid cudaMalloc/cudaFree on every call - cache buffers and reuse.
  - Keep the number of kernel launches at 1 for the common short path.
- Do NOT try to micro-optimize the GPU kernel for tiny cases - the launch floor
  dominates. Focus there only on: 1 launch, minimal host logic.

### Realistic targets (launch floor aware)
- seqlen <= ~64: target ~12-16us (launch floor), reference ~12.5us
- seqlen ~100-200: target ~20-35us
- long seq: target ~1.0x of flash
If your tiny cases are at 20us and reference is 12.5us, the gap is host-side
launch overhead, not the GPU kernel.

## Teacher Feedback from Previous Student Iterations

Previous students got to 51/63. Here is what held them back (this is feedback, not code):

1. **Tiny-case host overhead**: Their run_kernel called getenv("GQA_NS") + atoi + a complex
   split-decision branch + cudaMalloc/cudaFree check on EVERY call. On cases 2/9 (~12us GPU
   work), this host CPU work added ~7us of measured latency. The reference run_kernel is:
   parameter validation, one split-choice function call, then launches. Keep yours lean.
   Read env/config ONCE (or hardcode the policy), not per call.

2. **Combine launch on medium cases**: They launched main+combine for cases 5/7 where a
   well-chosen single split would suffice. Fewer launches = less time on medium workloads.

3. **What they did RIGHT (keep doing)**: MMA via the raw intrinsic with reverse-engineered
   fragments, warp-shuffle softmax stats, register-cached Q, page-aligned adaptive splits,
   high occupancy (7 blocks/SM), fused single-split path. These are correct directions.

4. **Measurement trap**: They trusted their own benchmark loop timing during iteration, which
   differed from the authoritative cuda_event measurement. Always report BOTH and reconcile.

## Teacher Feedback - Generation 2 (reached 61/63)

Gen2 fixed the host overhead and hit 61/63. Remaining gaps (measured best_of_3):

| case | shape | gen2 | ref | gap |
|------|-------|------|-----|-----|
| 9 | batch=32, seq=8, kv=4 | 22.6us | 12.65us | **MAIN GAP** |
| 7 | batch=64, seq=2048, kv=4 | 158us | 151us | small |
| 12 | batch=8, seq=32768, kv=4 | 333us | 307us | small |
| 13 | batch=1, seq=58966, kv=4 | 173us | 165us | small |

### Case 9 analysis (the one clear opportunity)
- 128 (batch, kv) blocks, each processes 8 tokens (1 partial page of 16).
- Gen2 is 0.74x of flash; reference is 0.42x. Gap ~10us.
- The kernel loads a FULL page (16 tokens) into shared memory but only 8 are valid.
- Question to investigate: is the per-block work (load 16 tokens, mask to 8, 8 MMA steps)
  dominating over the launch floor? Compare against an approach that skips the masked
  half of the page.
- Note: seq=8 means only 8 of 16 tokens are valid. A "half-page" specialization
  (only load/store the 8 valid tokens) could cut shared-memory traffic in half.
- Also consider: does gen2 launch the fused path but still touch unused tail? If the
  kernel iterates 16-token pages but seqlen is 8, half the loads are wasted.

### What gen2 did right (keep)
- Pure-arithmetic split policy, no getenv/atoi
- ns==1 fused path returns before any buffer alloc
- High occupancy, register-cached Q, warp-shuffle stats

### Next target for gen3
Close case 9 (22.6 -> ~13us). That alone moves 61 -> ~62.
If case 12/13 also close (~1.0x), 63 is reachable.

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