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