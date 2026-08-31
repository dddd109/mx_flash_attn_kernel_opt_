---
name: paged-gqa-cuda-optimizer-v5
description: Guide agent to optimize paged GQA attention using CUDA for MetaX C500
---

# Paged GQA Attention Optimization Guide v5

## Your Task
Optimize paged GQA attention decode kernel to achieve high performance on MetaX C500 GPU.

## Baseline
Use flash_attn_with_kvcache as your baseline:
```python
out = flash_attn_with_kvcache(
    q, k_cache, v_cache, None, None,
    cache_seqlens=cache_seqlens,
    block_table=block_table,
    causal=False,
    num_splits=0,  # Auto split
)
```

## Target Performance
Study `/root/code/optimized_c500_flash_attn.cu` to understand the optimization target.
Study `/root/code/task.md` to understand the OJ task requirements.

## Key Constraints (from task.md)
- num_heads = 32, headdim = 128, page_block_size = 16
- num_heads_k = 4 or 8 (GQA)
- causal = 0 (decode only)
- block_table padding may contain valid page IDs - only use cache_seqlens to determine validity
- num_blocks = batch_size * ceil(seqlen_k / page_block_size) (NO 3x redundancy)

## Optimization Hints (General Guidance)

### 1. Memory Access Patterns
- **Vectorized loads**: Use int4 (16-byte) loads for K/V data
- **Affine addressing**: Precompute base pointers, use fixed stride iteration
- **Block table access**: Use lane 0 to load, broadcast via warp shuffle
- **Bank conflicts**: Shared memory access patterns matter - consider swizzle patterns

### 2. Computation Strategy
- **MMA operations**: C500 has MMA hardware - explore mctlass library for BF16 MMA
- **Online softmax**: Implement numerically stable softmax incrementally
- **GQA efficiency**: Multiple query heads share KV heads - process together

### 3. Parallelism
- **Split-KV**: For long sequences, split KV across multiple work units
- **Wave-level**: Each wave processes one (batch, KV head, split) combination
- **Occupancy**: Balance threads per block for good SM utilization

### 4. MetaX C500 Specifics
- **Architecture**: SM80-compatible, 104 APs
- **Shared memory**: 64KB per block, 32 banks
- **Warp shuffle**: Available! Use for reductions within warp
- **Useful headers**: `<mctlass/numeric_types.h>`, `<cute/arch/mma_sm80.hpp>`

## Data Layout Reminder
```
Q: (batch, seqlen_q=1, num_heads, headdim)
K/V cache: (num_blocks, page_block_size, num_heads_k, headdim)
block_table: (batch, blocks_per_batch) where blocks_per_batch = ceil(seqlen_k / page_block_size)
cache_seqlens: (batch_size,) - actual KV length per batch

Token t (0 <= t < cache_seqlens[b]) location:
  page = block_table[b, t // page_block_size]
  offset = t % page_block_size
```

## Suggested Implementation Steps

1. **Start with correct baseline**:
   - Implement flash_attn-equivalent using simple loops
   - Verify correctness against flash_attn output

2. **Add vectorized memory access**:
   - Use int4 loads for K/V data
   - Coalesced memory access patterns

3. **Implement shared memory caching**:
   - Load page into shared memory
   - Process all tokens in page before loading next

4. **Add GQA optimization**:
   - Process multiple query heads together
   - Share KV data within group

5. **Explore MMA and split-KV**:
   - Consider using mctlass MMA primitives
   - Split long sequences for parallelism

## Common Pitfalls
- **Wrong block_table indexing**: Remember blocks_per_batch may not equal ceil(seqlen_k/16) exactly
- **Ignoring cache_seqlens**: Each batch element has different actual length
- **Bank conflicts**: Linear access patterns can cause shared memory conflicts
- **Register pressure**: Too many registers causes spills to slow local memory

## Profiling Tips
```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    run_kernel(...)
# Check key_averages() for kernel time breakdown
```

## Verification
```python
assert torch.allclose(output, reference, rtol=1e-2, atol=1e-2)
```

## Environment
```bash
export MACA_PATH=/opt/maca/
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH
```

## Compilation Example
```bash
/opt/maca/mxgpu_llvm/bin/mxcc -std=c++17 -c your_kernel.cu -o your_kernel.o \
    -I/opt/maca/include -I/opt/maca/tools/cu-bridge/include
```