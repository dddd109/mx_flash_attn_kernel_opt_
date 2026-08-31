---
name: paged-gqa-cuda-optimizer
description: Guide agent to optimize paged GQA attention using CUDA/mctlass for MetaX C500
---

# Paged GQA Attention Optimization Guide v5 (CUDA/mctlass)

## ⚠️ CRITICAL: This is a CUDA/mctlass Optimization Task ⚠️

You MUST implement a CUDA kernel using mctlass library, NOT just tune Triton or flash_attn parameters.

## Your Goal
Optimize paged GQA attention decode kernel to match or exceed the performance of optimized_c500_flash_attn.cu (v9).

## Reference Implementation
Study `/root/code/optimized_c500_flash_attn.cu` to understand the optimization techniques used.

## Target Performance
- Beat flash_attn baseline (~180us for case 8: batch=16, seq=4096)
- Match optimized_c500_flash_attn.cu performance (~1800us is the OJ target, but local testing shows ~200us achievable)

## Test Configuration (from task.md)
```python
# Case 8 (typical perf case):
batch_size = 16
seqlen_k = 4096
num_heads_k = 4  # or 8
headdim = 128
num_heads = 32
page_block_size = 16
seqlen_q = 1
causal = 0
num_blocks = batch_size * ceil(seqlen_k / page_block_size)  # NO 3x redundancy!
```

## Data Layout (CRITICAL)

### Input Tensors
```python
# q: (batch_size, seqlen_q=1, num_heads, headdim) - bf16
# k_cache_paged: (num_blocks, page_block_size, num_heads_k, headdim) - bf16
# v_cache_paged: (num_blocks, page_block_size, num_heads_k, headdim) - bf16
# cache_seqlens: (batch_size,) - int32 (actual KV length per batch)
# block_table: (batch_size, num_blocks/batch_size) - int32
```

### GQA Mapping
```python
gqa_ratio = num_heads // num_heads_k  # 8 for kv4, 4 for kv8
kv_head = query_head // gqa_ratio
```

### KV Token Location
```python
# Token t (0 <= t < cache_seqlens[b]) is at:
#   block_table[b, t // page_block_size] -> physical page
#   offset in page = t % page_block_size
```

## CUDA/mctlass Implementation Tips

### Key Components from optimized_c500_flash_attn.cu

1. **BF16 MMA Operation**
```cpp
using MctBfloat16 = mctlass::bfloat16_t;
using MmaOp = cute::MACA_16x16x16_F32BF16BF16F32;
// Use MmaOp::fma for bf16 matrix multiplication
```

2. **Shared Memory Swizzle for K/V**
```cpp
// K uses Swizzle<3,3,3> pattern:
// slot = tile * 1024 + (row*64 + col) ^ ((row*64 + col >> 6) << 3)

// V uses XOR swizzle:
// slot = token * 128 + (d ^ (token << 3))
```

3. **Warp-level Operations**
```cpp
// C500 DOES support __shfl_xor_sync unlike earlier assumed!
// Use for reduction within warp
value = __shfl_xor_sync(uint64_t(-1), value, 16);
value = __shfl_xor_sync(uint64_t(-1), value, 32);
```

4. **Split-KV for Long Sequences**
```cpp
// Split KV across multiple waves for parallelism
// Each wave processes a portion of KV tokens
```

## Environment Setup
```bash
export MACA_PATH=/opt/maca/
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH
```

## Compilation
```bash
/opt/maca/mxgpu_llvm/bin/mxcc -march=sm80 -c your_kernel.cu -o your_kernel.o
```

## Verification
```python
# Compare against flash_attn output
assert torch.allclose(your_output, flash_output, rtol=1e-2, atol=1e-2)
```

## Key Optimizations to Study

From optimized_c500_flash_attn.cu:
1. **MMA-based computation** (not scalar bf16→float→fmaf)
2. **Shared memory swizzle** for bank conflict avoidance
3. **Warp shuffle reductions** for online softmax
4. **Split-KV parallelization** for long sequences
5. **Affine address calculation** (precompute base, increment by fixed stride)

## Common Mistakes

1. **DON'T use scalar bf16 operations** - Use MMA when possible
2. **DON'T ignore GQA** - Multiple query heads share same KV head
3. **DON'T use sequential page access** - Use block_table for actual mapping
4. **DON'T forget cache_seqlens** - Each batch has different actual length

## Submission
Your `run_kernel` function will be called with the OJ interface:
```cpp
extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output,
    const int32_t* cache_seqlens,
    const int32_t* block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t num_blocks,
    int64_t causal
);
```