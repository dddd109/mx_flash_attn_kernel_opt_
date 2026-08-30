---
name: paged-gqa-optimizer-final
description: Guide agent to write custom kernel for paged GQA attention - FINAL VERSION
---

# Paged GQA Attention Optimization Guide v4 (FINAL)

## ⚠️ CRITICAL REQUIREMENT ⚠️

**You MUST implement a CUSTOM kernel. flash_attn parameter tuning is NOT acceptable.**

The goal is to learn how to optimize kernels, not just tune parameters.

## Target
- Beat flash_attn baseline (~460us for bs=4, seq=4096)
- Target: ~80-90us with custom kernel

## Data Layout (CRITICAL - Understand This!)

### Input Tensors
```python
# Q: (batch_size, num_qo_heads, head_dim) - bf16
# kv_data: (num_blocks, 2, page_block_size, num_kv_heads, head_dim) - bf16
#   kv_data[:, 0] = K cache
#   kv_data[:, 1] = V cache
# block_table: (batch_size, blocks_per_batch) - int32
#   block_table[b, logical_page] = physical_block_id
# cache_seqlens: (batch_size,) - int32
```

### K/V Flattening
After `k = kv_data[:, 0, :, :, :]`:
- Shape: (num_blocks, page_block_size, num_kv_heads, head_dim)

After `k = k.reshape(num_blocks * page_block_size * num_kv_heads, head_dim)`:
- Shape: (num_blocks * page_block_size * num_kv_heads, head_dim)
- Index: `block_id * page_block_size * num_kv_heads + page_offset * num_kv_heads + kv_head_id`

### Attention Pattern (Decode)
For decode (seq_len_q=1):
- Each query token attends to ALL KV tokens
- Q shape: (num_qo_heads, head_dim) per batch
- K shape: (seq_len_kv, num_kv_heads, head_dim) per batch
- Output shape: (num_qo_heads, head_dim) per batch

## Triton Implementation Guidance

### Kernel Structure
```python
@triton.jit
def kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    block_table_ptr, cache_seqlens_ptr,
    batch_size, seq_len_kv, num_qo_heads, num_kv_heads,
    head_dim, page_block_size, blocks_per_batch,
):
    # Grid: (batch_size * num_qo_heads,)
    # Each instance computes attention for ONE (batch, qo_head)
    
    pid = tl.program_id(0)
    batch_id = pid // num_qo_heads
    qo_head_id = pid % num_qo_heads
    kv_head_id = qo_head_id // (num_qo_heads // num_kv_heads)  # GQA mapping
    
    # Load Q for this head
    # ... load q ...
    
    # Iterate over KV pages
    # For each token in page: compute attention, update online softmax state
    
    # Store final output
```

### Online Softmax Pattern
```python
m_i = -inf  # max of exponentials so far
l_i = 0     # sum of exponentials so far  
acc = 0     # accumulated weighted sum

for each token t:
    s = q @ k_t  # dot product (scalar for decode)
    m_new = max(m_i, s)
    p = exp(s - m_new)
    l_new = l_i * exp(m_i - m_new) + p
    acc = acc * exp(m_i - m_new) + p * v_t
    m_i = m_new
    l_i = l_new

output = acc / l_i
```

### Debugging Tips
1. **Print intermediate values**: Use `tl.debug_print` to see values
2. **Verify K/V access**: Check if you're reading the right indices
3. **Test with small inputs**: Single batch, single head, short sequence
4. **Compare against reference**: Flash_attn output should match

## Common Mistakes

### Mistake 1: Wrong K/V Index Calculation
```python
# WRONG:
offset = block_id * page_block_size * num_kv_heads + t * num_kv_heads

# CORRECT (for flattened layout):
offset = (block_id * page_block_size * num_kv_heads + t * num_kv_heads + kv_head_id) * head_dim
```

### Mistake 2: Ignoring GQA
```python
# Multiple QO heads share same K/V head
kv_head_id = qo_head_id // (num_qo_heads // num_kv_heads)
```

### Mistake 3: Wrong Softmax Normalization
```python
# WRONG: Normalize directly
output = acc / l_i

# CORRECT: Account for numerical stability
output = acc / l_i  # Actually this is correct if using online softmax correctly
```

## Verification Steps
```python
# 1. Run your kernel
your_output = run_kernel(...)

# 2. Get flash_attn reference
ref_output = flash_attn_with_kvcache(...)

# 3. Compare
assert torch.allclose(your_output, ref_output, rtol=1e-2, atol=1e-2)
```

## Profiling
```python
with torch.profiler.profile(...) as prof:
    run_your_kernel(...)
# Check YOUR kernel name in key_averages()
```

## Environment
```bash
export MACA_PATH=/opt/maca/
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH
```