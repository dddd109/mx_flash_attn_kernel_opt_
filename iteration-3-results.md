# Skill v3 Iteration Results

## Agent Test v3 Summary
- **Skill v3**: Emphasized "MUST write custom kernel"
- **Agent result**: Custom Triton kernel, 5us (92x faster), but incorrect output
- **Root cause**: Agent's kernel has a bug in the attention computation

## What Agent Did Right
1. Wrote a custom Triton kernel
2. Used block_table for paged KV access
3. Implemented online softmax
4. Achieved 5us kernel time (excellent performance)

## What Agent Did Wrong
1. **Scalar qk computation**: `qk = tl.sum(q * k)` treats q and k as vectors but tl.sum collapses to scalar
2. **Wrong attention pattern**: For decode, each Q token needs to attend to ALL K tokens, producing a vector
3. **Iterating over pages**: The kernel iterates page-by-page which may cause numerical instability

## Agent's Kernel Code Issue
```python
# Agent wrote:
qk = tl.sum(q * k)  # This is a SCALAR - dot product of full q and k vectors

# Should be:
qk = tl.dot(q, k)  # This gives element-wise product, then sum = scalar
# OR for decode attention where Q has seq_len=1:
# qk is just a scalar since we're computing q @ k^T for one query position
```

Wait, actually for decode attention where seq_len_q=1, q is shape (head_dim,) and k is shape (head_dim,), so `tl.sum(q * k)` IS the correct dot product. The issue might be elsewhere.

## Possible Bug Location
1. **K/V layout after reshape**: May not be correct
2. **Block table indexing**: May not match actual data layout
3. **Softmax normalization**: Division by l_i at the end may not account for numerical stability

## Key Insight
The kernel structure is mostly correct (block_table iteration, online softmax pattern), but there's a subtle bug. The agent needs help with:
1. Verifying K/V layout after reshape
2. Debugging the attention computation step by step

## Lessons for Final Skill
1. Need to guide agent to DEBUG the kernel systematically
2. Provide way to verify intermediate values
3. Explain the K/V data layout more clearly

## Next Steps
- Create final skill v4 with debugging guidance
- Or conclude that current skill is insufficient and more fundamental changes needed