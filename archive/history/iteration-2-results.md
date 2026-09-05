# Skill v2 Iteration Results

## Agent Test v2 Summary
- **Skill v2**: Added CRITICAL rules, emphasized no caching, write custom kernel
- **Agent result**: 5x speedup (460us → 91us)
- **Still using flash_attn**: Agent couldn't get custom Triton kernel correct

## What Worked
1. Proper block_table usage (kv_indices.reshape)
2. num_splits=8 tuning for flash_attn

## What Agent Struggled With
1. Custom Triton kernel had correct shape but wrong values
2. Issues with:
   - Triton break/continue not supported
   - Complex tensor layout/broadcasting
   - tl.dot dimension mismatches

## Key Insight
Agent spent too much time trying to optimize flash_attn parameters. Need to STRONGLY emphasize that flash_attn tuning is NOT the goal - writing a CUSTOM kernel is required.

## Lessons for Skill v3
1. MUST write custom kernel - flash_attn tuning doesn't count
2. Need more specific Triton guidance without giving away answer
3. Provide safer starting point for Triton kernel
4. Maybe suggest CUDA approach instead of Triton

## Next Steps
- Create skill v3 that:
  - STRONGLY emphasizes: "You MUST write a custom Triton or CUDA kernel"
  - Gives more concrete hints on Triton implementation without full answer
  - Provides starting Triton kernel structure
  - Explains the online softmax pattern