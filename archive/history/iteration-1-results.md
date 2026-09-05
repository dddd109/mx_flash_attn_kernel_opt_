# Skill v1 Iteration Results

## Agent Test Summary
- **Skill v1**: Minimal teaching style, basic guidance
- **Agent result**: 3x speedup (460us → 146us)
- **Issues found**:
  1. Used module-level caching that breaks with different inputs
  2. Only tuned flash_attn parameters, didn't write custom kernel
  3. Sequential block_table doesn't match real test (random permutation)

## What Worked
1. `num_splits=4` for seq >= 4096 is a valid flash_attn tuning
2. Sequential block_table access is faster than random

## What Didn't Work
1. Module-level caching - doesn't work in multi-call OJ scenario
2. Not actually optimizing the kernel itself

## Lessons for Skill v2
1. Must NOT use module-level caching (each call is independent)
2. Must respect the actual block_table passed in (don't assume sequential)
3. Should guide agent to write CUSTOM kernel, not just tune flash_attn params
4. Need to emphasize this is a TEACHING skill - agent should learn optimization principles

## Next Steps
- Create skill v2 that:
  - Emphasizes no module-level caching
  - Guides toward writing custom Triton/CUDA kernel
  - Teaches optimization principles without giving away the answer