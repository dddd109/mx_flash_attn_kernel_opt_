# Paged GQA Optimization Session Notes

## Project Context
- **Task**: Optimize paged GQA attention kernel for MetaX C500 GPU
- **Competition**: https://gitlink.org.cn/metax-maca/op_optimization.git (Track 1)
- **OJ Account**: muxi2026C2032@example.com (DO NOT SUBMIT - already submitted many times)
- **Baseline**: FlashAttention/FlashInfer implementations on MetaX-MACA GitHub

## Code Files
| File | Language | Description |
|------|----------|-------------|
| `metax_c500_paged_gqa_decode_attempt.cu` | CUDA | Paged GQA decode kernel |
| `submission_c500_regions.py` | Triton | Multi-region Triton implementation |
| `benchmark_paged_gqa.py` | Python | Benchmark script |
| `profile_paged_gqa.sh` | Bash | Profiling script |

## MetaX Software Stack

### Environment Setup
```bash
export MACA_PATH=/opt/maca/
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export CUDA_PATH=$MACA_PATH/tools/cu-bridge
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH
```

### Version Info
- **MACA SDK**: 3.7.1.5
- **LLVM**: mxgpu_llvm (based on clang/LLVM)
- **Compiler**: `mxcc` (wraps clang for MACA)
- **cu-bridge**: 3.7.1.5 (CUDA compatibility layer)

### Key Libraries & Headers
| Path | Purpose |
|------|---------|
| `/opt/maca/include/flash_attn/` | Official FlashAttention API |
| `/opt/maca/include/mcflashinfer/` | FlashInfer kernels (attention, page, fused_moe) |
| `/opt/maca/include/mctlass/` | CUTLASS-style templates for MACA |
| `/opt/maca/include/mctlass/arch/` | Architecture-specific intrinsics (SM80, etc.) |
| `/opt/maca/lib/` | Prebuilt libraries |

### Architecture-Specific Details (C500/SM80)

#### SIMD/MMA Support
- Located in: `/opt/maca/include/mctlass/arch/maca_mma.h`
- `MacaMma<>` template for matrix multiply-add operations
- Supported shapes: 16x16x16 (int8), 16x16x32 (int8 C6XX), etc.
- Builtin: `__builtin_mxc_mma_16x16x16i8(a, b, c)` for C500

#### Warp Operations
- `__lane_id()` returns lane ID within warp (instead of NVIDIA's `laneId`)
- Located in: `/opt/maca/include/mctlass/arch/arch.h`

#### Memory Operations
- Located in: `/opt/maca/include/mctlass/arch/memory_sm80.h`
- `cp_async` (async copy) available but `MACA_CP_ASYNC_ACTIVATED` is 0 by default
- Cache operations: `CacheOperation::Always`, `Global`, `Streaming`, `LastUse`, `WriteBack`, `WriteThrough`

#### Important Notes
1. **C500 corresponds to SM80 architecture** (Ampere-generation)
2. **No native bf16 MMA** - must use fp32 accumulation with bf16 loads
3. **`__ldg` intrinsic** - available for read-only global memory cache load
4. **Warp shuffle** - `__shfl_*` intrinsics may work differently; check `LaneId()` implementation

## Important Repositories
1. **FlashAttention**: https://github.com/MetaX-MACA/flashattn
2. **FlashInfer**: https://github.com/MetaX-MACA/McFlashInfer
3. **mcTriton**: https://github.com/MetaX-MACA/mcTriton

## Skill File
- Location: `.opencode/skills/paged-gqa-optimizer/SKILL.md`
- Purpose: Guide agent to optimize the kernel iteratively

## Profiler Usage
```bash
mcProfiler perf_exec \
  --cmdline "python3 benchmark_paged_gqa.py" \
  --kernelname "paged_gqa_decode_kernel" \
  --casename "decode_bs4_seq16k" \
  --cwd /root/code \
  --metrics "sm_efficiency,achieved_occupancy,dram_utilization,l2_utilization" \
  --per-kernel --single-pass
```

## Key Metrics
- `sm_efficiency`: SM utilization (target > 80%)
- `achieved_occupancy`: Warp occupancy (target > 60%)
- `dram_utilization`: Memory bandwidth (target > 70%)
- `l2_utilization`: L2 cache hit rate (target > 40%)

## Identified Optimization Points (CUDA)

### 1. Dot Product Reduction (REQUIRES ANALYSIS)
- **Location**: Lines 214-218
- **Issue**: Shared memory reduction for 16 lanes
- **Status**: Warp shuffle does NOT apply directly (tid < Group threads cannot exchange all 16 lane data)
- **Alternative**: Keep shared memory approach but optimize other parts

### 2. bf16 to Float Conversion (MEDIUM Priority)
- **Location**: Lines 195-198
- **Issue**: Separate bf16→float conversion in tight loop
- **Try**: Prefetch Q values into registers before inner loop

### 3. Memory Access Pattern (HIGH Priority)
- **Location**: Lines 166-181
- **Issue**: Indirect memory access through pointer arithmetic
- **Try**: Use `__ldg` intrinsic for read-only cache load

### 4. Online Softmax (LOW Priority)
- **Location**: Lines 211-240
- **Try**: Reduce sync barriers if possible

## Identified Optimization Points (Triton)

### 1. BLOCK_N Selection (HIGH Priority)
- **Location**: `_pick_block_n()` function (lines 882-887)
- **Current**: 16 if seq<=16, 32 if seq<=256, else 64
- **Try**: Adjust thresholds based on C500 memory hierarchy

### 2. NUM_SPLITS Selection (MEDIUM Priority)
- **Location**: `_pick_natural_splits()` function
- **Impact**: Parallelism vs overhead tradeoff for long sequences

### 3. D_TILE Parameter (LOW Priority)
- **Location**: Line 34
- **Current**: 64
- **Try**: Test 32 or 128 for different head_dim

## MetaX-Specific Considerations
1. **Compiler Differences**: MACA's `mxcc` may have different optimization behaviors than NVCC
2. **Memory Hierarchy**: C500 has different L2 cache size/structure than NVIDIA GPUs
3. **Intrinsics**: Some NVIDIA intrinsics may not be available on MACA (e.g., `__shfl_xor_sync`)

## Git Workflow
- Branch: `master` is main branch
- Commit message format: Clear description of change
- Before optimization: Always commit current state
- If change causes regression: `git restore <file>`

## TODO
- [x] Initialize git repository
- [x] Create skill file
- [x] Create benchmark script
- [ ] Create profile script (done)
- [ ] Run baseline profile on GPU (needs模力方舟)
- [ ] Implement first optimization
- [ ] Verify correctness
- [ ] Submit to OJ if improvement significant

## Session Summary
Date: 2026-08-29
Current state: Analyzed code, created skill and scripts, identified optimization points
Next step: Run profile on GPU when available, then implement safe optimizations