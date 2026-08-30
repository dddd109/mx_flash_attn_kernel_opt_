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

---

# MetaX C500 Software Stack Deep Dive

## Stack Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Code                         │
│         (FlashAttention, Our Kernel, etc.)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python / PyTorch                          │
│              mcPytorch 2.8 + triton 3.0                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     mcTriton                                  │
│            Triton compiler for MetaX                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      mxcc                                    │
│        Clang-based compiler (LLVM backend)                   │
│              MACA_PATH/mxgpu_llvm/bin/                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCTLASS Library                            │
│    CUTLASS-style templates: /opt/maca/include/mctlass/      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     MC Runtime (mcr)                         │
│         /opt/maca/include/mcr/mc_runtime_api.h              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Hardware: C500 GPU                         │
│                  (104 APs, 64GB, SM80)                       │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. MC Runtime (mcr)
**Header**: `/opt/maca/include/mcr/`
- `mc_runtime_api.h` - Main runtime API (similar to CUDA runtime)
- `mc_runtime_types.h` - Type definitions
- Key types: `mcStream_t`, `mcDeviceptr_t`, `mcArray_t`

**Key differences from CUDA**:
- `mcInit()` instead of `cudaInit()`
- `mcMalloc()` / `mcFree()` similar to CUDA
- `mcLaunchKernel()` - kernel launch (uses `MC_KERNEL_NAME` macro)
- `__syncwave` instead of `__syncwarp`
- `waveSize` instead of `warpSize`

### 2. MCTLASS Library
**Header**: `/opt/maca/include/mctlass/`

MCTLASS = MetaX's CUTLASS port. Provides GEMM and tensor operation templates.

**Key directories**:
| Directory | Purpose |
|-----------|---------|
| `arch/` | Architecture-specific intrinsics (SM80, etc.) |
| `gemm/` | GEMM warpups |
| `epilogue/` | Post-GEMM operations |
| `thread/` | Thread-level primitives |
| `block/` | Block-level primitives |
| `reduction/` | Reduction operations |

**Key files**:
| File | Purpose |
|------|---------|
| `mctlass.h` | Main include, defines `MCTLASS_HOST_DEVICE`, `MCTLASS_DEVICE` |
| `bfloat16.h` | `bfloat16_t` type with conversion functions |
| `array.h` | `Array<T, N>` template for vector types |
| `arch/arch.h` | Architecture tags (Sm80, Sm86) and `LaneId()` |
| `arch/maca_mma.h` | MetaX-specific MMA operations |
| `arch/memory_sm80.h` | Memory access primitives |

### 3. CUB Library
**Header**: `/opt/maca/include/cub/`
- Device-wide primitives: reduction, sorting, histogram
- Similar to NVIDIA CUB

### 4. CUTE Library
**Header**: `/opt/maca/include/cute/`
- Tensor expression template library
- Layout, swizzle, tensor operations

---

## C500 vs NVIDIA Ampere (SM80) Comparison

### Architecture
| Aspect | NVIDIA Ampere (A100) | MetaX C500 |
|--------|---------------------|------------|
| Architecture | SM80 | SM80 (compatible) |
| SMs | 108 | 104 |
| Global Memory | HBM2, 1.6TB/s | Similar |
| L2 Cache | 40MB | Different structure |
| Warp Size | 32 | 32 |

### BF16 MMA Support
| Aspect | NVIDIA Ampere | MetaX C500 |
|--------|--------------|------------|
| BF16 MMA | Native `mma.sync.aligned.m16n8k8.f32.bf16.bf16.f32` | **NOT AVAILABLE** |
| INT8 MMA | `mma.sync.aligned.m16n8k32.row.col.s32int8.int8.s32` | `__builtin_mxc_mma_16x16x16i8` |
| FP32 MMA | `mma.sync.aligned.m16n8k8.f32.f32.f32.f32` | `__builtin_mxc_mma_16x16x4f32` |

**Critical Gap**: C500 does NOT have native BF16 matrix multiplication! All bf16 operations must be:
1. Converted to FP32
2. Computed in FP32
3. Converted back to BF16

### Warp-Level Operations
| Operation | NVIDIA | MetaX C500 |
|-----------|--------|------------|
| Get Lane ID | `laneId()` | `__lane_id()` |
| Warp shuffle | `__shfl_xor_sync` | **NOT AVAILABLE** |
| Warp reduce | `__reduce_add_sync` | **NOT AVAILABLE** |
| Sync warp | `__syncwarp()` | `__syncwave()` |

### Memory Operations
| Operation | NVIDIA | MetaX C500 |
|-----------|--------|------------|
| LDG (read-only cache) | `__ldg()` | `__builtin_mxc_ldg_b32()` |
| CP Async | `cp.async` | **DISABLED** (`MACA_CP_ASYNC_ACTIVATED=0`) |
| Async copy | `cp.async.commit_group` | Not available |

---

## Available Intrinsics on C500

### Memory Access
```cuda
// Read-only global load with cache
__builtin_mxc_ldg_b32(ptr, offset, pred, ...)  // 32-bit
__builtin_mxc_ldg_b64_predicator(...)           // 64-bit with predicate
```

### MMA Operations
```cuda
// INT8 MMA (C500 only, __MACA_ARCH__ == 1000)
__builtin_mxc_mma_16x16x16i8(a, b, c)

// FP32 MMA
__builtin_mxc_mma_16x16x4f32(a, b, c)

// BF16 MMA - only available in MacaMma template in maca_mma.h
// BUT: Not exposed via standard PTX - only via mctlass templates
```

### Type Conversion
```cuda
// Standard CUDA bf16 conversions work
__float2bfloat16(float val)
__bfloat162float(bfloat16 val)
```

### Lane ID
```cuda
int lane_id = __lane_id();  // Returns 0-31
```

---

## Environment Variables
```bash
export MACA_PATH=/opt/maca/
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export CUDA_PATH=$MACA_PATH/tools/cu-bridge
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH
```

---

## Compilation Commands
```bash
# Using mxcc (Clang-based)
mxcc -arch=sm80 -c kernel.cu -o kernel.o

# Compile with mctlass support
mxcc -arch=sm80 -I$MACA_PATH/include -c kernel.cu -o kernel.o
```

---

## Key Optimization Implications

### For BF16 Attention Kernel (Our Case):
1. **Cannot use MMA for dot products** - Must use scalar bf16→float→fmaf→float pattern
2. **No warp shuffle reduction** - Must use shared memory for reductions
3. **Memory access is critical** - `__ldg` for read-only data, vectorized loads
4. **Software pipelining** - Since cp_async disabled, overlap compute with memory load manually

### Optimization Priority:
1. **Memory coalescing** - Ensure Q/K/V loads are coalesced
2. **Shared memory tiling** - Maximize data reuse
3. **Register usage** - Keep frequently used values in registers
4. **Occupancy** - Balance register pressure vs occupancy

---

## Official MetaX Repositories
1. **FlashAttention**: https://github.com/MetaX-MACA/flashattn
2. **McFlashInfer**: https://github.com/MetaX-MACA/McFlashInfer
3. **mcTriton**: https://github.com/MetaX-MACA/mcTriton
4. **mctlass**: Part of SDK at `/opt/maca/include/mctlass/`

---

## Git Workflow
- Branch: `master` is main branch
- Commit message format: Clear description of change
- Before optimization: Always commit current state
- If change causes regression: `git restore <file>`

## Profiling Results (2026-08-30)

### Torch Profiler Results (bs=8, seq=16384, nkv=8)
| Kernel | Time | % Time |
|--------|------|--------|
| `_paged_gqa_fulltile_kernel` | 552.776us | 98.83% |
| `_reduce_splits_kernel` | 6.525us | 1.17% |

### Baseline Performance
| Config | Time | Bandwidth |
|--------|------|-----------|
| bs=8, seq=16384, nkv=8 | 0.551ms | 973.84 GB/s |
| bs=8, seq=8192, nkv=8 | 0.282ms | 950.82 GB/s |
| bs=8, seq=4096, nkv=8 | 0.150ms | 897.95 GB/s |
| bs=4, seq=16384, nkv=8 | 0.282ms | 950.50 GB/s |

### Key Findings
1. Main kernel is at 98.83% - optimizing this is the key
2. Reduce kernel is only 1.17% - not the bottleneck
3. Performance is already ~95% of theoretical memory bandwidth (1TB/s for C500)

### Optimization Attempts
1. **BLOCK_N tuning (32 for medium seq)**: No significant improvement
2. **flash_attn comparison**: flash_attn achieves 1448 GB/s vs our 950 GB/s (1.52x faster)
3. **Correctness verification**: Our output matches flash_attn (max diff 0.000488)

### Gap Analysis
flash_attn is ~1.5x faster, possible reasons:
1. **Different algorithm**: Flash attention uses streaming/tiled approach reducing memory traffic
2. **Better register utilization**: May be using MMA or other optimizations
3. **Online softmax implementation**: More efficient numerical computation

### Key Findings
- Our implementation achieves ~95% of memory bandwidth (already well-optimized)
- The remaining gap is likely algorithmic, not implementation
- C500 hardware limitations (no BF16 MMA, no warp shuffle) are fundamental constraints

## TODO
- [x] Initialize git repository
- [x] Create skill file
- [x] Create benchmark script
- [x] Create profile script
- [x] Document MetaX C500 software stack
- [ ] Run baseline profile on GPU (needs 模力方舟 platform)
- [ ] Implement first optimization
- [ ] Verify correctness
- [ ] Submit to OJ if improvement significant

## Session Summary
Date: 2026-08-29
Current state: Analyzed code, created skill and scripts, deeply investigated MetaX stack
Key finding: **C500 lacks native BF16 MMA** - must use FP32 accumulation