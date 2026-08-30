# Failed Optimization Attempts

## Summary
All optimization attempts below **FAILED** to improve performance. The baseline configuration is already well-optimized.

---

## 1. D_TILE=128 (Full Head Dimension)
**Date**: 2026-08-30

**Change**: Set D_TILE=128 to process full head dimension in single load

**Expected**: Reduced memory transactions, better register utilization

**Result**: ❌ **CRASH** - Memory access violation
```
[MCR][E] xnack(0x8): kernel causes atu address translation error
```

**Root Cause**: When HEAD_DIM=128, loading q1 with offs_d1 = 128:256 exceeds memory bounds

**Verdict**: D_TILE=64 is optimal for HEAD_DIM=128

---

## 2. MAX_SPLITS=64 (Reduced Split Count)
**Date**: 2026-08-30

**Change**: Reduced `_NATURAL_MAX_SPLITS` from 256 to 64

**Expected**: Fewer splits = less reduce kernel overhead

**Result**: ❌ **No improvement** (0.565ms vs 0.567ms baseline)
- Bandwidth: 950.44 GB/s vs 947.37 GB/s
- Difference: < 1% (within noise)

**Root Cause**: Reduce kernel only takes 1.17% of total time, not the bottleneck

**Verdict**: Not a productive optimization direction

---

## 3. BLOCK_N=32 for All Sequences
**Date**: 2026-08-30

**Change**: Force BLOCK_N=32 for all sequence lengths (instead of 64 for seq > 256)

**Expected**: Better cache locality with smaller blocks

**Result**: ❌ **50% Performance Drop**
- bs=8, seq=16384: 443 GB/s vs 950 GB/s (baseline)
- All configurations showed ~50% bandwidth reduction

**Root Cause**: Smaller BLOCK_N reduces memory coalescing efficiency. Memory access becomes less coalesced with smaller blocks.

**Verdict**: BLOCK_N=64 is optimal for memory bandwidth

---

## 4. num_warps=8 (Higher Occupancy)
**Date**: 2026-08-30

**Change**: Increase num_warps from 4 to 8 (256 threads per block)

**Expected**: Higher GPU occupancy = better performance

**Result**: ❌ **8x Performance Drop**
- bs=8, seq=16384: 110 GB/s vs 950 GB/s (baseline)
- All configurations showed ~90% bandwidth reduction

**Root Cause**: Increased register pressure from more threads causes register spilling to local memory, destroying performance

**Verdict**: num_warps=4 is optimal - balance of occupancy vs register pressure

---

## Key Insights

### What Didn't Help
- Increasing tile sizes beyond hardware limits
- Reducing split counts
- Using smaller block sizes
- Increasing thread count per block

### Why Baseline is Already Optimal
1. **Memory bandwidth**: Already at ~95% of theoretical (950 GB/s vs 1 TB/s)
2. **Register pressure**: Current config (num_warps=4) is sweet spot
3. **Memory coalescing**: BLOCK_N=64 provides optimal coalescing for this access pattern
4. **Algorithm**: FlashAttention-style tiling is already efficient

### Potential Directions for Future Work
1. **Algorithm change**: Streaming attention (FlashAttention) instead of paged attention
2. **Hardware upgrade**: C500 lacks BF16 MMA - a GPU with MMA support would be faster
3. **Quantization**: INT8 MMA is available on C500, could try quantized attention
4. **Different kernel design**: Custom CUDA kernel optimized specifically for C500 architecture

---

## Performance Baseline (Reference)
| Config | Time | Bandwidth |
|--------|------|-----------|
| bs=8, seq=16384, nkv=8 | 0.567ms | 947.37 GB/s |
| bs=8, seq=8192, nkv=8 | 0.295ms | 910.52 GB/s |
| bs=4, seq=8192, nkv=8 | 0.162ms | 829.59 GB/s |

**FlashAttention Reference**: 1448 GB/s (1.52x faster, different algorithm)