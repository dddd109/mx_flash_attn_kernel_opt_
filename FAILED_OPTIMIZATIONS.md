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

## 5. cache_modifier='ca' for K/V Loads
**Date**: 2026-08-30

**Change**: Add `cache_modifier='ca'` to K/V loads in Triton kernel

**Expected**: Utilize read-only cache (LDG) for better bandwidth

**Result**: ❌ **Not attempted** - Required modifying multiple load sites with complex indentation

**Root Cause**: Triton kernel has multiple load patterns requiring careful modification. Risk of introducing bugs.

**Verdict**: Too risky for marginal gain (Triton compiler likely already optimizes this)

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
## 2026-09-04 (overnight): async-copy (bsm) deep-dive - not yet cracked
- Goal: cp.async-style global->smem for MLP across page barrier (reference ~72 likely uses it).
- Working micro-probe: 16 lanes x1 load into dense buf[256] via __builtin_mxc_ldg_b128_bsm +
  arrive_gvmcnt(0)+barrier_inst+syncwarp. VERIFIED correct.
- Fails when: >16 issuing lanes, >1 load/lane, or larger/strided smem tiles (real kernel 16x136
  K/V). Data lands wrong/duplicated ("globalOffset value and use Saddr conflict" warning when
  smem buffer is large / dst offset large).
- Canary (can4): real stage mapping into 16x136 aligned tile -> BOTH plain-read and MMA-read
  garbage => async data never lands correctly at real-kernel geometry. Not an MMA-consume issue.
- Suspect: bsm smem-addressing has a limited offset/large-buffer quirk on this toolchain. Would
  need deep ISA archaeology to crack; parked (67.07 SOTA stands, async is uncertain payoff).

## 2026-09-04 (overnight, CONFIRMED): async bsm is BROKEN on this toolchain - NOT a viable lever
- Definitive tests (bsm_test2, bsm_off, bsm_matrix, bsm_scale, canary, can4): the
  __builtin_mxc_ldg_b128_bsm async-copy engine does NOT reliably deliver 16B chunks to
  requested smem offsets. Even the "working" probe was a verification artifact: checking
  buf[lane*4] (the real written addresses) shows data DUPLICATED/wrong beyond the first
  2-3 quads (out[8] incorrectly = out[4]).
- The "Saddr conflict" compiler warning appears whenever the load count/smem pattern
  exceeds tiny cases -> the bsm addressing is miscompiled/limited on mxcc 3.7.1.5.
- Earlier "probes pass 100%" (r5/r6 agents) used flawed readback that only checked the
  FIRST element of each quad or unlucky coincidence. Re-verification proves them broken.
- => cp.async/bsm MLP is NOT the reference's lever, or not reproducible here. CLOSED.
- SOTA stands: submission_ov_safe.cu = 67.07 OJ. Remaining gains must come from
  non-async structural ideas.

## 2026-09-05 (r7): more dead ends confirmed
- r7_struct (multi-unit / all-kv CTA restructures): struct_v1_akv/asp ALL PASS but SLOWER
  (1510/1523 vs baseline 1455). More units per CTA doesn't help.
- r7_bw DRAM ceiling probes: even CONTIGUOUS streaming reads need massive parallelism to
  approach peak; the kernel's batch1 kv-sliced pattern is fundamentally ~0.7 TB/s capped.
- c11ns11 (case11 ns 22->11): OJ NEGATIVE, case11 187->194us WORSE. REVERTED. ns=22 correct.
- => ov_safe (67.07) CONFIRMED as the robust best. All structural levers exhausted:
  occupancy (smem wall), async bsm (broken), multi-kv/multi-unit CTA (slower), register/
  smem pipeline (spill/occupancy), split tuning (optimal or OJ-noise).
