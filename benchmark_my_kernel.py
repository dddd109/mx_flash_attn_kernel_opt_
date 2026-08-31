"""
Comprehensive benchmark script for the optimized kernel
"""
import torch
import ctypes
import math
import time

# Load the compiled kernel
lib = ctypes.CDLL('/root/code/my_optimized_kernel.so')

# Define the run_kernel function signature
lib.run_kernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.run_kernel.restype = None

def run_kernel_wrapper(
    q, k_cache_paged, v_cache_paged, output,
    cache_seqlens, block_table,
    batch_size, seqlen_k, seqlen_q,
    num_heads, num_heads_k, headdim,
    page_block_size, num_blocks, causal
):
    q_ptr = q.data_ptr()
    k_ptr = k_cache_paged.data_ptr()
    v_ptr = v_cache_paged.data_ptr()
    out_ptr = output.data_ptr()
    cache_ptr = cache_seqlens.data_ptr()
    block_ptr = block_table.data_ptr()

    lib.run_kernel(
        ctypes.cast(q_ptr, ctypes.c_void_p),
        ctypes.cast(k_ptr, ctypes.c_void_p),
        ctypes.cast(v_ptr, ctypes.c_void_p),
        ctypes.cast(out_ptr, ctypes.c_void_p),
        ctypes.cast(cache_ptr, ctypes.c_void_p),
        ctypes.cast(block_ptr, ctypes.c_void_p),
        batch_size, seqlen_k, seqlen_q,
        num_heads, num_heads_k, headdim,
        page_block_size, num_blocks, causal
    )

def run_flash_attn_baseline(q, k_cache, v_cache, block_table, cache_seqlens):
    from flash_attn import flash_attn_with_kvcache

    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, None, None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=None,
        block_table=block_table,
        causal=False,
        window_size=(-1, -1),
        rotary_interleaved=False,
        alibi_slopes=None,
        num_splits=0,
    )
    return out

def benchmark_case(batch_size, seqlen_k, num_heads_k, num_iters=100, warmup=50):
    """Benchmark a specific test case"""
    headdim = 128
    num_heads = 32
    page_block_size = 16
    seqlen_q = 1
    causal = 0

    blocks_per_batch = math.ceil(seqlen_k / page_block_size)
    num_blocks = batch_size * blocks_per_batch

    torch.manual_seed(42)

    cache_seqlens = torch.randint(1, seqlen_k + 1, (batch_size,), dtype=torch.int32, device='cuda')
    cache_seqlens[0] = seqlen_k
    if batch_size > 1:
        cache_seqlens[1] = 1

    q = torch.randn(batch_size, seqlen_q, num_heads, headdim, device='cuda', dtype=torch.bfloat16)
    k_cache = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device='cuda').reshape(batch_size, blocks_per_batch)

    output_my = torch.zeros(batch_size, seqlen_q, num_heads, headdim, dtype=torch.bfloat16, device='cuda')

    def run_my_kernel():
        run_kernel_wrapper(
            q, k_cache, v_cache, output_my,
            cache_seqlens, block_table,
            batch_size, seqlen_k, seqlen_q,
            num_heads, num_heads_k, headdim,
            page_block_size, num_blocks, causal
        )

    def run_flash_attn():
        return run_flash_attn_baseline(q, k_cache, v_cache, block_table, cache_seqlens)

    # Warmup
    for _ in range(warmup):
        run_my_kernel()
    torch.cuda.synchronize()

    # Benchmark my kernel
    times_my = []
    for _ in range(num_iters):
        start = time.time()
        run_my_kernel()
        torch.cuda.synchronize()
        times_my.append((time.time() - start) * 1e6)
    my_time_us = sum(times_my) / len(times_my)

    # Warmup flash_attn
    for _ in range(warmup):
        run_flash_attn()
    torch.cuda.synchronize()

    # Benchmark flash_attn
    times_flash = []
    for _ in range(num_iters):
        start = time.time()
        run_flash_attn()
        torch.cuda.synchronize()
        times_flash.append((time.time() - start) * 1e6)
    flash_time_us = sum(times_flash) / len(times_flash)

    speedup = flash_time_us / my_time_us

    return my_time_us, flash_time_us, speedup

def main():
    print("=" * 70)
    print("Comprehensive Benchmark: My Optimized Kernel vs Flash Attention")
    print("=" * 70)

    # Test Case 8: batch=16, seqlen_k=4096, num_heads_k=4
    print("\nCase 8: batch=16, seqlen_k=4096, num_heads_k=4")
    print("-" * 70)

    my_time, flash_time, speedup = benchmark_case(16, 4096, 4, num_iters=100, warmup=50)

    print(f"  My optimized kernel: {my_time:.2f} us")
    print(f"  Flash Attention baseline: {flash_time:.2f} us")
    print(f"  Speedup: {speedup:.3f}x")

    print("\n" + "=" * 70)
    print("SUMMARY - Case 8")
    print("=" * 70)
    print(f"  Kernel time: {my_time:.2f} us")
    print(f"  Speedup vs baseline: {speedup:.3f}x")
    print("=" * 70)

if __name__ == "__main__":
    main()