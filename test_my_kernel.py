"""
Test script to verify and benchmark the optimized kernel
"""
import torch
import ctypes
import math
import time

# Load the compiled kernel
lib = ctypes.CDLL('/root/code/my_optimized_kernel.so')

# Define the run_kernel function signature
lib.run_kernel.argtypes = [
    ctypes.c_void_p,  # q
    ctypes.c_void_p,  # k_cache_paged
    ctypes.c_void_p,  # v_cache_paged
    ctypes.c_void_p,  # output
    ctypes.c_void_p,  # cache_seqlens
    ctypes.c_void_p,  # block_table
    ctypes.c_int64,   # batch_size
    ctypes.c_int64,   # seqlen_k
    ctypes.c_int64,   # seqlen_q
    ctypes.c_int64,   # num_heads
    ctypes.c_int64,   # num_heads_k
    ctypes.c_int64,   # headdim
    ctypes.c_int64,   # page_block_size
    ctypes.c_int64,   # num_blocks
    ctypes.c_int64,   # causal
]
lib.run_kernel.restype = None

def run_kernel_wrapper(
    q, k_cache_paged, v_cache_paged, output,
    cache_seqlens, block_table,
    batch_size, seqlen_k, seqlen_q,
    num_heads, num_heads_k, headdim,
    page_block_size, num_blocks, causal
):
    """Wrapper to call the CUDA kernel"""
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
    """Flash_attn baseline for comparison"""
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

def verify_correctness(batch_size, seqlen_k, num_heads_k, headdim=128, num_heads=32, page_block_size=16):
    """Verify correctness against flash_attn baseline"""
    seqlen_q = 1
    causal = 0
    blocks_per_batch = math.ceil(seqlen_k / page_block_size)
    num_blocks = batch_size * blocks_per_batch

    torch.manual_seed(42)

    # cache_seqlens - at least one sequence reaches capacity
    cache_seqlens = torch.randint(1, seqlen_k + 1, (batch_size,), dtype=torch.int32, device='cuda')
    cache_seqlens[0] = seqlen_k  # First sequence uses full capacity

    # Q: (batch, seqlen_q=1, num_heads, headdim)
    q = torch.randn(batch_size, seqlen_q, num_heads, headdim, device='cuda', dtype=torch.bfloat16)

    # K/V cache: (num_blocks, page_block_size, num_heads_k, headdim)
    k_cache = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)

    # block_table: (batch, blocks_per_batch) - consecutive mapping
    block_table = torch.arange(num_blocks, dtype=torch.int32, device='cuda').reshape(batch_size, blocks_per_batch)

    # Output buffers
    output_my = torch.zeros(batch_size, seqlen_q, num_heads, headdim, dtype=torch.bfloat16, device='cuda')
    output_flash = torch.zeros_like(output_my)

    # Run my kernel
    run_kernel_wrapper(
        q, k_cache, v_cache, output_my,
        cache_seqlens, block_table,
        batch_size, seqlen_k, seqlen_q,
        num_heads, num_heads_k, headdim,
        page_block_size, num_blocks, causal
    )

    # Run flash_attn baseline
    output_flash = run_flash_attn_baseline(q, k_cache, v_cache, block_table, cache_seqlens)

    # Verify
    diff = torch.abs(output_my - output_flash)
    max_diff = torch.max(diff).item()
    relative_diff = diff / (torch.abs(output_flash) + 1e-8)
    max_relative_diff = torch.max(relative_diff).item()

    print(f"Verify correctness: batch={batch_size}, seqlen_k={seqlen_k}, num_heads_k={num_heads_k}")
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Max relative diff: {max_relative_diff:.6f}")
    print(f"  Close: {torch.allclose(output_my, output_flash, rtol=1e-2, atol=1e-2)}")

    return torch.allclose(output_my, output_flash, rtol=1e-2, atol=1e-2)

def benchmark_case(case_num, batch_size, seqlen_k, num_heads_k, num_iters=100, warmup=10):
    """Benchmark a specific test case"""
    headdim = 128
    num_heads = 32
    page_block_size = 16
    seqlen_q = 1
    causal = 0

    blocks_per_batch = math.ceil(seqlen_k / page_block_size)
    num_blocks = batch_size * blocks_per_batch

    torch.manual_seed(42)

    # cache_seqlens
    cache_seqlens = torch.randint(1, seqlen_k + 1, (batch_size,), dtype=torch.int32, device='cuda')
    cache_seqlens[0] = seqlen_k

    # Q
    q = torch.randn(batch_size, seqlen_q, num_heads, headdim, device='cuda', dtype=torch.bfloat16)

    # K/V cache
    k_cache = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)

    # block_table
    block_table = torch.arange(num_blocks, dtype=torch.int32, device='cuda').reshape(batch_size, blocks_per_batch)

    # Output
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

    # Warmup my kernel
    for _ in range(warmup):
        run_my_kernel()
    torch.cuda.synchronize()

    # Benchmark my kernel
    start = time.time()
    for _ in range(num_iters):
        run_my_kernel()
    torch.cuda.synchronize()
    my_time_us = (time.time() - start) / num_iters * 1e6

    # Warmup flash_attn
    for _ in range(warmup):
        run_flash_attn()
    torch.cuda.synchronize()

    # Benchmark flash_attn
    start = time.time()
    for _ in range(num_iters):
        run_flash_attn()
    torch.cuda.synchronize()
    flash_time_us = (time.time() - start) / num_iters * 1e6

    speedup = flash_time_us / my_time_us

    print(f"Case {case_num}: batch={batch_size}, seqlen_k={seqlen_k}, num_heads_k={num_heads_k}")
    print(f"  My kernel: {my_time_us:.2f} us")
    print(f"  Flash Attention: {flash_time_us:.2f} us")
    print(f"  Speedup: {speedup:.3f}x")

    return my_time_us, flash_time_us, speedup

def main():
    print("=" * 60)
    print("Testing and Benchmarking My Optimized Kernel")
    print("=" * 60)

    # First verify correctness with valid num_heads_k (4 or 8)
    print("\n--- Correctness Verification ---\n")

    # Use small but valid test cases
    test_cases = [
        (4, 8, 4),    # batch=4, seqlen_k=8, num_heads_k=4
        (4, 8, 8),    # batch=4, seqlen_k=8, num_heads_k=8
        (16, 17, 4),  # batch=16, seqlen_k=17, num_heads_k=4
        (16, 4096, 4), # case 8 perf
    ]

    all_passed = True
    for batch, seqlen, nkv in test_cases:
        try:
            if verify_correctness(batch, seqlen, nkv):
                print("  PASS")
            else:
                print("  FAIL")
                all_passed = False
        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False

    if not all_passed:
        print("\nCorrectness verification failed!")
        return

    print("\n--- Benchmarking ---\n")

    # Case 8: batch=16, seqlen_k=4096, num_heads_k=4
    my_time, flash_time, speedup = benchmark_case(8, 16, 4096, 4, num_iters=25, warmup=25)

    print("\n" + "=" * 60)
    print("Case 8 Results:")
    print(f"  Kernel time: {my_time:.2f} us")
    print(f"  Speedup vs baseline: {speedup:.3f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()