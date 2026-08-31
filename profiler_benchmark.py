"""
Profiler-based benchmark for flash_attn baseline and optimized kernel.
Measures GPU kernel time via torch.profiler.
"""
import torch
import math
import ctypes
from flash_attn import flash_attn_with_kvcache


def profile_kernel(fn, warmup=10, reps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()

    return prof


def get_kernel_times(prof):
    """Return dict of kernel name -> total device_time (us)"""
    results = {}
    for evt in prof.key_averages():
        if evt.device_time > 0:
            results[evt.key] = evt.device_time
    return results


# Load optimized kernel
lib = ctypes.CDLL('/root/code/my_optimized_kernel.so')
lib.run_kernel.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 9
lib.run_kernel.restype = None


def run_optimized(q, k_cache, v_cache, output, cache_seqlens, block_table,
                  batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k,
                  headdim, page_block_size, num_blocks, causal):
    lib.run_kernel(
        ctypes.cast(q.data_ptr(), ctypes.c_void_p),
        ctypes.cast(k_cache.data_ptr(), ctypes.c_void_p),
        ctypes.cast(v_cache.data_ptr(), ctypes.c_void_p),
        ctypes.cast(output.data_ptr(), ctypes.c_void_p),
        ctypes.cast(cache_seqlens.data_ptr(), ctypes.c_void_p),
        ctypes.cast(block_table.data_ptr(), ctypes.c_void_p),
        batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k,
        headdim, page_block_size, num_blocks, causal)


def setup_case(batch_size, seqlen_k, num_heads_k):
    headdim = 128
    num_heads = 32
    page_block_size = 16
    seqlen_q = 1
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
    output = torch.zeros(batch_size, seqlen_q, num_heads, headdim, dtype=torch.bfloat16, device='cuda')
    return (q, k_cache, v_cache, output, cache_seqlens, block_table,
            batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k,
            headdim, page_block_size, num_blocks)


def main():
    test_cases = [
        (1, 4, 8, 4),
        (2, 4, 2, 8),
        (3, 16, 17, 8),
        (4, 64, 64, 4),
        (7, 64, 2048, 4),
        (8, 16, 4096, 4),
        (10, 1, 8192, 4),
        (11, 16, 12251, 4),
        (12, 8, 32768, 4),
        (13, 1, 58966, 4),
    ]

    print("=" * 78)
    print("Profiler-based benchmark: flash_attn vs optimized (my_optimized_kernel.so)")
    print("=" * 78)
    print(f"{'case':>5} {'batch':>6} {'seqlen_k':>9} {'nkv':>4} {'flash_us':>10} {'opt_us':>10} {'ratio':>8}")
    print("-" * 78)

    for case_num, batch_size, seqlen_k, num_heads_k in test_cases:
        data = setup_case(batch_size, seqlen_k, num_heads_k)
        q, k_cache, v_cache, output, cache_seqlens, block_table = data[:6]
        batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k = data[6:11]
        headdim, page_block_size, num_blocks = data[11:14]

        # Flash attention
        def run_flash():
            flash_attn_with_kvcache(
                q, k_cache, v_cache, None, None,
                cache_seqlens=cache_seqlens, block_table=block_table,
                causal=False, num_splits=0)

        prof_flash = profile_kernel(run_flash, warmup=5, reps=30)
        flash_us = 0
        for k, v in get_kernel_times(prof_flash).items():
            if 'flash' in k.lower():
                flash_us = v / 30.0

        # Optimized kernel
        def run_opt():
            run_optimized(q, k_cache, v_cache, output, cache_seqlens, block_table,
                          batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k,
                          headdim, page_block_size, num_blocks, 0)

        prof_opt = profile_kernel(run_opt, warmup=5, reps=30)
        opt_us = 0
        for k, v in get_kernel_times(prof_opt).items():
            if 'run_kernel' in k.lower() or 'kernel' in k.lower():
                opt_us = v / 30.0
        if opt_us == 0:
            # fallback: sum all device time
            for evt in prof_opt.key_averages():
                if evt.device_time > 0:
                    opt_us += evt.device_time / 30.0

        ratio = flash_us / opt_us if opt_us > 0 else 0
        print(f"{case_num:>5} {batch_size:>6} {seqlen_k:>9} {num_heads_k:>4} "
              f"{flash_us:>10.1f} {opt_us:>10.1f} {ratio:>7.2f}x")


if __name__ == "__main__":
    main()