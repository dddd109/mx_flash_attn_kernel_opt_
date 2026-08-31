"""
Test and profile optimized_c500_flash_attn.cu kernel
"""
import torch
import time
import math

# Define the run_kernel function signature
import ctypes

# Load the compiled kernel
lib = ctypes.CDLL('/root/code/optimized_c500_flash_attn.o')

def run_kernel_wrapper(
    q,  # (batch, 1, num_heads, headdim) bf16
    k_cache,  # (num_blocks, page_block_size, num_heads_k, headdim) bf16
    v_cache,  # (num_blocks, page_block_size, num_heads_k, headdim) bf16
    output,
    cache_seqlens,
    block_table,
    batch_size,
    seqlen_k,
    seqlen_q,
    num_heads,
    num_heads_k,
    headdim,
    page_block_size,
    num_blocks,
    causal
):
    """Wrapper to call the CUDA kernel"""
    # This is a placeholder - actual kernel loading would need proper linking
    pass

def benchmark_flash_attn(q, k_cache, v_cache, block_table, cache_seqlens, iterations=100):
    """Flash_attn baseline for comparison"""
    from flash_attn import flash_attn_with_kvcache

    def run():
        flash_attn_with_kvcache(
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

    # Warmup
    for _ in range(10):
        run()
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        run()
    torch.cuda.synchronize()

    return (time.time() - start) / iterations * 1000  # ms


def main():
    # Test case 8 from task.md
    batch_size = 16
    seqlen_k = 4096
    num_heads_k = 4
    headdim = 128
    num_heads = 32
    page_block_size = 16
    seqlen_q = 1
    causal = 0

    blocks_per_batch = math.ceil(seqlen_k / page_block_size)
    num_blocks = batch_size * blocks_per_batch

    torch.random.manual_seed(42)

    # cache_seqlens
    cache_seqlens = torch.randint(1, seqlen_k + 1, (batch_size,), dtype=torch.int32, device='cuda')
    cache_seqlens[0] = seqlen_k
    cache_seqlens[1] = 1

    # Q: (batch, seqlen_q=1, num_heads, headdim)
    q = torch.randn(batch_size, seqlen_q, num_heads, headdim, device='cuda', dtype=torch.bfloat16)

    # K/V cache: (num_blocks, page_block_size, num_heads_k, headdim)
    k_cache = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)

    # block_table: (batch, blocks_per_batch)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device='cuda').reshape(batch_size, blocks_per_batch)

    output = torch.zeros(batch_size, seqlen_q, num_heads, headdim, dtype=torch.bfloat16, device='cuda')

    # Benchmark flash_attn
    flash_ms = benchmark_flash_attn(q, k_cache, v_cache, block_table, cache_seqlens)

    print(f"Test Case 8: batch={batch_size}, seqlen_k={seqlen_k}, num_heads_k={num_heads_k}")
    print(f"Flash Attention: {flash_ms:.3f}ms ({flash_ms*1000:.1f}us)")
    print(f"\nNote: optimized_c500_flash_attn.cu is a CUDA kernel that needs")
    print(f"to be linked and called via ctypes/CFFI to profile directly.")
    print(f"For now, flash_attn_with_kvcache (num_splits=0) is the baseline.")


if __name__ == "__main__":
    main()