"""
Single-case profiler benchmark. Run in SEPARATE process per kernel to avoid CUDA caching.
Usage: python3 profiler_single.py flash <case_num>
       python3 profiler_single.py opt <case_num>
"""
import sys, torch, math, ctypes
from flash_attn import flash_attn_with_kvcache

CASES = {
    1: (4, 8, 4),
    2: (4, 2, 8),
    3: (16, 17, 8),
    4: (64, 64, 4),
    5: (16, 141, 8),
    7: (64, 2048, 4),
    8: (16, 4096, 4),
    9: (32, 8, 4),
    10: (1, 8192, 4),
    11: (16, 12251, 4),
    12: (8, 32768, 4),
    13: (1, 58966, 4),
}

def main():
    which = sys.argv[1]
    case_num = int(sys.argv[2])
    batch_size, seqlen_k, num_heads_k = CASES[case_num]

    headdim = 128; num_heads = 32; page_block_size = 16
    blocks_per_batch = math.ceil(seqlen_k / page_block_size)
    num_blocks = batch_size * blocks_per_batch
    torch.manual_seed(42)
    cache_seqlens = torch.randint(1, seqlen_k+1, (batch_size,), dtype=torch.int32, device='cuda')
    cache_seqlens[0] = seqlen_k
    if batch_size > 1: cache_seqlens[1] = 1
    q = torch.randn(batch_size, 1, num_heads, headdim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device='cuda', dtype=torch.bfloat16)
    bt = torch.arange(num_blocks, dtype=torch.int32, device='cuda').reshape(batch_size, blocks_per_batch)
    out = torch.zeros_like(q)

    if which == 'flash':
        def run():
            flash_attn_with_kvcache(q, k, v, None, None, cache_seqlens=cache_seqlens,
                                    block_table=bt, causal=False, num_splits=0)
    else:
        lib = ctypes.CDLL('/root/code/my_optimized_kernel.so')
        lib.run_kernel.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int64]*9
        lib.run_kernel.restype = None
        def run():
            lib.run_kernel(ctypes.cast(q.data_ptr(), ctypes.c_void_p),
                ctypes.cast(k.data_ptr(), ctypes.c_void_p),
                ctypes.cast(v.data_ptr(), ctypes.c_void_p),
                ctypes.cast(out.data_ptr(), ctypes.c_void_p),
                ctypes.cast(cache_seqlens.data_ptr(), ctypes.c_void_p),
                ctypes.cast(bt.data_ptr(), ctypes.c_void_p),
                batch_size, seqlen_k, 1, num_heads, num_heads_k, headdim, page_block_size, num_blocks, 0)

    for _ in range(10): run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(30): run()
    torch.cuda.synchronize()

    # Sum all kernel device times (main + combine)
    total = 0
    for evt in prof.key_averages():
        if evt.device_time > 0:
            total += evt.device_time
    print(f"case={case_num} {which} total_kernel_time={total/30:.2f}us")
    # Also print breakdown
    for evt in prof.key_averages():
        if evt.device_time > 0:
            name = evt.key.split('(')[0]
            print(f"  {name}: {evt.device_time/30:.2f}us")

if __name__ == '__main__':
    main()