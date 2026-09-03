"""
Correct benchmark using torch.cuda.Event timing.
Run in SEPARATE process per kernel to avoid interference.
Usage: python3 bench_cuda_event.py flash <case_num> <reps>
       python3 bench_cuda_event.py opt <case_num> <reps>
"""
import sys, torch, math, ctypes
from flash_attn import flash_attn_with_kvcache

# REAL case table (confirmed from SPJ output of xpuoj submissions)
# (case, batch, seqlen_k_cap, num_heads_k)
# type: edge for 1-3, perf for 4-14
CASES = {
    1: (1, 1, 4),      # edge: batch=1, cap=1
    2: (4, 2, 8),      # edge
    3: (16, 17, 4),    # edge
    4: (64, 64, 8),
    5: (16, 141, 4),
    6: (16, 362, 8),
    7: (64, 2048, 8),
    8: (16, 4096, 4),
    9: (32, 4096, 8),
    10: (1, 8192, 4),
    11: (16, 12251, 4),
    12: (8, 32768, 8),
    13: (1, 58966, 8),
    14: (1, 61519, 4),
}

# iters per case (from official table)
CASE_ITERS = {1:100, 2:100, 3:100, 4:50, 5:50, 6:50, 7:12, 8:25, 9:12, 10:25, 11:12, 12:12, 13:25, 14:25}

def main():
    which = sys.argv[1]
    case_num = int(sys.argv[2])
    reps = 100
    if which == 'flash':
        reps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    batch_size, seqlen_k, num_heads_k = CASES[case_num]

    headdim = 128; num_heads = 32; page_block_size = 16
    blocks_per_batch = math.ceil(seqlen_k / page_block_size)
    num_blocks = batch_size * blocks_per_batch
    torch.manual_seed(42)
    # cache_seqlens generation per official semantics:
    # - every case: >=1 sequence pinned at capacity (seqlen_k)
    # - batch>1: also >=1 sequence at length 1
    # - case 1 special: actual cache_seqlens = 1 (all sequences length 1)
    if case_num == 1:
        cache_seqlens = torch.full((batch_size,), 1, dtype=torch.int32, device='cuda')
    else:
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
        libpath = sys.argv[3] if len(sys.argv) > 3 else '/root/code/my_optimized_kernel.so'
        reps = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        lib = ctypes.CDLL(libpath)
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

    # Warmup
    for _ in range(20): run()
    torch.cuda.synchronize()

    # Measure with cuda events
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_evt.record()
    for _ in range(reps): run()
    end_evt.record()
    torch.cuda.synchronize()
    total_ms = start_evt.elapsed_time(end_evt)
    per_us = total_ms * 1000 / reps

    print(f"case={case_num} {which} batch={batch_size} seqlen={seqlen_k} nkv={num_heads_k} per_call={per_us:.2f}us (reps={reps})")

if __name__ == '__main__':
    main()