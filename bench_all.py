"""Local benchmark all 14 cases: our kernel vs flash_attn baseline.
Uses SPJ-confirmed case table (HANDOFF/OJ results).
Usage: python3 bench_all.py <lib.so> [reps_mult]
reps_mult scales official iters for timing stability.
"""
import sys, torch, math, ctypes
from flash_attn import flash_attn_with_kvcache

# SPJ-confirmed: (case, batch, seqlen_cap, num_heads_k)
CASES = {
    1: (1, 1, 4), 2: (4, 2, 8), 3: (16, 17, 4), 4: (64, 64, 8),
    5: (16, 141, 4), 6: (16, 362, 8), 7: (64, 2048, 8), 8: (16, 4096, 4),
    9: (32, 4096, 8), 10: (1, 8192, 4), 11: (16, 12251, 4), 12: (8, 32768, 8),
    13: (1, 58966, 8), 14: (1, 61519, 4),
}
OFFICIAL_ITERS = {1:100, 2:100, 3:100, 4:50, 5:50, 6:50, 7:12, 8:25, 9:12, 10:25, 11:12, 12:12, 13:25, 14:25}

def make_case(c):
    batch, seq, kv = CASES[c]
    headdim = 128; nh = 32; pbs = 16
    bpb = math.ceil(seq / pbs)
    nb = batch * bpb
    torch.manual_seed(42)
    if c == 1:
        cs = torch.full((batch,), 1, dtype=torch.int32, device='cuda')
    else:
        cs = torch.randint(1, seq + 1, (batch,), dtype=torch.int32, device='cuda')
        cs[0] = seq
        if batch > 1: cs[1] = 1
    q = torch.randn(batch, 1, nh, headdim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(nb, pbs, kv, headdim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(nb, pbs, kv, headdim, device='cuda', dtype=torch.bfloat16)
    bt = torch.arange(nb, dtype=torch.int32, device='cuda').reshape(batch, bpb)
    out = torch.zeros_like(q)
    return batch, seq, kv, bpb, nb, cs, q, k, v, bt, out

def timeit(fn, warm, iters):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters

def main():
    libpath = sys.argv[1]
    mult = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    lib = ctypes.CDLL(libpath)
    lib.run_kernel.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int64]*9
    lib.run_kernel.restype = None
    flash_ms = {}
    t1 = 0.0
    for c in sorted(CASES):
        batch, seq, kv, bpb, nb, cs, q, k, v, bt, out = make_case(c)
        nh = 32; headdim = 128; pbs = 16
        iters = OFFICIAL_ITERS[c]
        # flash baseline
        tf = timeit(lambda: flash_attn_with_kvcache(q, k, v, None, None,
                     cache_seqlens=cs, block_table=bt, causal=False, num_splits=0), 5, max(iters * mult, 10))
        # ours
        to = timeit(lambda: lib.run_kernel(ctypes.cast(q.data_ptr(), ctypes.c_void_p),
                     ctypes.cast(k.data_ptr(), ctypes.c_void_p),
                     ctypes.cast(v.data_ptr(), ctypes.c_void_p),
                     ctypes.cast(out.data_ptr(), ctypes.c_void_p),
                     ctypes.cast(cs.data_ptr(), ctypes.c_void_p),
                     ctypes.cast(bt.data_ptr(), ctypes.c_void_p),
                     batch, seq, 1, nh, kv, headdim, pbs, nb, 0), 5, max(iters * mult, 10))
        t1 += to
        print(f"case {c:>2} (b{batch:>3} seq{seq:>6} kv{kv}): ours={to:8.3f}us flash={tf:8.3f}us speedup={tf/to:5.2f}x")
    print(f"SUM ours={t1:.1f}us")

if __name__ == '__main__':
    main()
