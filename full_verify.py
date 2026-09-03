"""Full 14-case correctness + timing for a kernel .so.
Usage: python3 full_verify.py <libpath>
"""
import torch, math, ctypes, sys, time
from flash_attn import flash_attn_with_kvcache

CASES = {1:(4,2,4),2:(4,2,8),3:(16,17,4),4:(16,64,8),5:(16,141,4),6:(16,362,8),
7:(64,2048,4),8:(16,4096,4),9:(32,4096,8),10:(1,8192,4),11:(16,12251,8),
12:(8,32768,8),13:(1,58966,4),14:(1,61519,4)}

lib = ctypes.CDLL(sys.argv[1])
lib.run_kernel.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int64]*9
lib.run_kernel.restype=None

allok = True
for c,(batch,seq,kv) in CASES.items():
    headdim=128; nh=32; pbs=16
    bpb=math.ceil(seq/pbs); nb=batch*bpb
    torch.manual_seed(42)
    if c==1:
        cs=torch.full((batch,),1,dtype=torch.int32,device='cuda')
    else:
        cs=torch.randint(1,seq+1,(batch,),dtype=torch.int32,device='cuda'); cs[0]=seq
        if batch>1: cs[1]=1
    q=torch.randn(batch,1,nh,headdim,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(nb,pbs,kv,headdim,device='cuda',dtype=torch.bfloat16)
    v=torch.randn(nb,pbs,kv,headdim,device='cuda',dtype=torch.bfloat16)
    bt=torch.arange(nb,dtype=torch.int32,device='cuda').reshape(batch,bpb)
    out=torch.zeros_like(q)
    lib.run_kernel(ctypes.cast(q.data_ptr(),ctypes.c_void_p),ctypes.cast(k.data_ptr(),ctypes.c_void_p),
     ctypes.cast(v.data_ptr(),ctypes.c_void_p),ctypes.cast(out.data_ptr(),ctypes.c_void_p),
     ctypes.cast(cs.data_ptr(),ctypes.c_void_p),ctypes.cast(bt.data_ptr(),ctypes.c_void_p),
     batch,seq,1,nh,kv,headdim,pbs,nb,0)
    torch.cuda.synchronize()
    ref=flash_attn_with_kvcache(q,k,v,None,None,cache_seqlens=cs,block_table=bt,causal=False,num_splits=0)
    diff=(out-ref).abs(); tol=1.6e-2+1.6e-2*ref.abs()
    m=((diff<=tol).float().mean().item()); o=bool((diff>8*tol).any().item())
    ok = m>=0.99 and not o
    if c<=3: ok = m>=1.0 and not o  # edge: exact
    if not ok: allok=False
    print(f"case {c:>2}: match={m:.4f} outlier={o} {'OK' if ok else 'FAIL'}")
print(f"\nALL {'PASS' if allok else 'FAIL'}")
