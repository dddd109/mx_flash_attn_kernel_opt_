"""Test case-4-like config under various cache_seqlens distributions.
Usage: python3 c4_dist.py <lib.so> <mode>
mode: full (all=64), one (all=1), half (all=32), rand (local style), skew
"""
import torch, math, ctypes, sys
from flash_attn import flash_attn_with_kvcache
lib = ctypes.CDLL(sys.argv[1]); lib.run_kernel.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int64]*9; lib.run_kernel.restype=None
mode = sys.argv[2]
batch,seq,kv=64,64,8
headdim=128;nh=32;pbs=16;bpb=math.ceil(seq/pbs);nb=batch*bpb
torch.manual_seed(42)
if mode=='full': cs=torch.full((batch,),seq,dtype=torch.int32,device='cuda')
elif mode=='one': cs=torch.full((batch,),1,dtype=torch.int32,device='cuda')
elif mode=='half': cs=torch.full((batch,),32,dtype=torch.int32,device='cuda')
elif mode=='rand':
    cs=torch.randint(1,seq+1,(batch,),dtype=torch.int32,device='cuda');cs[0]=seq;cs[1]=1
elif mode=='skew':  # most full, few short
    cs=torch.randint(32,seq+1,(batch,),dtype=torch.int32,device='cuda');cs[0]=seq;cs[1]=1
elif mode=='saw':   # 0,64,1,64,2,63...
    vals=[(i%2)*seq + (i//2 if i%2==0 else seq-(i//2)) for i in range(batch)]
    cs=torch.tensor(vals,dtype=torch.int32,device='cuda')
q=torch.randn(batch,1,nh,headdim,device='cuda',dtype=torch.bfloat16)
k=torch.randn(nb,pbs,kv,headdim,device='cuda',dtype=torch.bfloat16)
v=torch.randn(nb,pbs,kv,headdim,device='cuda',dtype=torch.bfloat16)
bt=torch.arange(nb,dtype=torch.int32,device='cuda').reshape(batch,bpb)
out=torch.zeros_like(q)
def run():
    lib.run_kernel(ctypes.cast(q.data_ptr(),ctypes.c_void_p),ctypes.cast(k.data_ptr(),ctypes.c_void_p),ctypes.cast(v.data_ptr(),ctypes.c_void_p),ctypes.cast(out.data_ptr(),ctypes.c_void_p),ctypes.cast(cs.data_ptr(),ctypes.c_void_p),ctypes.cast(bt.data_ptr(),ctypes.c_void_p),batch,seq,1,nh,kv,headdim,pbs,nb,0)
for _ in range(5): run()
torch.cuda.synchronize()
import statistics
res=[]
for r in range(5):
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50): run()
    e.record();torch.cuda.synchronize()
    res.append(s.elapsed_time(e)*1000/50)
print(f"c4-dist={mode}: {statistics.median(res):.1f}us")
