"""Test case-11-like (b16 kv4 cap12251) and case-9-like (b32 kv8 cap4096)
under skewed cache_seqlens to reproduce OJ's slower-than-local times."""
import torch, math, ctypes, sys
from flash_attn import flash_attn_with_kvcache
CASES={11:(16,12251,4),9:(32,4096,8)}
c=int(sys.argv[1]); mode=sys.argv[2]
lib=ctypes.CDLL(sys.argv[3]); lib.run_kernel.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int64]*9; lib.run_kernel.restype=None
batch,seq,kv=CASES[c];headdim=128;nh=32;pbs=16;bpb=math.ceil(seq/pbs);nb=batch*bpb
torch.manual_seed(42)
if mode=='local':
    cs=torch.randint(1,seq+1,(batch,),dtype=torch.int32,device='cuda');cs[0]=seq;cs[1]=1
elif mode=='full':
    cs=torch.full((batch,),seq,dtype=torch.int32,device='cuda')
elif mode=='half':
    cs=torch.randint(seq//2,seq+1,(batch,),dtype=torch.int32,device='cuda');cs[0]=seq;cs[1]=seq//2
elif mode=='skew1':  # 1 full + rest short-ish (batch1-heavy skew)
    cs=torch.randint(1,seq//4,(batch,),dtype=torch.int32,device='cuda');cs[0]=seq;cs[1]=1
elif mode=='skew2':  # half full half 1 (extreme imbalance)
    cs=torch.full((batch,),1,dtype=torch.int32,device='cuda');cs[:batch//2]=seq
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
    for _ in range(20): run()
    e.record();torch.cuda.synchronize()
    res.append(s.elapsed_time(e)*1000/20)
print(f"case{c} {mode}: {statistics.median(res):.1f}us")
