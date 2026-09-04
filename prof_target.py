"""Profile target: run one case many times for mcProfiler.
Usage: python3 prof_target.py <lib.so> <case>
"""
import sys, torch, math, ctypes
from flash_attn import flash_attn_with_kvcache

CASES = {1:(1,1,4), 2:(4,2,8), 3:(16,17,4), 4:(64,64,8), 5:(16,141,4), 6:(16,362,8),
         7:(64,2048,8), 8:(16,4096,4), 9:(32,4096,8), 10:(1,8192,4), 11:(16,12251,4),
         12:(8,32768,8), 13:(1,58966,8), 14:(1,61519,4)}

lib = ctypes.CDLL(sys.argv[1])
lib.run_kernel.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int64]*9
lib.run_kernel.restype = None
c = int(sys.argv[2])
batch, seq, kv = CASES[c]
headdim=128; nh=32; pbs=16
bpb=math.ceil(seq/pbs); nb=batch*bpb
torch.manual_seed(42)
if c==1: cs=torch.full((batch,),1,dtype=torch.int32,device='cuda')
else:
    cs=torch.randint(1,seq+1,(batch,),dtype=torch.int32,device='cuda')
    cs[0]=seq
    if batch>1: cs[1]=1
q=torch.randn(batch,1,nh,headdim,device='cuda',dtype=torch.bfloat16)
k=torch.randn(nb,pbs,kv,headdim,device='cuda',dtype=torch.bfloat16)
v=torch.randn(nb,pbs,kv,headdim,device='cuda',dtype=torch.bfloat16)
bt=torch.arange(nb,dtype=torch.int32,device='cuda').reshape(batch,bpb)
out=torch.zeros_like(q)
for _ in range(5):
    lib.run_kernel(ctypes.cast(q.data_ptr(),ctypes.c_void_p),ctypes.cast(k.data_ptr(),ctypes.c_void_p),
     ctypes.cast(v.data_ptr(),ctypes.c_void_p),ctypes.cast(out.data_ptr(),ctypes.c_void_p),
     ctypes.cast(cs.data_ptr(),ctypes.c_void_p),ctypes.cast(bt.data_ptr(),ctypes.c_void_p),
     batch,seq,1,nh,kv,headdim,pbs,nb,0)
torch.cuda.synchronize()
# many iterations for profiler to sample
for _ in range(200):
    lib.run_kernel(ctypes.cast(q.data_ptr(),ctypes.c_void_p),ctypes.cast(k.data_ptr(),ctypes.c_void_p),
     ctypes.cast(v.data_ptr(),ctypes.c_void_p),ctypes.cast(out.data_ptr(),ctypes.c_void_p),
     ctypes.cast(cs.data_ptr(),ctypes.c_void_p),ctypes.cast(bt.data_ptr(),ctypes.c_void_p),
     batch,seq,1,nh,kv,headdim,pbs,nb,0)
torch.cuda.synchronize()
