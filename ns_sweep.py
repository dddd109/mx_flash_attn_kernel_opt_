"""Sweep ns for a case under a given distribution via NS_OVERRIDE env."""
import torch, math, ctypes, os, sys, statistics
CASES={11:(16,12251,4),9:(32,4096,8)}
c=int(sys.argv[1]); mode=sys.argv[2]; ns=int(sys.argv[3])
lib=ctypes.CDLL('/tmp/ov_env.so'); lib.run_kernel.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int64]*9; lib.run_kernel.restype=None
os.environ['NS_OVERRIDE']=str(ns)
batch,seq,kv=CASES[c];headdim=128;nh=32;pbs=16;bpb=math.ceil(seq/pbs);nb=batch*bpb
torch.manual_seed(42)
if mode=='local':
    cs=torch.randint(1,seq+1,(batch,),dtype=torch.int32,device='cuda');cs[0]=seq;cs[1]=1
elif mode=='half':
    cs=torch.randint(seq//2,seq+1,(batch,),dtype=torch.int32,device='cuda');cs[0]=seq;cs[1]=seq//2
elif mode=='full':
    cs=torch.full((batch,),seq,dtype=torch.int32,device='cuda')
q=torch.randn(batch,1,nh,headdim,device='cuda',dtype=torch.bfloat16)
k=torch.randn(nb,pbs,kv,headdim,device='cuda',dtype=torch.bfloat16)
v=torch.randn(nb,pbs,kv,headdim,device='cuda',dtype=torch.bfloat16)
bt=torch.arange(nb,dtype=torch.int32,device='cuda').reshape(batch,bpb)
out=torch.zeros_like(q)
def run():
    lib.run_kernel(ctypes.cast(q.data_ptr(),ctypes.c_void_p),ctypes.cast(k.data_ptr(),ctypes.c_void_p),ctypes.cast(v.data_ptr(),ctypes.c_void_p),ctypes.cast(out.data_ptr(),ctypes.c_void_p),ctypes.cast(cs.data_ptr(),ctypes.c_void_p),ctypes.cast(bt.data_ptr(),ctypes.c_void_p),batch,seq,1,nh,kv,headdim,pbs,nb,0)
for _ in range(4): run()
torch.cuda.synchronize()
res=[]
for r in range(4):
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20): run()
    e.record();torch.cuda.synchronize()
    res.append(s.elapsed_time(e)*1000/20)
print(f"case{c} {mode} ns={ns}: {statistics.median(res):.1f}us")
