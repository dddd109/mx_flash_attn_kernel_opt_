"""Benchmark taking min of 3 runs (best-case, stable for launch-floor cases).
Usage: python3 bench_min.py <flash|opt> <case> <libpath> [reps]
"""
import sys, torch, math, ctypes, subprocess

which = sys.argv[1]
case = sys.argv[2]
libpath = sys.argv[3] if len(sys.argv) > 3 else '/tmp/agent-v7/your_kernel.so'
reps = sys.argv[4] if len(sys.argv) > 4 else '100'

cmd = ['python3', 'bench_cuda_event.py', which, case, reps]
if which != 'flash':
    cmd = ['python3', 'bench_cuda_event.py', which, case, libpath, reps]

best = 1e9
for i in range(3):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd='/root/code')
    for line in r.stdout.strip().split('\n'):
        if 'per_call=' in line:
            us = float(line.split('per_call=')[1].split('us')[0])
            best = min(best, us)
print(f"case={case} {which} best_of_3={best:.2f}us")
