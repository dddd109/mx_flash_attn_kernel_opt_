"""Quick best-of-3 per case for a given .so. Usage: python3 bestof.py <libpath>"""
import subprocess, sys

lib = sys.argv[1]
cases = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
results = {}
for c in cases:
    best = 1e9
    for i in range(3):
        r = subprocess.run(['python3','bench_cuda_event.py','opt',str(c),lib,'100'],
                          capture_output=True,text=True,cwd='/root/code',timeout=150)
        for line in r.stdout.split('\n'):
            if 'per_call=' in line:
                us = float(line.split('per_call=')[1].split('us')[0])
                best = min(best,us)
    results[c]=best
    print(f"case {c}: {best:.2f}us")
print(f"TOTAL: {sum(results.values()):.0f}us")
