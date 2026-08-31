#!/bin/bash
# Run all benchmark cases for flash and optimized kernels
# Usage: bash run_all_bench.sh <lib_path>
LIB=${1:-/root/code/my_optimized_kernel.so}
for c in 1 2 3 4 5 7 8 9 10 11 12 13; do
  echo "--- case $c ---"
  timeout 120 python3 bench_cuda_event.py flash $c 50
  timeout 120 python3 -c "
import sys
sys.argv = ['bench', 'opt', '$c', '50']
code = open('/root/code/bench_cuda_event.py').read()
code = code.replace('my_optimized_kernel.so', '$LIB')
exec(code)
"
done
