#!/bin/bash
# Profile script for paged GQA kernel on MetaX C500 GPU
# Usage: bash profile_paged_gqa.sh <batch_size> <seq_len_kv> <num_kv_heads>

set -e

BATCH_SIZE=${1:-4}
SEQ_LEN=${2:-8192}
NUM_KV_HEADS=${3:-8}
GQA_RATIO=4  # 32 Q heads / 8 KV heads

echo "=============================================="
echo "Paged GQA Kernel Profiling"
echo "=============================================="
echo "Batch size: $BATCH_SIZE"
echo "Seq len KV: $SEQ_LEN"
echo "Num KV heads: $NUM_KV_HEADS"
echo "GQA ratio: $GQA_RATIO"
echo "=============================================="

# Create a temporary Python script to profile
cat > /tmp/profile_run.py << 'EOF'
import torch
import sys
sys.path.insert(0, '/root/code')

import submission_c500_regions as regions

# Test parameters
batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
seq_len_kv = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
num_kv_heads = int(sys.argv[3]) if len(sys.argv) > 3 else 8
num_qo_heads = 32
head_dim = 128
page_size = 16

print(f"Testing: batch={batch_size}, seq={seq_len_kv}, nkv={num_kv_heads}")

# Setup
seq_lens = [seq_len_kv] * batch_size
cache_seqlens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
num_blocks = sum((s + page_size - 1) // page_size for s in seq_lens)
blocks_per_batch = num_blocks // batch_size
block_table = torch.arange(batch_size * blocks_per_batch, dtype=torch.int32, device="cuda")

# Create tensors
q = torch.rand(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda")
k_cache = torch.randn(num_blocks, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
v_cache = torch.randn(num_blocks, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
output = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda")

# Warmup
print("Warming up...")
for _ in range(10):
    regions.run_kernel(
        q, k_cache, v_cache, output,
        cache_seqlens, block_table,
        batch_size, seq_len_kv, 1,
        num_qo_heads, num_kv_heads, head_dim,
        page_size, num_blocks, 0
    )
torch.cuda.synchronize()

# Profile with torch profiler
print("Profiling...")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(100):
        regions.run_kernel(
            q, k_cache, v_cache, output,
            cache_seqlens, block_table,
            batch_size, seq_len_kv, 1,
            num_qo_heads, num_kv_heads, head_dim,
            page_size, num_blocks, 0
        )
torch.cuda.synchronize()

print("\nTop kernels by CUDA time:")
print(prof.key_averages().table(sort_by="cuda_time", row_limit=20))

# Save to file
prof.export_chrome_trace("/tmp/trace.json")
print("\nTrace saved to /tmp/trace.json")
EOF

# Run with mcProfiler if available, otherwise just run Python
if command -v mcProfiler &> /dev/null; then
    echo "Using mcProfiler..."
    mcProfiler perf_exec \
        --cmdline "python3 /tmp/profile_run.py $BATCH_SIZE $SEQ_LEN $NUM_KV_HEADS" \
        --kernelname "paged_gqa" \
        --casename "decode_bs${BATCH_SIZE}_seq${SEQ_LEN}_nkv${NUM_KV_HEADS}" \
        --cwd /root/code \
        --metrics "sm_efficiency,achieved_occupancy,dram_utilization,l2_utilization,ipc" \
        --per-kernel --single-pass
else
    echo "mcProfiler not available, running with Python profiler..."
    python3 /tmp/profile_run.py $BATCH_SIZE $SEQ_LEN $NUM_KV_HEADS
fi

echo "=============================================="
echo "Profile complete"
echo "=============================================="