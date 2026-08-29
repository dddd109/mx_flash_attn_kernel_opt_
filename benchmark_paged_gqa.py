"""
Benchmark script for paged GQA decode kernel on MetaX C500 GPU
"""

import itertools
import os
import random
from datetime import datetime
import numpy as np
import torch
import triton
import triton.language as tl

# Parameters matching the competition
PAGE_SIZE = 16
NUM_QO_HEADS = 32
HEAD_DIM = 128
DTYPE = torch.bfloat16


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_csv_path(prefix):
    return f"{prefix}_{get_timestamp()}.csv"


def generate_random_seqlens(batch_size, min_len=1024, max_len=16384):
    return [random.randint(min_len, max_len) for _ in range(batch_size)]


def compute_reps(batch_size, seq_len, head_dim, base_reps=100):
    workload = batch_size * seq_len * head_dim
    if workload < 1e5:
        return base_reps
    elif workload < 1e6:
        return base_reps // 2
    elif workload < 1e7:
        return base_reps // 4
    elif workload < 1e8:
        return base_reps // 8
    elif workload < 1e9:
        return base_reps // 16
    else:
        return base_reps // 32


def run_with_profiler(fn, warmup=10, reps=100, print_result=False, target_kernels=None):
    """Run function with torch.profiler and return sum of specific kernel times in ms"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()

    if print_result:
        print(prof.key_averages().table(sort_by="device_time", row_limit=20))

    if target_kernels is None:
        target_kernels = []

    kernel_times_us = 0.0
    for evt in prof.key_averages():
        if any(k in evt.key for k in target_kernels):
            kernel_times_us += evt.device_time

    ms = kernel_times_us / 1e3
    return ms


def run_benchmark_triton():
    """Benchmark the Triton version of paged GQA"""
    import submission_c500_regions as regions

    records = []
    batch_sizes = [1, 2, 4, 8]
    seq_lens_kv = [1024, 2048, 4096, 8192, 16384]
    num_kv_heads_options = [4, 8]

    print("[Paged GQA Triton] Starting benchmark")
    test_cases = list(itertools.product(batch_sizes, seq_lens_kv, num_kv_heads_options))
    total_cases = len(test_cases)

    for idx, (bs, seq_len_kv, num_kv_heads) in enumerate(test_cases, 1):
        gqa_ratio = NUM_QO_HEADS // num_kv_heads

        # Setup paged KV cache
        seq_lens = [seq_len_kv] * bs
        cache_seqlens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        num_blocks = sum((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens)
        blocks_per_batch = num_blocks // bs

        # Create block table (simple consecutive mapping)
        block_table = torch.arange(bs * blocks_per_batch, dtype=torch.int32, device="cuda")

        # Create tensors
        q = torch.rand(bs, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
        k_cache = torch.randn(num_blocks, PAGE_SIZE, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
        v_cache = torch.randn(num_blocks, PAGE_SIZE, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
        output = torch.zeros(bs, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")

        # Run benchmark
        def run_kernel():
            regions.run_kernel(
                q, k_cache, v_cache, output,
                cache_seqlens, block_table,
                bs, seq_len_kv, 1,
                NUM_QO_HEADS, num_kv_heads, HEAD_DIM,
                PAGE_SIZE, num_blocks, 0
            )

        reps = compute_reps(bs, seq_len_kv, HEAD_DIM, base_reps=100)
        try:
            ms = run_with_profiler(run_kernel, reps=reps, target_kernels=["paged_gqa"])
        except Exception as e:
            print(f"  [{idx}/{total_cases}] bs={bs}, seq={seq_len_kv}, nkv={num_kv_heads}: ERROR - {e}")
            continue

        io = q.numel() * q.element_size() + k_cache.numel() * k_cache.element_size() + v_cache.numel() * v_cache.element_size()
        flops = 2 * bs * seq_len_kv * NUM_QO_HEADS * num_kv_heads * HEAD_DIM
        bw = io / ms / 1e6
        tflops = flops / ms / 1e9

        records.append({
            "api": "paged_gqa_triton",
            "batch_size": bs,
            "seq_len_q": 1,
            "seq_len_kv": seq_len_kv,
            "num_qo_heads": NUM_QO_HEADS,
            "num_kv_heads": num_kv_heads,
            "head_dim": HEAD_DIM,
            "gqa_ratio": gqa_ratio,
            "time_ms": ms,
            "bandwidth_GB_s": bw,
            "tflops": tflops,
        })
        print(f"  [{idx}/{total_cases}] bs={bs}, kv_len={seq_len_kv}, nkv={num_kv_heads}: {ms:.3f}ms, {bw:.2f} GB/s, {tflops:.2f} TFLOPs")

    return records


if __name__ == "__main__":
    np.random.seed(42)
    torch.random.manual_seed(42)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        exit(0)

    records = run_benchmark_triton()
    df = pd.DataFrame(records)
    csv_path = get_csv_path("paged_gqa_triton")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")