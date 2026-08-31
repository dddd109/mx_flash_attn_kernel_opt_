题目描述
你需要实现 FlashAttention paged KV cache decode 的前向算子（可用 TileLang / Triton / CUDA C++ 实现，见对应的接口约定）。

CUDA C++ 实现需要使用 mctlass 库以及 cute 库。

本题为 GQA（Grouped Query Attention） decode，测试范围覆盖 Qwen3-30B-A3B 风格的 h32/kv4/d128 decode，并加入 h32/kv8/d128 分组作为同类 GQA 变化。query head 数固定为 num_heads=32，KV head 数 num_heads_k 随测试点取 4 或 8。令 gqa_ratio = num_heads / num_heads_k，每个 query head h 使用 KV head h // gqa_ratio。每个 batch 只有 1 个 query token，K/V cache 按 page 存储。

seqlen_k 是 KV cache 的最大容量，不是每条序列的实际长度。每条序列实际有效的 KV 长度来自 cache_seqlens[b]，逐序列不同。正确实现必须按 cache_seqlens[b] 截断 page 遍历，只读取每行 block_table 前 ceil(cache_seqlens[b] / page_block_size) 个有效 page。

注意：block_table 每行后面的 padding 槽位不保证是 -1 或其他哨兵值，评测中它们可能也是合法 page id。正确 kernel 不应读取这些槽位；是否有效只由 cache_seqlens[b] 决定。

评测程序会调用你提交代码中的 run_kernel 函数。你需要根据 cache_seqlens 和 block_table 读取 paged KV cache，并将结果写入 output。

baseline 使用 FlashAttention 的 Python API（num_splits=0 让其自动选择 split-KV 切分数，反映真实的高性能 decode 标杆）：

out = flash_attn_with_kvcache(
    q, k_cache_paged, v_cache_paged, None, None,
    cache_seqlens=cache_seqlens,
    cache_batch_idx=None,
    block_table=block_table,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=False,
    alibi_slopes=None,
    num_splits=0,
)
output.copy_(out)
如何提交代码详见评测指南。

接口约定
评测机环境：镜像 maca torch 2.8.0+metax 3.7.1.5。

你必须在提交的 CUDA 源码中提供如下 C 符号，函数名、参数类型、顺序必须完全一致，并使用 extern "C" 防止 name mangling：

#include <stdint.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output,
    const int32_t* cache_seqlens,
    const int32_t* block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t num_blocks,
    int64_t causal
);
参数说明
q：decode query tensor，shape (batch_size, seqlen_q, num_heads, headdim)，连续 bf16
k_cache_paged：paged key cache，shape (num_blocks, page_block_size, num_heads_k, headdim)，连续 bf16
v_cache_paged：paged value cache，shape (num_blocks, page_block_size, num_heads_k, headdim)，连续 bf16
output：输出缓冲区，shape (batch_size, seqlen_q, num_heads, headdim)，连续 bf16
cache_seqlens：每个 batch 的实际有效 KV 长度，shape (batch_size)，连续 int32；逐序列不同，需据此截断 page 遍历
block_table：每个 batch 的 page 映射表，shape (batch_size, num_blocks / batch_size)，连续 int32；行宽可由 num_blocks / batch_size 得到。每行前 ceil(cache_seqlens[b] / page_block_size) 项为有效 page，其余为 padding，正确 kernel 不应读取；padding 槽位可能仍是合法 page id，不能当作哨兵判断
num_heads / num_heads_k：query / KV head 数（本题 num_heads=32，num_heads_k 取 4 或 8）；query head h 对应 KV head h // (num_heads / num_heads_k)
seqlen_k：KV cache 最大容量（非每序列实际长度）
seqlen_q：query 长度，评测中固定为 1
page_block_size：page size，评测中固定为 16
causal：是否启用 causal mask，评测中固定为 0
run_kernel 内部需要自行计算合适的 launch 配置并启动 CUDA kernel。为保证计时准确，不建议在 run_kernel 内部做 cudaDeviceSynchronize() 或显式同步。

输入格式
本题输入由评测程序在 GPU 上构造，并按接口约定中的顺序传入 run_kernel。

q/k_cache_paged/v_cache_paged/output 均为连续 torch.bfloat16 CUDA tensor，cache_seqlens/block_table 均为连续 torch.int32 CUDA tensor。

本题为 GQA decode：num_heads=32 个 query head 共享 num_heads_k 个 KV head；num_heads_k 随测试点取 4 或 8，对应 gqa_ratio=8 或 4。KV cache layout 固定为 flash_attn_with_kvcache 的 paged cache 布局：(num_blocks, page_block_size, num_heads_k, headdim)。

cache_seqlens[b] 给出第 b 条序列的实际有效 KV 长度，逐序列不同（模拟 continuous batching），范围 [1, seqlen_k]；seqlen_k 只是 KV cache 的最大容量。

block_table 的行宽可由 num_blocks / batch_size 得到。评测中该行宽等于 ceil(seqlen_k / page_block_size)，每行前 ceil(cache_seqlens[b] / page_block_size) 项为有效 page id，其余为 padding。padding 槽位仍可能保存合法 page id，不能用其数值判断是否有效；正确 kernel 应只根据 cache_seqlens[b] 决定读取范围。

输入分布用 torch.randn（标准正态）。

输出格式
输出写入 output，shape 为 (batch_size, 1, num_heads, headdim)，类型为 bfloat16。

样例
以 batch_size = 4、seqlen_k = 512（KV cache 最大容量）、page_block_size = 16、num_heads = 32、num_heads_k = 4 为例。

seqlen_k 只是容量；每条序列的实际有效长度来自 cache_seqlens，逐序列不同：

cache_seqlens = [512, 1, 200, 480]     # 逐序列变长，范围 [1, seqlen_k]
block_table.shape = (4, 32)            # 每行 ceil(512/16)=32 个 page 槽位
对第 b 条序列，只有前 ceil(cache_seqlens[b] / 16) 个 page 有效，其余为 padding，不应读取。例如序列 0 用满 32 个 page，序列 1 只用 1 个 page（ceil(1/16)=1），序列 2 用 ceil(200/16)=13 个。padding 槽位不保证为 -1，即使里面是合法 page id，也不属于该序列的有效 KV。

第 t 个 KV token（0 <= t < cache_seqlens[b]）位于 block_table[b, t // 16] 指向的物理 page 中，页内偏移为 t % 16。

GQA 分组：query head h 使用 KV head h // 8（gqa_ratio = 32/4 = 8），即 head 0..7 共享 KV head 0，head 8..15 共享 KV head 1，依此类推。

数据范围与提示
数据范围
本题为 GQA（Grouped Query Attention）paged KV cache decode。评测输入由固定 seed 合成生成，覆盖 Qwen3-30B-A3B 风格的 h32/kv4/d128 decode、h32/kv8/d128 分组变化、短序列 tail、大 batch 吞吐、长 KV cache 和小 batch 极长上下文。以下参数在所有 case 恒定：

num_heads = 32（query head 数）
headdim = 128
seqlen_q = 1（decode，每序列 1 个 query token）
page_block_size = 16
causal = 0
dtype：q/k_cache_paged/v_cache_paged/output 为 bfloat16，cache_seqlens/block_table 为 int32
num_heads_k 随测试点取 4 或 8，对应 gqa_ratio = num_heads / num_heads_k 为 8 或 4。

各 case 的 batch_size、seqlen_k（KV cache 最大容量）与 KV head 数：

case	batch	seqlen_k（容量）	实际 cache_seqlens 范围	num_heads_k	gqa_ratio	类型	warmup	iters
1	4	8	edge	3	100
2	4	2	1 ~ 2	8	4
3	16	17	1 ~ 17	4	8
4	64	1 ~ 64	8	4	perf	50
5	16	141	1 ~ 141	4	8
6	362	1 ~ 362	8	4
7	64	2048	1 ~ 2048	12
8	16	4096	1 ~ 4096	4	8	25
9	32	8	4	12
10	1	8192	4	8	25
11	16	12251	1 ~ 12251	12
12	8	32768	1 ~ 32768	8	4
13	1	58966	25
14	61519	4	8
warmup 表示不计入计时的预热调用次数，iters 表示预热后实际参与计时的调用次数。表中的实际长度范围表示生成器允许的取值范围；每个测试点至少有一条序列被固定为容量上限，batch > 1 时还至少有一条长度为 1 的序列。

关键语义：

seqlen_k 是 KV cache 的最大容量，不是每序列的实际长度。 每条序列实际有效的 KV 长度由 cache_seqlens[b] 给出，逐序列不同（模拟 continuous batching），范围 [1, seqlen_k]。正确实现必须按 cache_seqlens[b] 截断 page 遍历。
block_table 形状 (batch_size, num_blocks / batch_size)。 行宽可由 num_blocks / batch_size 得到；评测中该行宽等于 ceil(seqlen_k / page_block_size)。每行前 ceil(cache_seqlens[b] / page_block_size) 项为有效 page id，其余为 padding，正确 kernel 不应读取。
padding 槽位不保证为 -1 或其他哨兵值，评测中可能仍是合法 page id。不要根据 block_table 项的数值判断是否有效，唯一依据是 cache_seqlens[b]。
物理 page 池按每条序列的容量行宽分配，即 num_blocks = batch_size * ceil(seqlen_k / page_block_size)；没有额外 3x page 冗余，也没有 max(1024, ...) 的下限。
GQA 分组：query head h 使用 KV head h // gqa_ratio，其中 gqa_ratio = num_heads / num_heads_k。
输入用 torch.randn（标准正态），固定 seed，逐 case 可复现。
正确性要求
数学定义（对每个 batch b、每个 query head h）：

kv_head = h // gqa_ratio
tokens  = 由 block_table[b] 与 cache_seqlens[b] 解出的有效 KV token
logits  = (q[b,0,h,:] · k[tokens, kv_head, :]ᵀ) * sm_scale     # sm_scale = 1/sqrt(headdim)
attn    = softmax(logits)                                       # 单 query，全有效 KV 可见（causal=0）
out[b,0,h,:] = attn · v[tokens, kv_head, :]
只需将结果原地写回 output（shape (batch_size, 1, num_heads, headdim)，bfloat16）。
第 t 个 KV token 的物理位置：block_table[b, t // page_block_size] 指向的 page，页内偏移 t % page_block_size。
对每个 batch 只遍历 0 <= t < cache_seqlens[b]。seqlen_k 只是上界，不能直接当作所有序列的循环终点。
输出不得含 NaN / Inf。
评测口径：

tol   = atol + rtol * |ref|            # rtol = atol = 1.6e-2
diff  = |target - ref|
# 普通 case：至少 99% 元素在 tol 内，且无元素超过 8x tol
# edge case（case 1/2/3）：要求 100% 元素在 tol 内（同样受 8x tol outlier 约束）
matched_ratio = mean(diff <= tol) >= required_matched_ratio   # perf: 0.99, edge: 1.0
outlier       = any(diff > 8 * tol)  ->  直接判错
bf16 decode 在长 KV 上做 softmax 累加会有小幅逐元素误差，故用匹配率而非全元素 allclose；同时用 8x tol outlier 上限与 edge case 严判，防止整段算错蒙混过关。

提示
建议先保证语义正确（可先用等价的 PyTorch/SDPA 参考实现验证），再优化性能。
解码每个输出元素时需要的元数据通常是：blocks_per_batch = num_blocks / batch_size、valid_pages = ceil(cache_seqlens[b] / page_block_size)、kv_head = h // (num_heads / num_heads_k)。不要把 num_heads_k 写死成某一个值。
长 KV + 小 batch 场景下，split-KV（flash-decoding）是关键优化：单 query 要扫很长的 KV，不切分难以铺满所有 SM。baseline 使用 num_splits=0（自动切分），性能标杆反映真实高性能 decode。
GQA 的性能重点是让一组共享同一 KV head 的 query head 复用 KV cache 的读取（例如一个 block 处理一组 query head，复用 shared memory 里的 K/V）。
累加建议用 fp32，最后转 bf16 输出。
注意末页不满（cache_seqlens[b] % page_block_size != 0）与单 token（cache_seqlens[b] == 1）的边界。
