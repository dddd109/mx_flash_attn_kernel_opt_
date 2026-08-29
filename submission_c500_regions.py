import torch
import triton
import triton.language as tl


_REGION_SERIAL_REGULAR = 0
_REGION_SERIAL_LONG = 1
_REGION_SERIAL_PACKED = 2
_REGION_BATCHED_REGULAR = 3
_REGION_BATCHED_DENSE = 4
_REGION_BATCHED_LONG = 5

_SHORT_BLOCK_LIMIT = 16
_MEDIUM_BLOCK_LIMIT = 256
_DENSE_BATCH_MIN_SEQLEN = 2048
_LONG_BATCH_MIN_SEQLEN = 4096
_LONG_SERIAL_MIN_SEQLEN = 12000
_DENSE_RESIDENT_GROUPS = 512
_PACKED_MIN_GQA_RATIO = 8

_BASE_TARGET_PROGRAMS = 256
_BATCHED_LONG_TARGET_PROGRAMS = 512
_BATCHED_DENSE_TARGET_PROGRAMS = 1024
_SERIAL_LONG_TARGET_PROGRAMS = 2048

_NATURAL_MIN_TOKENS_PER_SPLIT = 256
_SERIAL_LONG_MIN_TOKENS_PER_SPLIT = 128
_NATURAL_MAX_SPLITS = 256

_PACKED_TARGET_PROGRAMS = 256
_PACKED_MIN_TOKENS_PER_SPLIT = 128
_PACKED_MAX_SPLITS = 128

_D_TILE = 64
_WORKSPACES = {}


@triton.jit
def _paged_gqa_masked_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    BLOCKS_PER_BATCH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_TILE: tl.constexpr,
    STORE_PARTIAL: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % NUM_SPLITS
    group_id = pid // NUM_SPLITS
    kv_head = group_id % NUM_KV_HEADS
    batch_id = group_id // NUM_KV_HEADS

    seq_len = tl.load(cache_seqlens_ptr + batch_id).to(tl.int32)

    chunk = (seq_len + NUM_SPLITS - 1) // NUM_SPLITS
    chunk = ((chunk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    split_start = tl.minimum(split_id * chunk, seq_len)
    split_end = tl.minimum(split_start + chunk, seq_len)

    offs_m = tl.arange(0, 16)
    offs_d0 = tl.arange(0, D_TILE)
    offs_d1 = D_TILE + tl.arange(0, D_TILE)
    query_heads = kv_head * GQA_RATIO + offs_m
    active_head = offs_m < GQA_RATIO

    q_base = (batch_id * NUM_HEADS + query_heads[:, None]) * HEAD_DIM
    q0 = tl.load(
        q_ptr + q_base + offs_d0[None, :],
        mask=active_head[:, None],
        other=0.0,
    )
    q1 = tl.load(
        q_ptr + q_base + offs_d1[None, :],
        mask=active_head[:, None],
        other=0.0,
    )

    m_i = tl.full((16,), -float("inf"), tl.float32)
    l_i = tl.zeros((16,), tl.float32)
    acc0 = tl.zeros((16, D_TILE), tl.float32)
    acc1 = tl.zeros((16, D_TILE), tl.float32)

    page_stride = PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM
    token_stride = NUM_KV_HEADS * HEAD_DIM

    for start_n in tl.range(split_start, split_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_token = offs_n < split_end

        logical_page = offs_n // PAGE_SIZE
        page_offset = offs_n % PAGE_SIZE

        physical_page = tl.load(
            block_table_ptr + batch_id * BLOCKS_PER_BATCH + logical_page,
            mask=valid_token,
            other=0,
        ).to(tl.int32)

        kv_token_base = (
            physical_page * page_stride
            + page_offset * token_stride
            + kv_head * HEAD_DIM
        )
        kv_token_base = tl.multiple_of(kv_token_base, HEAD_DIM)

        k0_offsets = kv_token_base[:, None] + offs_d0[None, :]
        k0 = tl.load(
            k_ptr + k0_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        qk = tl.dot(q0, tl.trans(k0))

        k1_offsets = kv_token_base[:, None] + offs_d1[None, :]
        k1 = tl.load(
            k_ptr + k1_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        qk = tl.dot(q1, tl.trans(k1), qk) * SM_SCALE
        qk = tl.where(valid_token[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2((m_i - m_ij) * 1.4426950408889634)
        p = tl.math.exp2((qk - m_ij[:, None]) * 1.4426950408889634)
        p_bf16 = p.to(tl.bfloat16)

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]

        v0 = tl.load(
            v_ptr + k0_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        acc0 = tl.dot(p_bf16, v0, acc0)

        v1 = tl.load(
            v_ptr + k1_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        acc1 = tl.dot(p_bf16, v1, acc1)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    if STORE_PARTIAL:
        stat_offsets = (
            (batch_id * NUM_HEADS + query_heads) * NUM_SPLITS + split_id
        )
        tl.store(partial_m_ptr + stat_offsets, m_i, mask=active_head)
        tl.store(partial_l_ptr + stat_offsets, l_i, mask=active_head)

        acc_base = stat_offsets[:, None] * HEAD_DIM
        tl.store(
            partial_acc_ptr + acc_base + offs_d0[None, :],
            acc0,
            mask=active_head[:, None],
        )
        tl.store(
            partial_acc_ptr + acc_base + offs_d1[None, :],
            acc1,
            mask=active_head[:, None],
        )
    else:
        inv_l = 1.0 / l_i
        out_base = (batch_id * NUM_HEADS + query_heads[:, None]) * HEAD_DIM
        tl.store(
            out_ptr + out_base + offs_d0[None, :],
            acc0 * inv_l[:, None],
            mask=active_head[:, None],
        )
        tl.store(
            out_ptr + out_base + offs_d1[None, :],
            acc1 * inv_l[:, None],
            mask=active_head[:, None],
        )


@triton.jit
def _paged_gqa_locality_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    BLOCKS_PER_BATCH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_TILE: tl.constexpr,
    STORE_PARTIAL: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    kv_head = pid % NUM_KV_HEADS
    batch_split = pid // NUM_KV_HEADS
    split_id = batch_split % NUM_SPLITS
    batch_id = batch_split // NUM_SPLITS

    seq_len = tl.load(cache_seqlens_ptr + batch_id).to(tl.int32)

    chunk = (seq_len + NUM_SPLITS - 1) // NUM_SPLITS
    chunk = ((chunk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    split_start = tl.minimum(split_id * chunk, seq_len)
    split_end = tl.minimum(split_start + chunk, seq_len)

    offs_m = tl.arange(0, 16)
    offs_d0 = tl.arange(0, D_TILE)
    offs_d1 = D_TILE + tl.arange(0, D_TILE)
    query_heads = kv_head * GQA_RATIO + offs_m
    active_head = offs_m < GQA_RATIO

    q_base = (batch_id * NUM_HEADS + query_heads[:, None]) * HEAD_DIM
    q0 = tl.load(
        q_ptr + q_base + offs_d0[None, :],
        mask=active_head[:, None],
        other=0.0,
    )
    q1 = tl.load(
        q_ptr + q_base + offs_d1[None, :],
        mask=active_head[:, None],
        other=0.0,
    )

    m_i = tl.full((16,), -float("inf"), tl.float32)
    l_i = tl.zeros((16,), tl.float32)
    acc0 = tl.zeros((16, D_TILE), tl.float32)
    acc1 = tl.zeros((16, D_TILE), tl.float32)

    page_stride = PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM
    token_stride = NUM_KV_HEADS * HEAD_DIM

    for start_n in tl.range(split_start, split_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_token = offs_n < split_end

        logical_page = offs_n // PAGE_SIZE
        page_offset = offs_n % PAGE_SIZE

        physical_page = tl.load(
            block_table_ptr + batch_id * BLOCKS_PER_BATCH + logical_page,
            mask=valid_token,
            other=0,
        ).to(tl.int32)

        kv_token_base = (
            physical_page * page_stride
            + page_offset * token_stride
            + kv_head * HEAD_DIM
        )
        kv_token_base = tl.multiple_of(kv_token_base, HEAD_DIM)

        k0_offsets = kv_token_base[:, None] + offs_d0[None, :]
        k0 = tl.load(
            k_ptr + k0_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        qk = tl.dot(q0, tl.trans(k0))

        k1_offsets = kv_token_base[:, None] + offs_d1[None, :]
        k1 = tl.load(
            k_ptr + k1_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        qk = tl.dot(q1, tl.trans(k1), qk) * SM_SCALE
        qk = tl.where(valid_token[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2((m_i - m_ij) * 1.4426950408889634)
        p = tl.math.exp2((qk - m_ij[:, None]) * 1.4426950408889634)
        p_bf16 = p.to(tl.bfloat16)

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]

        v0 = tl.load(
            v_ptr + k0_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        acc0 = tl.dot(p_bf16, v0, acc0)

        v1 = tl.load(
            v_ptr + k1_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        acc1 = tl.dot(p_bf16, v1, acc1)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    if STORE_PARTIAL:
        stat_offsets = (
            (batch_id * NUM_HEADS + query_heads) * NUM_SPLITS + split_id
        )
        tl.store(partial_m_ptr + stat_offsets, m_i, mask=active_head)
        tl.store(partial_l_ptr + stat_offsets, l_i, mask=active_head)

        acc_base = stat_offsets[:, None] * HEAD_DIM
        tl.store(
            partial_acc_ptr + acc_base + offs_d0[None, :],
            acc0,
            mask=active_head[:, None],
        )
        tl.store(
            partial_acc_ptr + acc_base + offs_d1[None, :],
            acc1,
            mask=active_head[:, None],
        )
    else:
        inv_l = 1.0 / l_i
        out_base = (batch_id * NUM_HEADS + query_heads[:, None]) * HEAD_DIM
        tl.store(
            out_ptr + out_base + offs_d0[None, :],
            acc0 * inv_l[:, None],
            mask=active_head[:, None],
        )
        tl.store(
            out_ptr + out_base + offs_d1[None, :],
            acc1 * inv_l[:, None],
            mask=active_head[:, None],
        )


@triton.jit
def _paged_gqa_fulltile_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    BLOCKS_PER_BATCH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_TILE: tl.constexpr,
    STORE_PARTIAL: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % NUM_SPLITS
    group_id = pid // NUM_SPLITS
    kv_head = group_id % NUM_KV_HEADS
    batch_id = group_id // NUM_KV_HEADS

    seq_len = tl.load(cache_seqlens_ptr + batch_id).to(tl.int32)
    chunk = (seq_len + NUM_SPLITS - 1) // NUM_SPLITS
    chunk = ((chunk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    split_start = tl.minimum(split_id * chunk, seq_len)
    split_end = tl.minimum(split_start + chunk, seq_len)

    offs_m = tl.arange(0, 16)
    offs_d0 = tl.arange(0, D_TILE)
    offs_d1 = D_TILE + tl.arange(0, D_TILE)
    query_heads = kv_head * GQA_RATIO + offs_m
    active_head = offs_m < GQA_RATIO

    q_base = (batch_id * NUM_HEADS + query_heads[:, None]) * HEAD_DIM
    q0 = tl.load(
        q_ptr + q_base + offs_d0[None, :],
        mask=active_head[:, None],
        other=0.0,
    )
    q1 = tl.load(
        q_ptr + q_base + offs_d1[None, :],
        mask=active_head[:, None],
        other=0.0,
    )

    m_i = tl.full((16,), -float("inf"), tl.float32)
    l_i = tl.zeros((16,), tl.float32)
    acc0 = tl.zeros((16, D_TILE), tl.float32)
    acc1 = tl.zeros((16, D_TILE), tl.float32)

    page_stride = PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM
    token_stride = NUM_KV_HEADS * HEAD_DIM
    split_tokens = split_end - split_start
    full_end = split_end - (split_tokens % BLOCK_N)

    for start_n in tl.range(split_start, full_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        logical_page = offs_n // PAGE_SIZE
        page_offset = offs_n % PAGE_SIZE
        physical_page = tl.load(
            block_table_ptr + batch_id * BLOCKS_PER_BATCH + logical_page,
        ).to(tl.int32)
        kv_token_base = (
            physical_page * page_stride
            + page_offset * token_stride
            + kv_head * HEAD_DIM
        )
        kv_token_base = tl.multiple_of(kv_token_base, HEAD_DIM)

        k0_offsets = kv_token_base[:, None] + offs_d0[None, :]
        k0 = tl.load(k_ptr + k0_offsets)
        qk = tl.dot(q0, tl.trans(k0))

        k1_offsets = kv_token_base[:, None] + offs_d1[None, :]
        k1 = tl.load(k_ptr + k1_offsets)
        qk = tl.dot(q1, tl.trans(k1), qk) * SM_SCALE

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2((m_i - m_ij) * 1.4426950408889634)
        p = tl.math.exp2((qk - m_ij[:, None]) * 1.4426950408889634)
        p_bf16 = p.to(tl.bfloat16)

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        v0 = tl.load(v_ptr + k0_offsets)
        acc0 = tl.dot(p_bf16, v0, acc0)
        v1 = tl.load(v_ptr + k1_offsets)
        acc1 = tl.dot(p_bf16, v1, acc1)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    if full_end < split_end:
        start_n = full_end
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_token = offs_n < split_end
        logical_page = offs_n // PAGE_SIZE
        page_offset = offs_n % PAGE_SIZE
        physical_page = tl.load(
            block_table_ptr + batch_id * BLOCKS_PER_BATCH + logical_page,
            mask=valid_token,
            other=0,
        ).to(tl.int32)
        kv_token_base = (
            physical_page * page_stride
            + page_offset * token_stride
            + kv_head * HEAD_DIM
        )
        kv_token_base = tl.multiple_of(kv_token_base, HEAD_DIM)

        k0_offsets = kv_token_base[:, None] + offs_d0[None, :]
        k0 = tl.load(
            k_ptr + k0_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        qk = tl.dot(q0, tl.trans(k0))
        k1_offsets = kv_token_base[:, None] + offs_d1[None, :]
        k1 = tl.load(
            k_ptr + k1_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        qk = tl.dot(q1, tl.trans(k1), qk) * SM_SCALE
        qk = tl.where(valid_token[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2((m_i - m_ij) * 1.4426950408889634)
        p = tl.math.exp2((qk - m_ij[:, None]) * 1.4426950408889634)
        p_bf16 = p.to(tl.bfloat16)

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        v0 = tl.load(
            v_ptr + k0_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        acc0 = tl.dot(p_bf16, v0, acc0)
        v1 = tl.load(
            v_ptr + k1_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        acc1 = tl.dot(p_bf16, v1, acc1)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    if STORE_PARTIAL:
        stat_offsets = (
            (batch_id * NUM_HEADS + query_heads) * NUM_SPLITS + split_id
        )
        tl.store(partial_m_ptr + stat_offsets, m_i, mask=active_head)
        tl.store(partial_l_ptr + stat_offsets, l_i, mask=active_head)
        acc_base = stat_offsets[:, None] * HEAD_DIM
        tl.store(
            partial_acc_ptr + acc_base + offs_d0[None, :],
            acc0,
            mask=active_head[:, None],
        )
        tl.store(
            partial_acc_ptr + acc_base + offs_d1[None, :],
            acc1,
            mask=active_head[:, None],
        )
    else:
        inv_l = 1.0 / l_i
        out_base = (batch_id * NUM_HEADS + query_heads[:, None]) * HEAD_DIM
        tl.store(
            out_ptr + out_base + offs_d0[None, :],
            acc0 * inv_l[:, None],
            mask=active_head[:, None],
        )
        tl.store(
            out_ptr + out_base + offs_d1[None, :],
            acc1 * inv_l[:, None],
            mask=active_head[:, None],
        )


@triton.jit
def _paged_gqa_packed_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    BLOCKS_PER_BATCH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    PACK_KV: tl.constexpr,
    PACKED_GROUPS: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    D_TILE: tl.constexpr,
    STORE_PARTIAL: tl.constexpr,
    QK_SCALE: tl.constexpr,
):
    packed_group = tl.program_id(0)
    split_id = tl.program_id(1)
    local_packed_group = packed_group % PACKED_GROUPS
    batch_id = packed_group // PACKED_GROUPS
    kv_head_base = local_packed_group * PACK_KV

    seq_len = tl.load(cache_seqlens_ptr + batch_id).to(tl.int32)
    chunk = (seq_len + NUM_SPLITS - 1) // NUM_SPLITS
    chunk = ((chunk + TOKEN_TILE - 1) // TOKEN_TILE) * TOKEN_TILE
    split_start = tl.minimum(split_id * chunk, seq_len)
    split_end = tl.minimum(split_start + chunk, seq_len)

    offs_m = tl.arange(0, 16)
    offs_d0 = tl.arange(0, D_TILE)
    offs_d1 = D_TILE + tl.arange(0, D_TILE)
    row_kv = offs_m // GQA_RATIO
    query_heads = kv_head_base * GQA_RATIO + offs_m

    q_base = (batch_id * NUM_HEADS + query_heads[:, None]) * HEAD_DIM
    q0 = tl.load(q_ptr + q_base + offs_d0[None, :])
    q1 = tl.load(q_ptr + q_base + offs_d1[None, :])
    q0 = (q0 * QK_SCALE).to(tl.bfloat16)
    q1 = (q1 * QK_SCALE).to(tl.bfloat16)

    m_i = tl.full((16,), -float("inf"), tl.float32)
    l_i = tl.zeros((16,), tl.float32)
    acc0 = tl.zeros((16, D_TILE), tl.float32)
    acc1 = tl.zeros((16, D_TILE), tl.float32)

    offs_n = tl.arange(0, 64)
    col_kv = offs_n // TOKEN_TILE
    token_lane = offs_n % TOKEN_TILE
    same_kv = row_kv[:, None] == col_kv[None, :]

    page_stride = PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM
    token_stride = NUM_KV_HEADS * HEAD_DIM

    for start_n in tl.range(split_start, split_end, TOKEN_TILE):
        page_index0 = start_n // PAGE_SIZE
        page0 = tl.load(
            block_table_ptr + batch_id * BLOCKS_PER_BATCH + page_index0,
            mask=start_n < split_end,
            other=0,
        ).to(tl.int32)

        if TOKEN_TILE == 16:
            physical_page = page0 + tl.zeros((64,), tl.int32)
            page_offset = token_lane
        else:
            page1 = tl.load(
                block_table_ptr + batch_id * BLOCKS_PER_BATCH + page_index0 + 1,
                mask=start_n + PAGE_SIZE < split_end,
                other=0,
            ).to(tl.int32)
            physical_page = tl.where(token_lane < PAGE_SIZE, page0, page1)
            page_offset = token_lane % PAGE_SIZE

        token_pos = start_n + token_lane
        valid_token = token_pos < split_end
        kv_head = kv_head_base + col_kv
        kv_token_base = (
            physical_page * page_stride
            + page_offset * token_stride
            + kv_head * HEAD_DIM
        )
        kv_token_base = tl.multiple_of(kv_token_base, HEAD_DIM)

        k0_offsets = kv_token_base[:, None] + offs_d0[None, :]
        k0 = tl.load(
            k_ptr + k0_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        qk = tl.dot(q0, tl.trans(k0))

        k1_offsets = kv_token_base[:, None] + offs_d1[None, :]
        k1 = tl.load(
            k_ptr + k1_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        qk = tl.dot(q1, tl.trans(k1), qk)
        qk = tl.where(same_kv & valid_token[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])
        p_bf16 = p.to(tl.bfloat16)

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]

        v0 = tl.load(
            v_ptr + k0_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        acc0 = tl.dot(p_bf16, v0, acc0)

        v1 = tl.load(
            v_ptr + k1_offsets,
            mask=valid_token[:, None],
            other=0.0,
        )
        acc1 = tl.dot(p_bf16, v1, acc1)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    stat_offsets = (
        (batch_id * NUM_HEADS + query_heads) * NUM_SPLITS + split_id
    )

    if STORE_PARTIAL:
        tl.store(partial_m_ptr + stat_offsets, m_i)
        tl.store(partial_l_ptr + stat_offsets, l_i)

        acc_base = stat_offsets[:, None] * HEAD_DIM
        tl.store(
            partial_acc_ptr + acc_base + offs_d0[None, :],
            acc0,
        )
        tl.store(
            partial_acc_ptr + acc_base + offs_d1[None, :],
            acc1,
        )
    else:
        inv_l = 1.0 / l_i
        out_base = (batch_id * NUM_HEADS + query_heads[:, None]) * HEAD_DIM
        tl.store(
            out_ptr + out_base + offs_d0[None, :],
            acc0 * inv_l[:, None],
        )
        tl.store(
            out_ptr + out_base + offs_d1[None, :],
            acc1 * inv_l[:, None],
        )


@triton.jit
def _reduce_splits_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    out_ptr,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    d_block = tl.program_id(1)

    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

    stat_offsets = bh * NUM_SPLITS + offs_s
    part_m = tl.load(partial_m_ptr + stat_offsets)
    global_m = tl.max(part_m, axis=0)
    alpha = tl.math.exp2((part_m - global_m) * 1.4426950408889634)

    part_l = tl.load(partial_l_ptr + stat_offsets)
    global_l = tl.sum(part_l * alpha, axis=0)

    acc_offsets = stat_offsets[:, None] * HEAD_DIM + offs_d[None, :]
    part_acc = tl.load(partial_acc_ptr + acc_offsets)
    out = tl.sum(part_acc * alpha[:, None], axis=0) / global_l

    tl.store(out_ptr + bh * HEAD_DIM + offs_d, out)


@triton.jit
def _reduce_splits_log2_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    out_ptr,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    d_block = tl.program_id(1)

    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    stat_offsets = bh * NUM_SPLITS + offs_s

    part_m = tl.load(partial_m_ptr + stat_offsets)
    global_m = tl.max(part_m, axis=0)
    alpha = tl.math.exp2(part_m - global_m)

    part_l = tl.load(partial_l_ptr + stat_offsets)
    global_l = tl.sum(part_l * alpha, axis=0)

    acc_offsets = stat_offsets[:, None] * HEAD_DIM + offs_d[None, :]
    part_acc = tl.load(partial_acc_ptr + acc_offsets)
    out = tl.sum(part_acc * alpha[:, None], axis=0) / global_l
    tl.store(out_ptr + bh * HEAD_DIM + offs_d, out)


def _select_region(batch_size, gqa_ratio, resident_groups, seqlen_k):
    if batch_size == 1:
        if seqlen_k >= _LONG_SERIAL_MIN_SEQLEN:
            if gqa_ratio >= _PACKED_MIN_GQA_RATIO:
                return _REGION_SERIAL_PACKED
            return _REGION_SERIAL_LONG
        return _REGION_SERIAL_REGULAR

    if seqlen_k >= _LONG_BATCH_MIN_SEQLEN:
        return _REGION_BATCHED_LONG
    if (
        seqlen_k >= _DENSE_BATCH_MIN_SEQLEN
        and resident_groups >= _DENSE_RESIDENT_GROUPS
    ):
        return _REGION_BATCHED_DENSE
    return _REGION_BATCHED_REGULAR


def _ceil_pow2_capped(value, maximum):
    if value <= 1:
        result = 1
    elif value <= 2:
        result = 2
    elif value <= 4:
        result = 4
    elif value <= 8:
        result = 8
    elif value <= 16:
        result = 16
    elif value <= 32:
        result = 32
    elif value <= 64:
        result = 64
    elif value <= 128:
        result = 128
    else:
        result = 256
    if result > maximum:
        return maximum
    return result


def _floor_pow2_capped(value, maximum):
    if value >= 256:
        result = 256
    elif value >= 128:
        result = 128
    elif value >= 64:
        result = 64
    elif value >= 32:
        result = 32
    elif value >= 16:
        result = 16
    elif value >= 8:
        result = 8
    elif value >= 4:
        result = 4
    elif value >= 2:
        result = 2
    else:
        result = 1
    if result > maximum:
        return maximum
    return result


def _natural_region_policy(region):
    if region == _REGION_SERIAL_LONG:
        return (
            _SERIAL_LONG_TARGET_PROGRAMS,
            _SERIAL_LONG_MIN_TOKENS_PER_SPLIT,
            _NATURAL_MAX_SPLITS,
        )
    if region == _REGION_BATCHED_LONG:
        return (
            _BATCHED_LONG_TARGET_PROGRAMS,
            _NATURAL_MIN_TOKENS_PER_SPLIT,
            _NATURAL_MAX_SPLITS,
        )
    if region == _REGION_BATCHED_DENSE:
        return (
            _BATCHED_DENSE_TARGET_PROGRAMS,
            _NATURAL_MIN_TOKENS_PER_SPLIT,
            _NATURAL_MAX_SPLITS,
        )
    return (
        _BASE_TARGET_PROGRAMS,
        _NATURAL_MIN_TOKENS_PER_SPLIT,
        _NATURAL_MAX_SPLITS,
    )


def _pick_natural_splits(region, resident_groups, seqlen_k):
    target_programs, min_tokens, max_splits = _natural_region_policy(region)
    needed = (target_programs + resident_groups - 1) // resident_groups
    allowed_raw = (seqlen_k + min_tokens - 1) // min_tokens
    desired = _ceil_pow2_capped(needed, max_splits)
    allowed = _floor_pow2_capped(allowed_raw, max_splits)
    if desired < allowed:
        return desired
    return allowed


def _pick_packed_splits(batch_size, packed_groups, seqlen_k):
    resident_groups = batch_size * packed_groups
    needed = (
        _PACKED_TARGET_PROGRAMS + resident_groups - 1
    ) // resident_groups
    allowed_raw = (
        seqlen_k + _PACKED_MIN_TOKENS_PER_SPLIT - 1
    ) // _PACKED_MIN_TOKENS_PER_SPLIT
    desired = _ceil_pow2_capped(needed, _PACKED_MAX_SPLITS)
    allowed = _ceil_pow2_capped(allowed_raw, _PACKED_MAX_SPLITS)
    if desired < allowed:
        return desired
    return allowed


def _pick_block_n(seqlen_k):
    if seqlen_k <= _SHORT_BLOCK_LIMIT:
        return 16
    if seqlen_k <= _MEDIUM_BLOCK_LIMIT:
        return 32
    return 64


def _pick_num_stages(block_n):
    if block_n >= 64:
        return 1
    return 2


def _pick_reduce_block_d(num_splits):
    if num_splits <= 8:
        return 128
    if num_splits <= 16:
        return 64
    if num_splits <= 128:
        return 32
    return 16


def _workspace(q, batch_size, num_heads, num_splits, headdim):
    device_index = q.device.index
    if device_index is None:
        device_index = -1

    key = (device_index, batch_size, num_heads, num_splits, headdim)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        stats_size = batch_size * num_heads * num_splits
        partial_m = torch.empty(stats_size, device=q.device, dtype=torch.float32)
        partial_l = torch.empty(stats_size, device=q.device, dtype=torch.float32)
        partial_acc = torch.empty(
            stats_size * headdim,
            device=q.device,
            dtype=torch.float32,
        )
        workspace = (partial_m, partial_l, partial_acc)
        _WORKSPACES[key] = workspace
    return workspace



def _launch_natural_reduce(
    partial_m,
    partial_l,
    partial_acc,
    output,
    batch_size,
    num_heads,
    num_splits,
    headdim,
):
    reduce_block_d = _pick_reduce_block_d(num_splits)
    reduce_grid = (batch_size * num_heads, headdim // reduce_block_d)
    _reduce_splits_kernel[reduce_grid](
        partial_m,
        partial_l,
        partial_acc,
        output,
        NUM_SPLITS=num_splits,
        HEAD_DIM=headdim,
        BLOCK_D=reduce_block_d,
        num_warps=4,
        num_stages=1,
    )


def _launch_log2_reduce(
    partial_m,
    partial_l,
    partial_acc,
    output,
    batch_size,
    num_heads,
    num_splits,
    headdim,
):
    reduce_block_d = _pick_reduce_block_d(num_splits)
    reduce_grid = (batch_size * num_heads, headdim // reduce_block_d)
    _reduce_splits_log2_kernel[reduce_grid](
        partial_m,
        partial_l,
        partial_acc,
        output,
        NUM_SPLITS=num_splits,
        HEAD_DIM=headdim,
        BLOCK_D=reduce_block_d,
        num_warps=4,
        num_stages=1,
    )



def run_kernel(
    q,
    k_cache_paged,
    v_cache_paged,
    output,
    cache_seqlens,
    block_table,
    batch_size,
    seqlen_k,
    seqlen_q,
    num_heads,
    num_heads_k,
    headdim,
    page_block_size,
    num_blocks,
    causal,
):
    batch_size = int(batch_size)
    seqlen_k = int(seqlen_k)
    num_heads = int(num_heads)
    num_heads_k = int(num_heads_k)
    headdim = int(headdim)
    page_block_size = int(page_block_size)
    num_blocks = int(num_blocks)

    blocks_per_batch = num_blocks // batch_size
    gqa_ratio = num_heads // num_heads_k
    resident_groups = batch_size * num_heads_k
    region = _select_region(
        batch_size,
        gqa_ratio,
        resident_groups,
        seqlen_k,
    )

    if region == _REGION_SERIAL_PACKED:
        pack_kv = 16 // gqa_ratio
        packed_groups = num_heads_k // pack_kv
        token_tile = 64 // pack_kv
        num_splits = _pick_packed_splits(
            batch_size,
            packed_groups,
            seqlen_k,
        )
        grid = (batch_size * packed_groups, num_splits)
        qk_scale = 0.12751743082459868

        partial_m, partial_l, partial_acc = _workspace(
            q,
            batch_size,
            num_heads,
            num_splits,
            headdim,
        )
        _paged_gqa_packed_kernel[grid](
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            partial_m,
            partial_l,
            partial_acc,
            BLOCKS_PER_BATCH=blocks_per_batch,
            NUM_HEADS=num_heads,
            NUM_KV_HEADS=num_heads_k,
            HEAD_DIM=headdim,
            PAGE_SIZE=page_block_size,
            GQA_RATIO=gqa_ratio,
            PACK_KV=pack_kv,
            PACKED_GROUPS=packed_groups,
            TOKEN_TILE=token_tile,
            NUM_SPLITS=num_splits,
            D_TILE=_D_TILE,
            STORE_PARTIAL=True,
            QK_SCALE=qk_scale,
            num_warps=4,
            num_stages=1,
        )
        _launch_log2_reduce(
            partial_m,
            partial_l,
            partial_acc,
            output,
            batch_size,
            num_heads,
            num_splits,
            headdim,
        )
        return

    num_splits = _pick_natural_splits(
        region,
        resident_groups,
        seqlen_k,
    )
    block_n = _pick_block_n(seqlen_k)
    num_stages = _pick_num_stages(block_n)
    grid = (resident_groups * num_splits,)

    if num_splits == 1:
        _paged_gqa_masked_kernel[grid](
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            output,
            output,
            output,
            BLOCKS_PER_BATCH=blocks_per_batch,
            NUM_HEADS=num_heads,
            NUM_KV_HEADS=num_heads_k,
            HEAD_DIM=headdim,
            PAGE_SIZE=page_block_size,
            GQA_RATIO=gqa_ratio,
            NUM_SPLITS=1,
            BLOCK_N=block_n,
            D_TILE=_D_TILE,
            STORE_PARTIAL=False,
            SM_SCALE=0.08838834764831845,
            num_warps=4,
            num_stages=num_stages,
        )
        return

    partial_m, partial_l, partial_acc = _workspace(
        q,
        batch_size,
        num_heads,
        num_splits,
        headdim,
    )

    if region == _REGION_BATCHED_LONG:
        _paged_gqa_fulltile_kernel[grid](
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            partial_m,
            partial_l,
            partial_acc,
            BLOCKS_PER_BATCH=blocks_per_batch,
            NUM_HEADS=num_heads,
            NUM_KV_HEADS=num_heads_k,
            HEAD_DIM=headdim,
            PAGE_SIZE=page_block_size,
            GQA_RATIO=gqa_ratio,
            NUM_SPLITS=num_splits,
            BLOCK_N=block_n,
            D_TILE=_D_TILE,
            STORE_PARTIAL=True,
            SM_SCALE=0.08838834764831845,
            num_warps=4,
            num_stages=num_stages,
        )
    elif region == _REGION_BATCHED_DENSE:
        _paged_gqa_locality_kernel[grid](
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            partial_m,
            partial_l,
            partial_acc,
            BLOCKS_PER_BATCH=blocks_per_batch,
            NUM_HEADS=num_heads,
            NUM_KV_HEADS=num_heads_k,
            HEAD_DIM=headdim,
            PAGE_SIZE=page_block_size,
            GQA_RATIO=gqa_ratio,
            NUM_SPLITS=num_splits,
            BLOCK_N=block_n,
            D_TILE=_D_TILE,
            STORE_PARTIAL=True,
            SM_SCALE=0.08838834764831845,
            num_warps=4,
            num_stages=num_stages,
        )
    else:
        _paged_gqa_masked_kernel[grid](
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            partial_m,
            partial_l,
            partial_acc,
            BLOCKS_PER_BATCH=blocks_per_batch,
            NUM_HEADS=num_heads,
            NUM_KV_HEADS=num_heads_k,
            HEAD_DIM=headdim,
            PAGE_SIZE=page_block_size,
            GQA_RATIO=gqa_ratio,
            NUM_SPLITS=num_splits,
            BLOCK_N=block_n,
            D_TILE=_D_TILE,
            STORE_PARTIAL=True,
            SM_SCALE=0.08838834764831845,
            num_warps=4,
            num_stages=num_stages,
        )

    _launch_natural_reduce(
        partial_m,
        partial_l,
        partial_acc,
        output,
        batch_size,
        num_heads,
        num_splits,
        headdim,
    )

