# 第二代 (Gen2) 思路梳理 - 2026-09-04 (v66.71, MMA-bound breakthrough)

## 现状
- 最高分 **66.71** = submission_nomax.cu (去 running-max softmax)。
- 第一 72.71，目标差 ~6 分 = cases 4-14 平均再快 ~1.25x。
- 分数模型: score = 100*(0.925 - 0.481*tk/tb)，total=14 case 均值。

## 第二代的核心洞察（探针验证）
用"删 MMA"探针把每个大 case 的耗时拆成三部分（等内存/软max vs QK-MMA vs PV-MMA）:

case7  (b64 kv8 2048): full 211.6us | 去PV(留QK) 118.8 | 去QK(留PV) 135.1 | 纯加载 ~24.6
case9  (b32 kv8 4096): full 222.1 | 去PV 109.7 | 纯加载 24.3
case12 (b8  kv8 32768): full 427  | 去PV 254   | 纯加载 38
case11 (b16 kv4 12251): full 175  | 去PV 121   | 纯加载 27

结论:
1. **MMA 执行占 85-90%**（memory+softmax 只有 ~25-40us）。不是内存/延迟瓶颈。
2. **PV(V-MMA) 比 QK 贵**: case7 去QK留PV=135 > 去PV留QK=118。PV 是更大头。
3. **16x16x16bf16 的 B 片在单条 MMA 内固定** → GQA 多 kv 填 16 行的想法**架构上不可能**
   (kv8 gqa=4 只有 4/16 A-行有用, kv4 gqa=8 只有 8/16)。trackA2kv 合并块实测 2.5x 退化证实。
4. ns 扫描 60-320 全平、8x 页复用只快 9% → 都吻合"MMA issue 饱和"。

## 为什么 PV 比 QK 贵（第二代要主攻的方向）
- QK: C[16 head][16 token] = q(head) x K(token)。A=q 只有 gqa 行真实(kv8: 4/16=25%)，
  B=K 全 16 token 有用。MMA 浪费在 A 行。
- PV: C[16 token][16 dim] 或类似 = p(token) x V(token)。需要 p 是 softmax 结果。
  **PV 的 16 "行"= 16 token 全有用，但每个 token 的 p 来自不同 head?** 需复核 PV 的
  A/B 映射 —— PV 贵可能因为它是按 (head,dim) 输出而每 head 的 p 不同 → 无法共享。
- 待测假说: PV 每输出行是 (head, dim)，16 行 = 16 dims，A=p[token]按 token 排列，
  但 gqa 个 head 需要 gqa 份不同的 p → PV 对每个 head 单独做 → 重复 4/8 次?

## 第二代候选方向（按优先级）
A. **[主攻] PV 结构重构**: 确认 PV 是否在重复做无用行。若 PV 的 16 MMA 行 = 16 dims
   而 A=p 只有 gqa 组不同，尝试让一次 PV-MMA 覆盖 16 个真实 dim 行（不再每 head 重做）。
B. **减少 MMA 指令数**: 128-dim 的 QK/PV 各需 8 条 MMA(k16)。若 PV 能一次算 2 个 head
   的相同 token p（共享 B=V），指令减半。核心矛盾 = 每 head p 不同。
C. **接受 MMA 总量**, 优化 MMA 之外的 ~10%: memory 与 MMA 完全重叠 (理论上限只是
   max(mem, mma) 而非 mem+mma，但 mem 只占 10%，收益有限)。
D. **对照第一名**: 72.71 ≈ cases 4-14 平均 tk/tb 0.41 (=2.43x)。可能第一名用了
   非 MMA 路径(标量 FMA 稀疏 4-head)或不同切分。

## 记录的反面教训（勿重蹈）
- 16 行填满(多kv/块): 架构不可能 + trackA2kv 实测 2.5x 退化。
- smem 双缓冲 / 寄存器预取跨 barrier: 全部 ~2x 退化 (mxcc 寄存器墙)。
- split 微调: 已到最优 (r3 穷举扫描确认每 case baseline 最佳)。
- 本地 randn cs 分布 ≠ OJ 分布: 只信 OJ tk 对比 / 本地 speedup 相对。

## Gen2 中期结论 (2026-09-04, 探针/布局/扫描全部完成)
- MMA 布局彻底解码: C[m=16 head-rows][n=16], thread(row,grp)->C[row][grp*4+comp];
  A fragment = thread(row,grp) 持 q[head=row][k=grp*4..+3]; B fragment 持 K[token=row][k=grp*4..+3].
- QK/PV 都是 8 MMA/page (k=128 分 8×k16), 条数结构性最小。16x16x32 不存在(是2条mma封装)。
- kv8 的 A 行只有 4/16 有用 (gqa=4), kv4 8/16 —— B 侧 16 token 全有用, 浪费在 A 侧 head 行,
  架构上无法跨 kv 填(每 MMA B 固定)。trackA2kv 实测 2.5x 退化证实。
- MMA issue 饱和: case12 ns 60-320 全平 (3840+ CTA 已填满 104SM x8); case13(batch1 仅8单位)
  需 ns90 才填满 (720 CTA), ns30=299->ns90=181。分派已最优。
- PV ≈ QK 成本 (各 ~100us/8MMA on case7), 都随 MMA 条数线性。
- 纯内存+softmax 底 ~25-40us (仅 ~10%)。MMA 重叠收益上限 ~10%。
- 本地==OJ on 大 case (7/8/9/12/13 差<6-15%) => 本地优化可信。

## 结论: 达到 72.7 需 ~25% (cases4-14 1.25x), 而 MMA 条数/行效率/occupancy 均已结构性封顶。
## 剩下的可能杠杆(诚实评估):
L1. MMA 与 memory 重叠优化 (~10% 上限)。
L2. PV 的 A 侧若也能压进更多真实 dim 行(16 行全用) — 需复核 PV 的 m 到底是 head 还是 dim。
   若 PV 的 m=16 是 16 个 dim(全有用)而 head 在别处, PV 可能本就高效; 若 m=head 则同 QK 浪费。
L3. 第一名可能用不同算法规避 GQA MMA 浪费(标量 FMA for 4 head? 但 MMA 更快通常是)。
L4. 其它 case 的 launch/combine 优化 (小 case 已近 floor)。

## 等待 OJ 结果期间的方向梳理 (2026-09-04, ov = 1449us 待 OJ)
ov 版 per-case speedup: 慢 case 7(1.28) 12(1.35) 13(1.22) 14(1.29) <- 最大头.
已确认(round5 两 agent + 我修正): 长 sweep 是 DRAM-load-latency / per-page-load-issue 受限,
不是 MMA 也不是纯带宽。r5_overlap agent 的核心建议:
"唯一没试过的杠杆 = 不牺牲 padded layout 的前提下提高 CTAs/SM"(memory-latency-bound,
更多 resident warp 才能藏每页 load stall)。当前 smem 8.5KB -> 8 CTAs/SM.
r5_alt agent 的核心建议: 每页加载模式(kv-sliced)在 batch1 只有 ~0.77TB/s 上限;
整页连续读能到 1.5TB/s 但所有实现路径(smem/reg/direct)都因 occupancy/reg墙/coalescing 失败.
它提的激进方向: 64 lane 协作整页连续拉取(32KB)到寄存器(~128B/thread) 再用 shuffle 蝴蝶重分布;
或 co-schedule 同一页的多个 kv-CTA 使 slice 读变成一条连续 DRAM 流.

候选方向(等 OJ 后按结果选):
A. 提高 CTAs/SM: 砍 smem 到 <8KB 且保持 VPAD=8 无冲突 -> 9-12 CTAs/SM. 
   具体: K tile 读一次后可放寄存器? 或 V 用 512B-chunk 的 ldg_b128_bsm 部分直接(agent 说该范围可靠)。
B. co-schedule 同页 kv-CTAs: batch1 (case 10/13/14) 的 kv=8 个 CTA 读同一物理页的 8 个 kv-slice,
   若它们同时驻留 SM, DRAM 能看到连续整页流。alt_v1 的 kv-fastest grid 已朝这个方向(case13/14 +2-4%)。
   -> 扩展: 不仅 grid 顺序, 还可用 L2 局部性 (同页 8 slice 共 32KB, L2 应能缓存, 第2-8个 CTA 命中)。
C. 降低每页 load 指令数: stage 每 thread 8x uint4 (4 tok*2(K,V)) -> 是否可 8B 对齐少指令或用 32B.
D. 若 OJ 显示某 case 分布敏感(像之前 case11), 针对性调 tps/ns。

正确性/回归纪律: 每次改动 full 14 case verify + interleaved A/B; regs<152; CTAs/SM 不掉.

## 深夜自主推进记录 (r6 完成后)
- r6_lat definitive diagnostic: per-page ~1438ns @ ns1, ~92% = LDG->smem stage. At ns>64 kernel
  hits ~1.0-1.07 TB/s = DRAM peak on BW-heavy cases (7/12). async pipeline only helps the
  LATENCY-bound batch1 cases (13: current 171us, all-capacity floor ~98us @ ns64-90; 14 similar).
- => async-copy (bsm) is THE remaining lever for case 13/14 (~70us potential each, score 58->~70).
- Reproduced the async failure in real kernel (match 0.0-0.12). Working probes exist (r5_overlap
  bsm_sem*.cu). Next: bisect why bsm-fed smem fails MMA reads in real kernel while probes pass.

## case 11 ns experiment candidate (needs OJ A/B, NOT locally resolvable)
- case11 (b16 kv4 12251): local-rand optimal ns=22 (149us); OJ=187us means OJ dist is LONGER
  than local-rand. Under long distributions (half/full), ns=11 beats ns=22 by ~9%
  (231 vs 254us half-dist). CONFLICT: changing policy to 11 makes local bench WORSE (195us)
  but likely helps OJ. => candidate OJ experiment: set case11 ns=11, submit, compare.
- case9 (b32 kv8): ns=11 optimal in both local and half-dist. No change.
- Do NOT fold this into the committed kernel until OJ confirms (would regress local gate).
