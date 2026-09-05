# SESSION STATE CACHE — 供上下文重置后恢复 (2026-09-05 赛后继续优化)

> 目的: 主 session 上下文膨胀/易失。此文件是唯一恢复点, 每次关键进展更新。
> 用法: 新 session 先读此文件 + 各工作区的 NOTES。

## 项目
- paged GQA decode kernel, MetaX C500。最终 67.36 (submission_clean.cu)。
- 赛后继续优化, 目标追 72.71。
- 主仓库: /root/workspace/mx_flash_attn_kernel_opt (已 push 到 github dddd109/...)。

## 分数轨迹 (OJ)
62.21→64.57→64.93→65.14→66.71(nomax)→67.07(ov_safe,case4修复)→**67.36(clean)**。

## 关键事实 (勿重推)
- 本地 SUM ~1455us (clean)。大 case 7-14 各 56-61 分。
- kernel: 64线程CTA=1warp, 16x16 MMA, per-(b,kv) 页 staged smem (V VPAD8, K+4),
  smem 8576B→7 CTA/SM (硬墙), nomax softmax, combine 向量化, split 7-regime。
- spec_read 已证: kernel 是**平衡 smem+MMA 流水**, 超过所有纯读副本 (case13 1.36 eff
  TB/s vs 纯读最好 ~1.2)。"1.82 ceiling"不可复现(机器状态波动)。L2(8MB)无同线复用
  不可利用。ns policy 已最优。case14(kv4,1.04)慢因 V/PV 侧重(非 CTA数/ns/L2)。
- 已证死(勿再试): async bsm 坏; 寄存器/smem 流水(reg墙152); 多kv/大块CTA/warp专用/
  grid同步; scalar FMA 慢5x; pad-drop bank conflict; ns 微调=噪声。

## 专职 Agent 分工 (SEESION 复用关键!)
> **复用方法: task 调用必须带 task_id 参数 = 下方记录的 id。不带=误开新session!**
> 每个专职 agent 在自己的 /tmp/agent_ws/<名>/ 工作, 笔记写在 NOTES.md。

| 专职 | 方向 | 工作区 | 笔记 | 活 task_id | 状态 |
|---|---|---|---|---|---|
| spec_read | 带宽/读/MLP | /tmp/agent_ws/spec_read | SPEC_READ_NOTES.md (396行) | `ses_f8d82e8beffe6ujn5275VnJO5y` | session2完成, session3(续跑指令)待执行: case14 PV解剖 + 寄存器-B QK + K smem squeeze |
| spec_policy | split/grid/combine 调度 | /tmp/agent_ws/spec_policy | SPEC_POLICY_NOTES.md (153行) | `ses_f8d82cddaffedI2d96DWQ7EYfQ` | session1-2完成。发现: combine 串行 launch bubble 值3.4%(长case省7-13us)。下一步: grid order batch>1 (exp_b.cu), 再考虑 mid-ns fuse-in-main |

## spec_read session3 结论 (已完成, 495行笔记)
- QK/PV 解剖 (load-preserving): kill 任一 MMA 阶段几乎免费 (case13 FULL 177.3 /
  killPV 176.9 / killQK 178.1; case14 119.9/115.3/119.5)。compute 完全藏在 load 流水下。
  **推翻 case14 V/PV-heavy 理论** — case14 只是 case13 一半读量, 同 load-limited rate。
  双倍 MMA 只 +1~3us → 有 ~40us 计算余量可偷。
- 寄存器-B QK: intrinsic B 本就是寄存器对 {b0,b1}; smem 只是喂它的 LDS。寄存器读 K 需
  MMA fragment 布局 → 强制 uncoalesced per-thread 16B gather vs 现在 coalesced uint4
  sweep + smem gather → dead end。
- K smem squeeze (KSTRIDE+2 + 2x32bit LDS): 14/14 过但中性到略差。__launch_bounds__(64,8)
  ±0.5us。occupancy 从不是杠杆。barrier/2页、split-major grid、case14 ns<100 全中性/更差。
- **结论: case13/14 在稳健局部最优。所有旋钮(load width/unroll/MMA数/occupancy/smem/
  LDS宽度/ns/grid/barrier)都在噪声内。未 shipped 任何 win, baseline (SUM 1460.7) 仍最佳。**
- **唯一剩余结构方向: 2-page 寄存器预取流水** (P+1 的 DRAM 延迟藏在 P 的 PV 下)。
  ~40us 计算余量可偷。历史全因寄存器墙(>152)失败; 需 launch_bounds(64,8)=128 reg 上限
  守卫, 用 ~16-24 空闲寄存器预取 P+1 的 8 uint4 到寄存器, PV 后 STS。spill 即弃。
  (spec_read session3 未实际实现, 仅推荐)

## 主 session 决策 (2026-09-05, 上下文压缩后)
- 结构到天花板确认。用户批准重构。下一步: 主 session 亲自尝试 2-page 寄存器预取流水
  (历史 agent 全败在寄存器管理, 需精细控制), 而非再派 agent。
- spec_policy session3 (grid order / fuse combine) 返回空未做 — 优先级低 (combine 串行
  bubble 最多 3.4%, 且是 launch 层非 kernel 层)。
- 服务器剩约 1.5 天。目标: 打破 case13 1.36 / case14 1.05 eff TB/s 的局部最优。

## ⚡ 关键发现 (2026-09-05 晚, 主 session, 决定性 NEGATIVE — 2-page 预取方向正式关闭)
> 这几条纠正了历史笔记里的**错误结论**, 是当前最硬的事实, 勿重推:

1. **寄存器可观测/可控制手段找到了**: `mxcc ... -resource-usage` 打印真实占用
   (clean kernel: **150 MTregisters + 52 STregisters**, 0B stack);
   `-maxrregcount=N` **真实生效**(150→130 无 spill, →128 12B stack, →96 148B stack)。
   而 `__launch_bounds__(64,8)` 的第二参数在 Maca 上**被静默忽略**
   (warning "set minimum blocks' number is illegal ... will be ignored") —
   它从未真正限制过寄存器。**因此历史 "occupancy 从不是杠杆 / lb8 中性" 的结论是无效的**
   (测的是空操作), 现在才有真正手段测它。
2. **寄存器上限是硬 MLP 壁, 不是 occupancy 杠杆**: A/B case13/14 (interleaved):
   - clean (150 regs): case13 175-176us / case14 118us (SUM 294)
   - `-maxrregcount=146` (真降到130 regs, 0B stack): **285/251us (SUM 536, +82%)**
   - `-maxrregcount=128`: 285/253us (同上)
   => 砍寄存器序列化了 in-flight 页加载 (编译器用 ~150 regs 当 load-MLP 槽)。
   增加 CTA 数带来的 occupancy 收益被逐页 load 串行化吞掉还倒亏。
   **150 regs 是这台机器/这个 load-pacing 结构的最优, 不是天花板缺陷。**
3. **2-page 寄存器预取 (SESSION_STATE 之前唯一剩余方向) 由此机械性关闭**:
   它需要 +16~24 regs, 但 152 墙内没有空闲寄存器; 强行压回 128 让路 = 上面的 +82% 灾难。
   曾经的 "spill 即弃" 假设被证明: 任何 reg 增减都 ≤ 现状。**不要在寄存器预取上再花时间。**
4. **case14 低 ns 无剩余**: case14_lowns (exp/, 未写入 NOTES) ns50-100 中 100 最佳
   (0.72→1.05 TB/s 单调, ns=100 = knee, 已 shipped)。无未收割 win。

## 主 session 结论 (2026-09-05 晚, 决定后)
- **正式停止所有结构/寄存器方向的尝试**。2-page prefetch、occupancy、smem squeeze、
  lb8 全已机械/实证关闭。case13/14 的 load-pacing + 150-reg 结构就是本设计的天花板。
- 唯一还未正式关闭的角落 (低价值): spec_policy 的 combine 串行 bubble (≤3.4%, launch 层)。
- **提交物规范已复核**: build/clean.so = 67.36 等价物, verify_real.py ALL-14 PASS,
  bench_all case13 175 / case14 118 (与历史一致)。shippable 状态无变化。
- 剩余服务器时间应投入: (a) 若有同事新 SOTA 则 diff 学习; (b) 微调 ns policy 用 OJ 真值;
  (c) 归档清理。若无新输入, 67.36 即终局。

## 协作注意
- GPU 共享, interleaved A/B 必须。机器状态波动大(绝对 TB/s 跨分钟不可比)。
- 提交物规范: verify_real.py 全过(edge 1-3 match 1.0) + bench_all.py SUM。
- 每轮结束把进展 append 回此文件 + 各自 NOTES。
