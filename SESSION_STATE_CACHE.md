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

## 协作注意
- GPU 共享, interleaved A/B 必须。机器状态波动大(绝对 TB/s 跨分钟不可比)。
- 提交物规范: verify_real.py 全过(edge 1-3 match 1.0) + bench_all.py SUM。
- 每轮结束把进展 append 回此文件 + 各自 NOTES。
