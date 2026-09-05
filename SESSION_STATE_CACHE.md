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

## spec_policy 待续任务 (session3)
- 完成 exp_b.cu: batch>1 用 3D grid (kv,b,ns) 让同页 kv CTA 相邻, 测 case7/9/12。
- 若 grid 无效: 试 fuse-in-main (mid-ns case 用 atomic 融合, 免第二 launch)。
- 全部 verify ALL-14 + interleaved A/B。

## spec_read 待续任务 (session3)
1. case14 PV vs QK load-preserving 解剖 (只换 PV MMA B operand 为常数, 其余live)。
2. 寄存器-B QK 可行性 (MMA B 能否来自寄存器) → 可行则 K 不进 smem。
3. 不行则 K smem squeeze (KSTRIDE+2 + 2x32bit LDS) 换第8 CTA/SM。

## 协作注意
- GPU 共享, interleaved A/B 必须。机器状态波动大(绝对 TB/s 跨分钟不可比)。
- 提交物规范: verify_real.py 全过(edge 1-3 match 1.0) + bench_all.py SUM。
- 每轮结束把进展 append 回此文件 + 各自 NOTES。
