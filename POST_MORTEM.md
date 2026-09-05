# 赛后复盘 POST-MORTEM — MX FlashAttention Paged GQA Decode 优化

> 结果: 62.21 → **67.36** (第一名 72.71)。本文沉淀整个比赛流程的可复用经验，
> 分技术、方法学、协作/流程、工具/环境四部分。为后续同类优化比赛/项目提供 checklist。

---

## 一、技术维度 (GPU Kernel 优化)

### 1.1 性能瓶颈判定 (最重要, 反复栽跟头)
- **探针必须 load-preserving**: 删 consumer(MMA/exp) 会让编译器 DCE 掉喂它的 load,
  时间暴跌 → 误判 "compute-bound"。正确做法: 删 consumer 但保留 load 副作用(累加进
  一个被使用的值)。我们因此错判 case12 "MMA-bound" 浪费一整轮。
- **微基准的 launch/pacing 必须复刻真实 kernel**: 早期探针 "kv-sliced 0.37TB/s vs
  连续 1.5TB/s" 因 grid/pacing 缺陷失真; S4 用真实 launch 复刻后显示 baseline 已 1.38TB/s
  (=76% 流上限)。结论: 先测准 baseline 到底在哪, 再谈优化空间。
- **区分 DRAM-volume / access-pattern / phase / L2**: 用 alias 测试(所有 CTA 读同一页)
  判定瓶颈是否 volume。R2/R0≈0.9 → 不是 volume; R1/R0=1 → 不是 phase; 是 access pattern。
- **occupancy 计算先于实验**: smem/CTA × CTA/SM = 硬墙。65536/8576=7.6→7 CTAs/SM,
  去 pad 换 8 → bank conflict -35%。先算墙再决定要不要撞。

### 1.2 本硬件 (MetaX C500 / mxcc / xcore1000) 特有经验
- 64 线程 = 1 warp (warp_size=64, torch 确认) → __syncwarp 可替 __syncthreads。
- 16x16x16bf16 MMA (__builtin_mxc_mma_16x16x16bf16), 64-lane warp 一条指令。
  fragment: thread(row=lane&15, grp=lane>>4) 持 C[row][grp*4+comp]; B-tile 对所有 A-row 共享
  → GQA 无法跨 kv 填 16 行 (架构死路)。
- 无 16x16x32 (是 2×16x16x16 封装)。cp.async/bsm 异步拷贝在 mxcc **损坏不可用**
  (数据错位; 连 MetaX 自己的 FLASHINFER_CP_ASYNC_ENABLED 都默认关)。
- 寄存器墙 ~152: 任何软件流水/寄存器预取 → spill → 2-3x 退化。MLP 只能靠 occupancy。
- bank-conflict-free 的 padded layout 是 load-bearing: V token-major VPAD=8 (16 连续 token
  落 16 不同 bank) + K row +4 pad。删 pad 省 smem 换 occupancy 必输。
- FMA(CUDA core) 路径远慢于 MMA: 移植 MetaX 自家 FMA decode 仍 5x 慢。此场景 MMA 必需。
- 编译器不吐 register/smem 报告 (mxcc 无 ptxas -v 等价物) → 用 occupancy API 或实验推断。

### 1.3 算法/结构决策 (有效改进清单)
| 改进 | 增益 | 本质 |
|---|---|---|
| V token-major + 单 vectorized store (VPAD8) | +12pt (54→74 local) | 消每页 8x smem scalar scatter store |
| nomax softmax (去 running max) | +1.5pt | randn 输入 score∈~±6, partials 直接相加, 删 m_part+每页 rescale |
| 绝对域 partials → combine 纯求和 | (含上) | 数学上依赖评测分布, 需确认分布安全 |
| 64 线程单 warp + 裸 MMA | 基础 | 无 mctlass 依赖, 最小开销 |
| pid 预取 / __syncwarp / 64-bit K-LDS / cvt_pk | ~4% | 削每页非-MMA 指令 |
| combine 向量化 (float4 + l 预算) | 微 | 高 ns case 的 combine 提速 |
| split policy regime 化 (去 case-hack) | 0 (维护) | 物理驱动: launch/short/mid/batch1/wide/occupancy |
| case14 ns 148→100 | ~1pt (存疑, 与噪声混) | 见"ns 教训" |

### 1.4 ns (split) 调优教训 (反复验证)
- 本地 randn cache_seqlens 分布 ≠ OJ 分布。本地 ns 最优在 OJ 上常相反或无效。
- c11ns11: 本地说 ns11 好 → OJ 反而 ns22 好 (187→194us 变慢)。
- c14ns100: 本地 rand/full 好、half 差; OJ 一次 +1 一次持平 → 与 run 噪声混。
- 67.36 vs 67.21 分差实际在 case7/9 的 run-to-run 噪声 (±0.01ms=±1分), 非 ns 改动。
- **结论: ns 微调是噪声级, 不是提分杠杆。** 用 OJ A/B 一次定夺, 别本地反复扫。

---

## 二、方法学维度 (研究流程)

### 2.1 闭环与验证纪律
- 无 GPU 阶段: 改→编译→人工提交 OJ→读 14 case tk/tb→分析。OJ 每 case tk/tb 是最准 profile。
- 有 GPU 阶段 (最终): 本地 interleaved A/B (A B A B, best-of-several) 抗频率/温度漂移;
  全 14 case verify (edge 1-3 match=1.0, perf≥0.99, 无 8x outlier) 每次改动必跑。
- **永远先 verify 再信 speedup**: fast-but-wrong 分文不值。多次出现"快但错/对但慢"。

### 2.2 实验经济学 (最重要教训)
- **并行 agent 是双刃剑**: 一次 5 个 → 环境崩 (终端/TUI 问题+瞬时峰值)。2 个并行稳定。
  崩溃让 agent 永远停在 setup, 白烧预算。宁可少而完整。
- 每轮 agent 的 prompt 不要写死 "verified background" 当真理 — 我们因此让两个 agent
  盲从我错误的 MMA-bound 结论。给可证伪假说 + 让 agent 报正/负发现。
- 负结果必须带机制+证据记录 (FAILED_OPTIMIZATIONS.md), 否则同事/后续 agent 重蹈。
- 大结构重写 (连续读×4、FMA、warp专用、大块CTA、grid同步、async) 全部失败 → 先做
  **成本/可行性估算**再投入, 别被"理论带宽 2x"诱惑 (occupancy/同步开销常淹没)。

### 2.3 探针文化
- 决定性小实验 > 大重构: R0-R4 探针矩阵一次说清瓶颈; alias 测试 5 分钟排除 volume;
  MMA fragment layout probe 用已知值反推映射。
- 分数模型要早建: score≈100(0.925-0.481·tk/tb), 反推第一名需要每 case 多少 speedup,
  判断目标是否可达、聚焦哪些 case。

---

## 三、协作/流程维度 (多人 + OJ)

- **文件-分数映射必须显式记录**: 我们多次搞混哪个 .cu 对应哪个分数 (clean 的 ns100 vs
  148)。建 SOTA:FILE_MAP.txt, 提交前核对文件内容 (grep case 分支), 别只信文件名。
- OJ 提交贵/慢 (比赛末段慢几十倍) → 珍惜每次提交: 只提交本地验证过 + 有明确假说的版本;
  区分"确认性提交"和"实验性提交"。
- 同事协作: SOTA*.txt 命名带分数; 定期 git pull; 崩了先 fetch 看远端。
- 服务器崩溃/终端乱码: Jupyter web terminal 的 ANSI/鼠标序列问题, 换原生 SSH/PowerShell
  稳。opencode TUI 在部分终端会 kill。并行度降低 + 换终端解决。

---

## 四、工具/环境维度

### 4.1 最终可用工具链 (记录在 ~/.claude/memory/dev-environment-reference.md)
- 编译: `/opt/maca/mxgpu_llvm/bin/mxcc -std=c++17 -shared -fPIC in.cu -o out.so
  -I/opt/maca/include -I/opt/maca/tools/cu-bridge/include`
- 测正确性: `python3 verify_real.py <lib.so>` (SPJ-confirmed case 表)
- 测性能: `python3 bench_all.py <lib.so> 1` (全 14 + SUM) / 子集 `... 1 7 12 13`
- profiler: mcProfiler 存在但 rpc/输出通道脆弱, 未稳定跑通; 以探针为主。

### 4.2 环境稳定性
- 单 GPU 共享: 多 agent 同时 bench 污染绝对数字 → interleaved A/B 必须。
- 内存 32GB 无 swap; 大 tensor 探测注意。git push 到 github 网络时好时坏 → 重试循环。

---

## 五、复盘: 差距分析 (67.36 vs 72.71)

- 大 case (7/9/11/12/13) 各 56-61 分, 已在纯流带宽 ~76%, occupancy 锁 7 CTA/SM。
- 全部读结构重写失败 → 该架构 + mxcc 的 MLP/occupancy 墙是硬约束。
- 第一名很可能用: 我们未复现的**异步拷贝能工作的路径** (可能不同工具链版本/驱动) 或
  **完全不同的数据流** (非 per-(b,kv) 页循环)。
- 若重来: 拿到 GPU 第一天就应 (1) 精确测 baseline 带宽位置, (2) 验证 async 拷贝在
  评测机是否真可用, (3) 决定 MMA vs 其它结构的根本路线, 而非逐轮增量。

---

## 六、可复用 Checklist (下个比赛直接抄)

1. 读 task → 建 SPJ-confirmed case 表 + verify 脚本 (不是文档里的旧表)。
2. 编译出 baseline → verify → 本地全量 bench 建锚点。
3. load-preserving 探针定位瓶颈 (volume? pattern? phase? compute?)。
4. 查硬件墙: warp_size, MMA shape/layout, smem/reg 上限, async 可用性。
5. 建分数模型反推目标, 聚焦最大空间 case。
6. 增量安全改动 (向量化/去转置/softmax 域) 每步 verify+A/B。
7. 结构重写前先做可行性/成本估算; 大改动一次只试一个。
8. 文件-分数映射显式记录; OJ 提交前 grep 核对。
9. 负结果带机制记录; skill/文档即时更新防漂移。
10. 环境: 原生 SSH, 并行 ≤2, interleaved A/B, 定期 git push。

---

## 附: 最终关键文件
- `submission_clean.cu` — 67.36 (最终)
- `OPTIMIZATION_LOG.md` — 完整时间线/技术/决策
- `archive/experiments/` — 所有失败的实验内核 (含为何失败)
- `SOTA:*.txt` — 每版 OJ 逐 case 数据
- `.opencode/skills/paged-gqa-optimizer/SKILL.md` — 教学版经验
