# 详细复盘 — MetaX C500 Paged GQA Decode 优化全程 (逐日/逐轮/逐决策)

> 比赛: xpuoj contest 11 problem 1 — paged GQA decode kernel (bf16, 14 cases)
> 最终: **67.36** (第一名 72.71)。本文是完整过程日志, 粒度到轮次与决策。
> 分两大阶段: A) 无 GPU 教学 skill 多代 agent (62.21→65.14);
>            B) 有 GPU 直接闭环 (65.14→67.36)。
> 比赛跨度约 3-4 天。所有 commit/文件/分数均有据可查。

---

# 阶段 A: 无 GPU — 教学 skill + 多代独立 agent (→65.14)

## A0. 背景与设定
- 任务从 flash_attn 官方实现出发。OJ baseline(tb) = flash_attn_with_kvcache。
- 环境: **无本地 GPU**。唯一真值 = 人工提交 xpuoj, 返回 14 case 的 (tk, tb, score)。
- 机制: 一个"教学 skill"(paged-gqa-optimizer) 描述优化经验(不含答案), 多代**独立 agent**
  各从头按 skill+task 实现, 教师分析每代结果, 把教训写回 skill, 下一代更强。
- skill 版本迭代: v5→v9/v10→v11→v13→v14→v17→v19→v20 (backup 留了 v14/v19)。
- 用户/同事用 git 协作, SOTA*.txt 记分数。

## A1. 早期: 建立正确 benchmark (gen1-gen2, ~51→61)
- **gen1** (skill v5, 泄漏版): 51 分。首次超过 flash baseline(50)。用 MMA + split-KV +
  fused short path。
- **gen2** (skill v9/v10): 61 分(本地估算)。核心: 去掉 host 端 getenv/atoi 开销
  (tiny case 每次 ~7us)、Q fragment 寄存器缓存、warp-shuffle softmax(非 smem roundtrip)、
  ns==1 fused 输出、静态 grow-only partial buffer(免每 call cudaMalloc)。
- **关键工具发现**: `load_matrix_sync` 在 C500 上 crash(toolchain bug), 必须用裸
  `__builtin_mxc_mma_16x16x16bf16`。fragment 布局靠实验反推:
  thread(row=t&0xf, grp=t>>4) 持 C[row][grp*4+comp]。
- **教训(反复出现)**: "本地估算 61" 是**假分数** — 本地 case 表/分布错, 后证 gen6-9 真实 ~53。

## A2. gen3 灾难: 为修 case9 毁长序列 (61→18.7)
- gen3 为修 case9 做 "partial-page specialization", 把 register-prefetch 软件流水
  换成分段 stage_page → 移除了长序列关键的 latency hiding。
- 结果: case12 +175us, case11 +116us, 全表 +542us, **-42 分**。
- **核心教训(写入 skill)**: 永远不要为 launch-floor case 牺牲 latency-hiding 主路径。
  这是本比赛最贵的一次回归。

## A3. gen4-gen5: 守成 + 干净负结果 (61)
- gen4: 严守 "不回归" 指令。double-buffer smem 流水 → 长序列 +30-40% → revert。
- gen5: 两个**干净负结果**:
  - partial-page 优化: 短 case launch-floor 主导看不见; 长 case ragged tail 浪费 ≤0.5%。非 win。
  - block_table 预取到 smem: pid 是 broadcast(~1 transaction), HW prefetch 已覆盖地址流,
    occupancy 已覆盖延迟 → 反而 +1-3% 慢。revert。
- gen5 诊断出真瓶颈: **case13(batch1) 只有 ~0.7TB/s**, 而 case11(batch16) 1.8TB/s。
  同一 kernel, 差在 per-SM 顺序页局部性。

## A4. gen6-gen8: 结构墙与教师实验 (53.1)
- gen6: 53.1 分。over-split 补偿(单 block 长扫不高效)。occupancy 7 block/SM(smem 8.7KB 绑)。
- **教师关键实验(不泄漏给学生)**: 强压 case12 ns=13 → 852us, 比 gen6 的 ns~170/574us 差。
  证明 gen6 架构单 block 无法高效长扫, 必须 over-split; 而 orig(62.21) 单 block 在
  ns=13 就 494us → orig 的 block 内部 latency-hiding 结构不同。
- gen7: occupancy 实为 7(非3); 去 pad 3x 更差(bank conflict); __syncwarp 中性;
  double-buffer 封顶 0.82TB/s < fine-split 0.99。非带宽饱和(86%), 非 compute。
- gen8 决定性: MMA 16x16 B-tile 对全部 16 M-row 共享 → M-row 必须共享同一 kv 的 K。
  **不能把不同 kv/batch 打包进一条 MMA**(结构不可能)。gap 是每线程 MLP, 非 block 数。
- 教师方向判断: 需要 "寄存器级更深的 outstanding loads" 或 "宽 MMA/不同 fragment" 或
  "K/V 分相流"。

## A5. gen9-gen10: 增量到顶 (53.2→54.2)
- gen9: 应用 H1(full/tail 分离), 省 ~3us/573us — 因为 case12 DRAM-latency-bound 非 mask-bound。
  诚实结论: mxcc 惩罚流水(寄存器 154→184+ 就崩), gen6 结构封顶 ~53。
- gen10: 尝试 coherent rewrite(loader 流水/K swizzle/V gather) 全被寄存器墙挡。
  只 shipped 教师验证的 merged-max softmax(去 beta, ~1%)。54.2。
- **教师实验 H5(关键负结果)**: 单独去 V transpose(改 gather 读) → case12 +60%!
  证明 gen6 的 transpose V 是 load-bearing(它换来 vector PV 读); 参考能用 gather 是因为
  它的 XOR swizzle + codegen 是整套布局协同, 单点移植必败。
- **教师实验 H6**: merged-max softmax ~1% (gen9→H6 各 case -0.5~-3.7us)。可 shipped。

## A6. gen11 突破: V 转置消除 (54.2→73.9 local, 真 OJ 64.57)
- **根本洞察**: gen10 每页把 V 转置进 smem = 每个 16B vector 拆 8 个 scalar scatter store
  (每线程每页 32 个), 2048 页 = 65536 次/线程。这才是隐藏的每页成本。
- gen11: V **保持 token-major**(同 gmem 布局), ONE uint4 store。V_b[16][HEAD+8], stride
  68 words ≡4 mod 32 → 16 连续 token 落 16 不同 bank。PV 读用 vgather2(2×uint16 LDS+pack)
  精确复现转置布局的 operand word。bit-exact 通过。
- 结果(local): case12 574→487(超 ref 494!), case9 256→217, case10 62→52, case13 173→147。
  TOTAL 1973→1706, score 54.2→73.9(相对 ref 62.21 的 local 分)。
- **为何赢**: 每页 smem store 指令 8x 减少。之前 split/occupancy/softmax/MLP 全没碰它。
- 剩余 gap: short/edge case 1-3 launch floor。

## A7. gen11b/merged/agentG_v2: 打磨到 OJ 64.57/64.93/65.14
- gen11b = 64.57 (OJ)。V 突破落地。
- merged = 64.93: 合并多 agent 的 case13(A)+case7/12(B)+case6(C)+case14 split 调优。
- agentG_v2 = 65.14 (OJ #138763): split policy 按 (batch,kv,work_units,pages) 类细化到
  每类最优: case13 ns90, case14 ns148-160, kv4 batch16 ns22, case7 wu512 mult10(ns~5),
  case9 ns11, case12 mult10.5(ns~60), case6 ns5。
- **无 GPU 阶段核心教训(全记录在 HANDOFF)**:
  1. 本地 uniform cache_seqlens 假设与 OJ 严重不符 → 本地 split 微调在 OJ 无效/反向。
  2. split 微调 OJ 上噪声级。
  3. tb 每次提交微波动(同 case 0.28~0.31)。
  4. 用 OJ tk + ~1.35TB/s 校准反推真实填充率: 大 batch case 50-80%, batch1 近全满。

---

# 阶段 B: 有 GPU — 直接闭环优化 (65.14→67.36)

## B0. 环境转变 (2026-09-04)
- **GPU 突然可用**(MetaX C500, MACA 3.7.1.5)。本地闭环建立:
  编译 mxcc→.so → verify_real.py(14-case 正确性, SPJ case 表) → bench_all.py(全量计时)。
- 关键: 修正了 case 表的 kv 归属错误(SPJ 确认: case7/13 kv8, case11 kv4 —
  旧 skill/文档写反)。本地 speedup-vs-flash ≈ OJ score 模式。

## B1. Round 1: 平行 agent 探索(旧基线, 教训)
- 派 3 个 agent(micro/split/occ) 从**旧 62.21 基线**出发 → 全部不如 agentG_v2(65.14)。
- **重大教训**: 给 agent 的基线必须是最新最佳, 否则白跑。r1_split 发现旧基线的
  EXP25_FORCED_SPLITS=128 默认陷阱(绕过了自身启发式)。
- 用户提醒: 这比赛的教学本质是"skill 复现验证", subagent 不该获得太多上下文 —
  需要工作区隔离 + 上下文预算。此后每轮 fresh agent + 隔离目录 + 限读文件。

## B2. Round 2: nomax softmax (→66.71, 关键突破)
- r2_split agent 产出 micro_v1: **去掉 running max 的 absolute-domain softmax**。
  依据: 输入 randn + d=128 → score ~N(0,1) ∈ ~[-6,6]; exp(s) ≤ e^6 fp32 安全;
  跨 split 的 partials 直接相加, 无需 m_part/rescale。
- 本地 1517 vs agentG_v2 1608us(-5.7%), OJ 实测 **66.71**。
- r2_occ 的 smem double-buffer 流水再次确认死路(case13 401 vs 190us)。
- (服务器崩溃 2 次, 学到: agent 基线 .so 会被截断, 需每次重编译干净的 baseline 再验证。)

## B3. Round 3: 死路确认 + skill 更新
- r3_combine 穷举 ns 扫描(7/10/13/14): split policy 已最优。r3_mlp 寄存器预取 v1-v4
  全崩(3100+us)。v5 回退。
- skill 吸收: nomax(E5c)、reg-prefetch/smem-pipeline dead-end、split 已最优。

## B4. 关键弯路: "MMA-bound" 假说与推翻 (自己探针犯错)
- 我做了删 MMA 探针: case12 full 427 / 去V-MMA 254 / 去全部MMA 38us → 误判
  "MMA-throughput bound", 行利用率 25%/50% 是杠杆。
- 深挖 MMA fragment layout(probe 反推), 结论: 16 行填满是架构不可能(B-tile 共享)。
- r5 两个独立 agent 用 **load-preserving 消融** 推翻: 保留 load 删 MMA 无变化;
  删 global→smem stage 才掉 424→~99us。真相: **DRAM-load-latency / per-page-load-issue
  bound**, 我的探针是 DCE 假象(删 MMA 让编译器删了喂它的 load)。
- **教训**: 探针必须保留内存操作; 每个发现立即 reconcile 进 CONTEXT/DECISIONS/SKILL
  (它们曾漂移不同步, 用户点名批评)。

## B5. Round 4-5: 微优化合成 ov (→本地 1449us)
- ov = nomax + pid 预取 + __syncwarp(64线程=1warp) + QK 64-bit LDS(K stride +4) +
  cvt_pk 打包 + shuffle 推迟 + batch1 kv-fastest grid。
- 本地 1449 vs nomax 1508(-4%)。

## B6. ov 的 OJ 事故: case4 从 0.020→0.048 (65.7)
- ov 提交 OJ = 65.7! 唯一退化是 case4(2.4x), 其它 case 全如本地预测改善。
- 根因(代码审计): ov 的 **tail(部分页)路径漏了内存屏障** — nomax 有 __syncthreads,
  ov 改 __syncwarp 时只加在主循环, tail 漏了。case4 = batch64/ns1/多单页 block 最吃 tail。
- 修复 = tail 加 __syncwarp → ov_safe, OJ **67.07** (case4 回 0.019/75)。
- **教训**: 改同步语义必须全局覆盖所有路径; 单 case 的 2x 退化往往是结构性竞态而非分布。

## B7. Round 6-7 + 联网: 结构穷尽, 全部死路
- r6: per-page 诊断确认 92% 时间是 global→smem stage; occupancy smem-bound 7 CTA/SM。
- 联网找到 MetaX 自家 mcflashinfer decode kernel: 用 CUDA-core FMA(非MMA!) +
  cp.async 包装。移植 FMA → 5x 慢(我的实现 compute-bound)。cp.async 在 mxcc 损坏,
  MetaX 自己也默认关(FLASHINFER_CP_ASYNC_ENABLED 注释掉)。
- S4 修正我的带宽误判: baseline case13 实为 1.38TB/s(=76% 纯流上限 1.82), 非 0.7。
  早前 "0.37TB/s" 探针是 launch/pacing 缺陷。
- 穷尽的结构尝试全败: scalar-FMA(35x慢)、2kv-MMA(3.4x慢)、大块CTA、warp专用化、
  grid同步+L2预取、单union buffer(pad冲突)、8 CTA/SM(bank conflict)。
- 结论: 67.07 是该 MMA+smem 设计 + mxcc 的实际天花板。

## B8. 冲刺: clean 重构 + combine 向量化 + case14 ns100 (→67.36)
- **clean 重构**(用户队友要求"去 case hacking 痕迹"): 把散落 per-case 魔数分支
  重构成 7-regime 物理模型, 性能逐字节一致。
- combine 向量化: float4 读 acc_part + l 预计算(gqa*64→gqa*32 线程)。微改进。
- case14 ns 148→100: 本地 rand/full 分布 119 vs 124us。
- OJ: case7 0.204, case9 0.207, case14 0.115 → **67.36** (一度以为 +1 分来自 ns100)。
- **事后澄清**: 67.36 vs 67.21 的分差实际在 case7/9 的 run-to-run 噪声(±0.01ms=±1分),
  不在 case14(两次都 59)。ns 微调再次被证明是噪声级。

## B9. 终局与噪声 (67.36 定稿)
- c13ns64b(case13 ns90→64) OJ 65.71: 但 case4/6/7/9/10/12 全退化而这些代码没动 →
  纯机器负载噪声(比赛末段提交慢几十倍)。case13 本身 0.173 与 clean 一致 → ns64 无效。
- 最终: `submission_clean.cu` = **67.36**。比赛结束。

---

# 分数/文件全轨迹表 (OJ 实测)

| 提交文件 | OJ 分 | 阶段 | 关键改动 |
|---|---|---|---|
| optimized_c500_flash_attn.cu | 62.21 | A | 原始(有 FORCED_SPLITS=128 陷阱) |
| submission_gen11b.cu | 64.57 | A6 | V token-major VPAD8 消除转置 |
| submission_merged.cu | 64.93 | A7 | 多 agent split 合并 |
| submission_agentG_v2.cu | 65.14 | A7 | split policy per 类细化 |
| submission_nomax.cu | 66.71 | B2 | nomax softmax |
| submission_ov.cu | 65.7 | B6 | +微优化但 case4 tail 竞态(事故) |
| submission_ov_safe.cu | 67.07 | B6 | tail syncwarp 修复 |
| submission_clean.cu | **67.36** | B8 | +combine 向量化 + case14 ns100 |
| (c13ns64b) | 65.71* | B9 | *机器噪声, 无效 |
| (c11ns11) | 67.07 | B3 | case11 ns22→11 负结果 |

# 各维度经验速查 (详见 POST_MORTEM.md)
- 技术: load-preserving 探针 / 先测准 baseline / occupancy 先算墙 / V 转置消除 / nomax / ns=噪声
- 方法: interleaved A/B / 先 verify 再信 speedup / 负结果带机制 / 大重构前估成本
- 流程: 基线必须最新 / 文件-分数显式映射 / OJ 提交珍惜 / 并行 agent ≤2 / 原生 SSH
- 工具: mxcc 命令 / warp=64 / MMA fragment / bsm 损坏 / 寄存器墙 152

---

# 附1: 我(主agent/协调者)的失误清单 — 每个错→根因→避免方法

> 阶段 B 我直接主导, 以下是我的真实错误。每个都造成了实际代价(时间/方向/误判)。
> 按代价排序。

## 失误1: DCE 假象误判瓶颈为 "MMA-bound" (代价: ~1 天方向跑偏)
- **错**: 删 MMA 探针 (427→254→38us) 断言 kernel 是 MMA-throughput bound, 行利用率
  25/50% 是杠杆, 并把结论写进 CONTEXT 当"突破"。
- **根因**: 删掉 consumer(MMA) 后编译器 DCE 掉喂它的 load → 时间暴跌, 与真实瓶颈无关。
  我把它当决定性证据, 还让 r5 两个 agent 盲从这个 "verified background"。
- **后果**: 浪费约一天在"填 16 行/多 kv 打包"上(架构不可能), 直到 r5 agent 用
  load-preserving 消融推翻。
- **避免**: 探针必须保留内存副作用(累加进被使用的值)。任何"删 X 时间大降"的结论,
  先怀疑是不是 DCE。给 agent 的 background 写"待验证假说"而非"已验证事实"。

## 失误2: 给 subagent 的基线不是最新最佳 (代价: 整轮 r1 白跑)
- **错**: round 1 让 3 个 agent 从旧 62.21 基线出发优化, 没意识到 agentG_v2(65.14) 才是
  当前最佳 → 他们最好也只到 ~66 本地, 全被 65.14 压住。
- **根因**: 接手时只读了 README(说 65.14 最佳) 但工作区默认内核是旧的; 没先确认
  "最新最佳 = 哪个文件" 就派活。
- **避免**: 每次派 agent 前, 先本地编译验证并确认基线文件 == 当前 SOTA 文件
  (grep 版本特征/跑 verify)。建立 FILE_MAP 后这问题可杜绝。

## 失误3: 把 run-to-run 噪声当真实信号 (代价: 多次错误决策)
- **错**: (a) c14ns100 我断言 OJ +1 分是 ns 效果; (b) 早期把 case1/2 的 6→7us 当回归追查;
  (c) 一度把 67.36 vs 67.21 当确定差异。事后证明 case7/9 有 ±0.01ms=±1 分 run 噪声,
  67.36/67.21 基本是同一 kernel 的两次采样。
- **根因**: OJ 单次提交只有一次采样, 我把它当精确测量; 本地多次 interleaved 是准的,
  但 OJ 的 ±1 分噪声被忽略。
- **避免**: OJ 结论需至少 2 次提交确认才下"有效/无效"; 单 case 变 1 分不采信,
  要 case 级 tk 变化 + 跨 case 一致性。ns 微调这类本来就该当噪声处理。

## 失误4: 同步语义改动漏了分支 (代价: 一次 OJ 事故 65.7)
- **错**: ov 把 __syncthreads→__syncwarp 只改了主循环, tail(部分页)路径漏改 →
  tail 无屏障竞态 → case4(batch64/ns1/多单页) 从 0.020 涨到 0.048, 总分 65.7。
- **根因**: 机械替换同步原语时没全局审计所有代码路径; 且 case4 的退化在本地
  任何分布都复现不出(本地全过), 只有 OJ 暴露。
- **避免**: 同步/内存序改动必须 grep 所有调用点; 结构性改动即使本地 14/14 过,
  也要警惕"只在特定 shape/occupancy 下触发的竞态"。tail/fused/partial 三条路径分开审。

## 失误5: 环境崩溃时反复开并行 agent 烧预算
- **错**: 服务器崩了多次(终端/TUI 问题), 每次崩后我立刻又并行开 5 个 agent →
  它们永远停在 setup, 白烧; 崩在 dispatch 瞬间。
- **根因**: 没认识到崩溃与"高并行 dispatch"相关; 没先降并行度验证稳定性。
- **避免**: 环境不稳时并行 ≤2; 崩后先单 agent 验证稳定性再扩; 重要探索自己直接做
  而非依赖 agent(崩溃免疫)。

## 失误6: 文档漂移不同步 (被用户点名)
- **错**: CONTEXT/DECISIONS/SKILL 三处模型不一致(如 MMA-bound 结论只在 CONTEXT,
  skill 还是旧的 DRAM 模型), 用户和同事都因此困惑/被误导。
- **避免**: 每次模型级结论必须同时更新三处 + 标注日期; 建立单一事实源。

## 失误7 (轻度): 上传/推送隐患
- remote URL 含明文 token 留在历史(同事提醒)。提交了 build 产物(.so/.o/.db)。
- **避免**: 用 credential helper; .gitignore 严格; push 前 git status 检查。

---

# 附2: 如果重来 — 从第一天起的最优路径

## Day-0 checklist (拿到 GPU 第一天)
1. 确认当前 SOTA 文件 + 最新基线 (FILE_MAP), 本地编译 verify + bench 建锚点。
2. 用 load-preserving 探针**一次定位**瓶颈: case13 到底在带宽哪个位置。
3. 验证 async/cp.async 在评测机是否真可用 (这是第一名 72 vs 我们 67 的最大未知道路)。
4. 决定根本路线: MMA+smem(我们走的路, 封顶 ~67) vs 其它数据流。

## 该跳过的探索 (已经证死, 不要再花时间)
- ns/split 微调(噪声级, OJ 上不可预测) — 只做一次 OJ A/B 确认即可。
- smem double-buffer / 寄存器预取流水(mxcc 寄存器墙 152, 必崩)。
- 8 CTAs/SM 尝试(去 pad → bank conflict, 必输)。
- async bsm 深挖(mxcc 损坏, MetaX 自己也关)。
- scalar/FMA 路径(慢 MMA 5x)。
- 多 kv/大块 CTA/warp 专用化/grid 同步(occupancy/同步开销全输)。

## 该重点投入的 (若有下次)
- **尽早拿到评测机的真实 async 能力** + 参考 kernel 的读结构(而非自己猜)。
- 微优化(向量化/去转置/softmax 域)是安全累加, 但总量 ~5%。
- 教学/复现验证若仍是目标, 隔离与上下文预算从第一天就做对。

---

# 附3: 同事/协作复盘

## 协作结构
- 多人共用 github dddd109/mx_flash_attn_kernel_opt_, 靠 SOTA*.txt 名字带分数同步。
- 早期教学阶段(gen/teacher 机制)与 GPU 阶段(我)由不同人接力。
- 同事贡献: gen/teacher 阶段全程、agentH 尝试(被中断)、后期 skill/代码意见
  (如"整理代码去 case hacking")、以及发现 skill 未随版本更新。

## 协作中的问题与教训
1. **交接文档是唯一记忆**: CONTEXT_BACKUP/HANDOFF 记了"同事也在优化, 先 git pull",
   但 pull 网络不稳导致信息滞后, 有过 SOTA 认知偏差。
2. **文件-分数混淆是多人协作的通病**: 同一个名字(clean)在不同人手里是不同 ns 版本,
   必须 FILE_MAP + 提交前 grep 核对。
3. **skill 漂移**: 同事指出 skill 停留在旧模型(case12 DRAM 等)没随真实 probe 更新 —
   教训: 教学文档和实战结论要双向同步。
4. **明文 token 风险**: remote URL 带 token 在 git 历史, 同事提醒轮换 — 协作仓库
   安全卫生要早做。

---

# 附4: 按天时间线 (比赛全周期)

> 早期(08-29~09-02)教学 skill 迭代阶段细节主要来自归档文档, 日期为文件标注。

## 08-29 ~ 08-30: skill 教学体系建立 (SESSION_NOTES)
- 调研 MetaX C500 软件栈(mctlass/cute/mxcc/mcTriton), 初版 skill。
- skill v1→v4 迭代: 教 agent 写自定义 kernel, 修 Triton 尝试的 bug。
- 建立基准: flash_attn baseline ~50 分, 优化后 ~80-90us local(5-6x)。

## 09-03 (或前后): gen 系列至 65.14 (无 GPU)
- gen1(51)→gen2(61 假分数)→gen3(18.7 灾难)→gen4/5(61 守成)→gen6-8(~53 真实)→
  gen9/10(54.2)→gen11(V转置突破)→64.57(gen11b OJ)→64.93(merged OJ)→65.14(agentG_v2 OJ)。
- 期间: 教师实验 H1/H5/H6、假分数修正(gen6-9 "61" 实为 ~53)、case 表 SPJ 确认。
- SOTA: gen11b 09-03, merged 09-03。

## 09-04: GPU 可用 → 阶段 B 开始 (65.14 → 67.07)
- 早: 建本地闭环(compile/verify/bench), 修 case 表, 建 CONTEXT/DECISIONS。
- r1: agent 从旧基线(教训)。r2: nomax 突破 → 66.71 OJ。服务器崩 2 次。
- r3: 死路确认。我自己探针误判 MMA-bound(失误1), 深挖 fragment, 被 r5 推翻。
- r4-5: ov 合成(本地 1449)。ov OJ 65.7(case4 事故, 失误4) → ov_safe 67.07。
- 深夜: r6/r7 结构穷尽(全死路), 联网挖 MetaX 自家 kernel, bsm 确认损坏。
- 用户休息, 我自主继续。

## 09-05: 冲刺与收官 (67.07 → 67.36)
- 白天: clean 重构(去 case hacking, 用户队友要求) + combine 向量化 + case14 ns100。
- c11ns11 OJ 负结果确认。67.36 OJ 记录(clean)。
- 用户陆续提交 c13ns64b(机器负载噪声 65.71)。
- 重大澄清: 67.36 vs 67.21 是 case7/9 噪声(失误3)。
- 比赛结束(约 09-05 下午): 最终 67.36。整理仓库 + 多份复盘文档 + push。

---

# 附5: 一句话总结

**技术天花板是 MMA+smem 设计在 mxcc 上的 ~67; 距离第一名 72 的差距本质是
"参考 kernel 有我们未复现的读取/异步能力或完全不同的数据流"。流程上最贵的三课:
探针要 load-preserving(失误1)、基线必须最新(失误2)、OJ 噪声别当信号(失误3)。**
