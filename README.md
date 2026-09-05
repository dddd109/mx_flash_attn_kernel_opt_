# MX FlashAttention Paged GQA Decode Kernel — 项目导航

## 📌 新 session 必读（按顺序）
1. **`DETAILED_POST_MORTEM.md`** ← 详细复盘：两阶段全程逐轮决策(A: 教学skill多代agent 62→65 / B: 有GPU闭环 65→67.36)
2. **`OPTIMIZATION_LOG.md`** ← 精炼优化日志：版本/分数/技术要点/教训
3. **`POST_MORTEM.md`** ← 可复用经验 + checklist
4. **`task.md`** ← OJ 赛题描述（接口、case 表、评测）
5. **`CONTEXT.md`** ← 当前处境、约束、开放问题（每步更新）
6. **`DECISIONS.md`** ← 决策日志（含负结果，防重复踩坑）

## 提交物
| 文件 | 分数(OJ) | 说明 |
|---|---|---|
| **`submission_clean.cu`** | **67.36** | **当前最佳（最终提交）**。case13 ns90 / case14 ns100 + combine 向量化 + 干净 regime split policy |
| `submission_ov_safe.cu` | 67.07 | 上一版（case4 修复 + pid预取/syncwarp/cvt_pk） |
| `optimized_c500_flash_attn.cu` | 62.21 | 原始参考（含 EXP25_FORCED_SPLITS=128 陷阱，勿用默认） |

## 目录结构
- `archive/agents/` — 历代 agent 内核 (gen6-gen11, merged, agentG_v2, nomax, ov 等)
- `archive/experiments/` — 实验内核（FMA/连续读/kv2/contig/c13ns64/c7ns8/c14ns120 等，均未超最佳）
- `archive/history/` — 历史文档 (GEN*_RESULTS/NOTE, WORKFLOW, SESSION_NOTES 等)
- `SOTA:*.txt` — 各版本 OJ 逐 case 记录
- `.opencode/skills/paged-gqa-optimizer/SKILL.md` — 教学 skill
- `bench_all.py` / `verify_real.py` — 本地全量计时 / 14-case 正确性（SPJ-confirmed case 表）

## 分数里程碑
- 62.21 → 64.57 → 64.93 → 65.14 → 66.71 → 67.07 → **67.36 (submission_clean.cu)**
- 第一名 = 72.71（结构上需新洞察，未达到）

## 协作须知
- 远端: github dddd109/mx_flash_attn_kernel_opt_（同事也在优化）
- SOTA 文件变化 = 同事有新进展，先 `git pull`
- 敏感文件勿提交: xpuoj_state.json, 各 API key
