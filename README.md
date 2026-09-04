# MX FlashAttention Kernel Optimization - 项目导航

## 📌 新 session 必读（按顺序，只需这 4 个）
1. **`WORKFLOW.md`** ← 详细操作流程（无 GPU 循环：改→编译→提交→分析）
2. **`HANDOFF_NEXT_SESSION.md`** ← 当前处境、分数、教训、下一步
3. **`task.md`** ← OJ 赛题描述（接口、case 表、评测）
4. **`OJ_BASELINE.md`** ← 各版本 OJ 实测分数对照

## 若要继续优化 kernel（读这些）
- `.opencode/skills/paged-gqa-optimizer/SKILL.md` ← 教学 skill（含已验证的 64 分配置 + 11 条经验）
- `submission_agentG_v2.cu` ← 当前最佳 kernel（65.14 分，可直接提交）
- `submission_merged.cu` ← 上一版（64.93）

## 历史文档（仅深入排查时才读，勿全读）
| 文件 | 内容 |
|------|------|
| `CONTEXT_BACKUP.md` | 完整技术细节 + case 表 + 评分 |
| `GEN11_BREAKTHROUGH.md` | V 转置消除（最大技术突破） |
| `GEN_STRUCTURAL_DIFF.md` | 参考 kernel vs student 结构差异分析 |
| `AGENT_FEEDBACK_INTEGRATION.md` | 历代 agent 经验汇总 |
| `CORRECTED_BENCHMARK.md` | 修正后 benchmark 数据 |
| `SOTA:*.txt` | 各版本 OJ 分数记录 |

## 分数里程碑
- 原版 optimized_c500_flash_attn.cu = **62.21**
- submission_gen11b.cu = 64.57
- submission_merged.cu = 64.93
- **submission_agentG_v2.cu = 65.14（当前最佳）**
- 第一名 = 72.71

## 协作须知
- 远端: github dddd109/mx_flash_attn_kernel_opt_（同事也在优化）
- SOTA 文件变化 = 同事有新进展，先 `git pull`
- 敏感文件勿提交: xpuoj_state.json, 各 API key
