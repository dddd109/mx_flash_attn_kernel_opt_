# 提交优先级（2026-09-05）

## 当前最佳 / 保底
1. **submission_ov_safe.cu** — OJ 实测 67.07（case4 修复版），最稳
2. **submission_clean.cu** — ov_safe 的干净重构版 + combine 向量化（本地略优），理论 ≥67.07

## OJ 实验（一次一个，看变化）
3. **submission_var_c14ns100.cu** — case14 ns 148→100。本地 case14 123→118us；
   rand/full 分布支持，half 分布退化。若 OJ case14 tk 从 120 下降则有效。

## 不建议（已验证负效果）
- submission_c11ns11.cu（case11 ns 实验，OJ 变慢）
- submission_contig.cu / kv2_mma.cu / fma_decode.cu（结构重写，全部远慢于 MMA）

## 2026-09-05 半小时冲刺更新
- 新最佳 67.36 = submission_clean.cu（case14 ns 100 + combine 向量化），OJ 已确认。
- 正在跑：submission_clean.cu (67.36) 确认 / c13ns64b。
- 下一个实验（若 c13ns64b 无效）：c13ns64b 证明 OJ 偏好低 ns 的话，试 case10 ns=80（本地微好）。
- 关键洞察：case14 在 OJ 偏好 ns=100（低于本地 full 的 148）→ OJ 分布比本地 rand 更"短尾"。
  case13 ns 64≈90 本地平 → OJ 若偏好低 ns，64 会赢。
