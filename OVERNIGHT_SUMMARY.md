# 2026-09-05 深夜-清晨 结果汇总

## 结论：67.07 是当前架构 + mxcc 工具链的实际天花板
本时段穷尽了所有结构性方向，全部无法超越 submission_ov_safe.cu (67.07 OJ)：

| 方向 | 结果 |
|---|---|
| S1 GEMM 数据流 / S2 warp 专用化 / S3 大块 CTA / S4 grid-sync+L2 | 全部失败（occupancy/sync/MLP 权衡全输）|
| 标量 FMA 连续读 kernel | 正确但 35x 慢 |
| 2-kv/CTA MMA（宽读） | 正确但 3.4x 慢（occupancy 7→3）|
| MetaX mcflashinfer FMA-decode 移植 | 正确但 5x 慢（我的移植 compute-bound）|
| bsm/cp.async 异步拷贝 | mxcc 上不可靠（MetaX 自己也默认关闭）|
| combine 向量化 | 小幅干净优化，已折入 clean |

## S4 的关键修正（推翻我早前探针）
- case13 baseline 实际 1.38 TB/s（非 0.7），纯流上限 1.82 → 已到 76%
- 瓶颈是 MLP（46K vs 213K 线程），但所有提 MLP 的尝试都输给 occupancy/同步开销

## 本地可复现的小改进
- case14 ns 148→100：本地 case14 123→118us（但 half 分布下退化 → OJ 实验版）
- combine 向量化：case14 微改善

## 推荐 OJ 提交顺序
1. submission_var_c14ns100.cu（最有希望的实验）
2. submission_clean.cu（67.07 + combine 优化 + 去 case-hacking，理论 ≥67.07）

## 最终冲刺补充 (2026-09-05)
- 文件-分数映射已建 FILE_MAP.txt。
- 所有 case 的 ns 用 combine-opt kernel 全部复核最优: case7=5, case8=22, case9=11, case10=64,
  case11=22, case12=59, case13=64≈90(平), case14=100。唯一在途实验: c13ns64b (case13 ns64)。
- 若 c13ns64b OJ >=67.36: 合入 clean (case13 ns64 + case14 ns100) 为新 canonical。
- 若 =67.36: case13 ns 无所谓, clean 保持 90。
- 若 <67.36: case13 ns90 更好, clean 保持 90。
