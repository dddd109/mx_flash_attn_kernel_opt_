# 深夜自主优化总结 (2026-09-04 夜)

## 结果
- **最高分保持 67.07** = submission_ov_safe.cu (case4 修复成功 75分/19us; case12 达 360us 最佳; 大 case 全面改善)
- 未找到能本地验证超过它的新 kernel。

## 本夜完成的事
1. case4 OJ 回退(20->48us)根因定位并修复: ov 的 tail(部分页)路径漏了 __syncwarp
   (nomax 有 __syncthreads)。case4=batch64/ns1/多单页 block 最吃 tail。→ ov_safe, OJ 67.07。
2. case1/2 OJ "6->7us 回退" 经 interleaved A/B 证实是噪声 (本地 ov_safe 更快)。
3. 决定性验证 async bsm/cp.async 路径在 mxcc 上是坏的 (数据错位; 之前 agent 的 "probe pass"
   是读错地址(buf[lane] vs buf[lane*4])的验证假象)。→ 该杠杆关闭。
4. occupancy: smem 8576B 锁 7 CTAs/SM; 砍 pad 到 8 CTA 会 bank-conflict (-35~41%)。死路。
5. split policy 用 local/half/full 三种分布压力测试: case 7/9/12 都已最优; **case 11 例外**。
6. case11 (b16 kv4 12251) 对分布极敏感: local-rand 最优 ns=22(149us), 但 OJ 分布更长,
   OJ=187us 介于 local-rand(149) 和 half(255) 之间 → ns=11 在长分布下快 ~9%。
   → 产出实验版 submission_c11ns11.cu (case11 ns 22->11), **需 OJ A/B 定夺** (本地会显慢)。

## 待用户做的下一个 OJ 实验
提交 **submission_c11ns11.cu**, 看 case11 tk 是否从 187us 下降。若下降, 合入主 kernel。
注意: 它只改了 case11 (b16 kv4 pages>=400) 的 ns 22->11, 其余与 ov_safe 相同。

## 记录位置
- SOTA:submission_ov_safe_cu_67.07.txt (当前最佳)
- submission_c11ns11.cu (待 A/B 实验)
- FAILED_OPTIMIZATIONS.md (bsm 坏 / occupancy 死路 / 分布敏感性)
- GEN2_PLAN.md (方向 + 分布测试结论)
