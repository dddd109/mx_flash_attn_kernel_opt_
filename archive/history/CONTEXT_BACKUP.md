# 完整上下文交接 (2026-09-04, GPU 已不可用)

## 任务
MetaX C500 上 FlashAttention paged GQA decode (xpuoj contest 11 problem 1)。
通过教学 skill + 多 subagent 从 flash_attn baseline 优化逼近第一名 72.71。

## 分数里程碑 (OJ 实测)
- submission_gen11b.cu = 64.57
- submission_merged.cu = 64.93  ← 曾是最佳
- agentG_v2.cu (G-work) = 本地进一步改善 case8/11/9，未提交 OJ 验证，预估 66-68
- 第一名 = 72.71
- 原版 optimized_c500_flash_attn.cu = 62.21

## 关键文件
- submission_merged.cu / libmerged.so: 64.93 提交版 (A+B+C 合并)
- agentG_v2.cu / libagentG_v2.so: 最新 (case8 82->71, case11 179->164, case9 ns11) - 待 OJ 验证
- submission_gen11b.cu: 64.57 版
- SOTA:*.txt: 各版本 OJ 记录
- OJ_BASELINE.md: 每 case OJ tk/tb/speedup/score
- .opencode/skills/paged-gqa-optimizer/SKILL.md: 教学 skill (含 64.57 config + 11 经验)

## 核心技术 (gen11 突破 + 后续)
1. V token-major VPAD=8 消除转置 (最大赢, score 54->74 local)
2. merged-max softmax 无 beta
3. split 策略按 (batch,kv,work_units) 类调优:
   - case13(batch1,kv8,长): ns=90; case14(batch1,kv4,长): ns=148-160
   - kv4 batch16 (case8/11): ns=22 (G 发现)
   - kv8 wu512(case7): mult=10(ns~5); wu256(case9): ns=11; wu64(case12): mult=10.5(ns~60)
   - case6(kv8 batch16 pages23): ns=5
4. 裸 __builtin_mxc_mma_16x16x16bf16, 不需 mctlass
5. 64 线程/block, grid=(ns, batch*nkv)

## 真实 OJ case 表 (SPJ 确认)
case: (batch, cap, kv)
1:(1,1,4) 2:(4,2,8) 3:(16,17,4) 4:(64,64,8) 5:(16,141,4) 6:(16,362,8)
7:(64,2048,8) 8:(16,4096,4) 9:(32,4096,8) 10:(1,8192,4) 11:(16,12251,4)
12:(8,32768,8) 13:(1,58966,8) 14:(1,61519,4)
edge: 1-3. iters: 1-3=100,4-6=50,7=12,8=25,9=12,10=25,11=12,12=12,13-14=25

## OJ 评分洞察
- score_ratio 由 speedup 决定 (case4 sp2.86 ratio0.74; case6 sp1.85 ratio0.645)
- 本地 tk 与 OJ tk 一致; OJ baseline 比本地 flash 慢 (cache_seqlens 更满)
- 本地 uniform 分布低估 kv4 长 case 工作; OJ 更 skew_long
- 低分 case 提升 speedup 直接涨分

## 带宽分析 (本地 uniform)
已饱和 (~1.2-1.26TB/s): case 7/9/12/13
未饱和 (<1TB/s, 有空间): case 6(0.41) 8(0.82) 11(1.12) 14(0.87)
=> kv4 大 case 是机会 (case 8/11/14), 但 OJ 上可能不同

## 尝试过且失败/无收益
- 软件流水 (寄存器/双缓冲) - 寄存器溢出/occupancy 降
- V 转置消除 (无 VPAD 版) - bank conflict 慢 60%
- 2 kv_head/block 结构 - 未完成 (agentH 被中断)
- Swizzle<3,3,3> K 布局 - 未在当前结构验证
- MMA M 行跨 kv 打包 - 结构不可能 (共享 B 操作数)

## 待尝试方向 (无 GPU 时用 xpuoj profile)
1. 提交 agentG_v2 验证分数 (预估 66-68)
2. kv=4 时 2 kv_head/block 填满 16 MMA 行 (case8/11, agentH 任务)
3. case 6/8/14 带宽未饱和优化

## 测量工具
- bench_cuda_event.py opt/flash <case> <lib> <reps> (真实 case 表)
- full_verify.py <lib> (14 case 正确性)
- 用 torch.cuda.Event, 勿用 torch.profiler (C500 低估 400x)

## 协作
- 远端: github dddd109/mx_flash_attn_kernel_opt_ (同事 branch 开发)
- SOTA:*.txt 记录双方最优, 名字带分数
- 注意 remote URL 含明文 token (安全风险, 建议同事轮换)
