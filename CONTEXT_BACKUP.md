# 会话关键信息 (防上下文溢出备份)

## 任务
FlashAttention paged GQA decode (xpuoj contest 11 problem 1, MetaX C500)。
目标: 通过教学 skill 让 agent 从 flash_attn baseline 优化，逼近/超过 optimized_c500_flash_attn.cu (62.21, 排18)。

## 当前最佳: submission_gen11b.cu = 64.57分 (OJ 实测)
关键文件: agent_gen11b.cu / libgen11b.so / submission_gen11b.cu

## 关键技术 (gen11b 达到 64.57)
1. V 消除转置: V token-major + VPAD=8 (stride 136 halves=68 words≡4mod32), vector store
2. PV 读: vgather2 = 2×uint16 LDS (V[t0][d] low, V[t0+1][d] high), 打包顺序必须与转置版一致
3. merged-max softmax (无 beta)
4. case13 修 split: batch1+kv8+长seq → ns=64 (原128)
5. 裸 __builtin_mxc_mma_16x16x16bf16 (不需 mctlass include, 用标准 mxcc 编译)

## 真实 OJ case 表 (SPJ 确认)
case: (batch, cap, kv)
1:(1,1,4) 2:(4,2,8) 3:(16,17,4) 4:(64,64,8) 5:(16,141,4) 6:(16,362,8)
7:(64,2048,8) 8:(16,4096,4) 9:(32,4096,8) 10:(1,8192,4) 11:(16,12251,4)
12:(8,32768,8) 13:(1,58966,8) 14:(1,61519,4)
edge: case 1-3. iters: 1-3=100,4-6=50,7=12,8=25,9=12,10=25,11=12,12=12,13-14=25

## 评分
S=100/(1+(Tk-Th)/(Tb-Th)), 本地 flash≈50, gen11b OJ 实测 64.57。
OJ baseline (tb) 比本地 flash 慢 (本地 cache_seqlens 偏轻)。
原始 optimized_c500_flash_attn.cu = 62.21 (用户提供)。

## 测量方法
torch.cuda.Event timing (profiler 在 C500 低估 ~400x 不可用)。
bench_cuda_event.py opt <case> <libpath> <reps> (真实 case 表)
full_verify.py <libpath> (14 case 正确性)
本地测的 gen11b case 13 = 220us 但 OJ = 222us 一致性好。

## skill 位置
.opencode/skills/paged-gqa-optimizer/SKILL.md (v20, 11条经验)

## 迭代历史
gen6(53.1) -> gen9(53.2, full/tail) -> gen10(54.2, merged-max) -> gen11(73.9本地/V转置消除) -> gen11b(OJ 64.57, case13 split修)
本地 73.9 vs OJ 64.57 差异 = cache_seqlens 分布假设不同 + kv case 表错配已修正。

## 备份
backup/: optimized_c500_flash_attn.cu (62.21原版), liboptimized_orig.so, gen2/9 kernel
agent_gen11_kernel.cu, libgen11.so (73.9本地版)

## 协作
远端 github dddd109/mx_flash_attn_kernel_opt_。SOTA txt 记录双方最优。
同事可能 branch 开发。注意 git pull 看新分支/SOTA 变化。
