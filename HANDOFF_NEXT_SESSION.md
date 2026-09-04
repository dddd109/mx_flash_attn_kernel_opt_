# 交接文档 (2026-09-04) - 下个 session 必读

## 当前处境
- 目标: MetaX C500 paged GQA decode (xpuoj contest 11 problem 1)
- **无本地 GPU** (比赛结束前可能都没有)。本地无法验证性能。
- 唯一验证手段: 提交 xpuoj 看每 case tk/tb/speedup (每 case 精确 profile!)
- 第一名 72.71。当前最佳 submission_agentG_v2.cu = 65.14 (OJ #138763)
- 同事可能也在优化 (git 协作, SOTA*.txt 同步)

## 分数历程 (全部 OJ 实测)
- 原版 optimized_c500_flash_attn.cu = 62.21 (用户初始提供)
- submission_gen11b.cu = 64.57
- submission_merged.cu = 64.93
- submission_agentG_v2.cu = 65.14  ← 当前最佳

## 核心技术 (agentG_v2)
1. V token-major VPAD=8 消除转置 (最大赢)
2. merged-max softmax (无 beta)
3. split 按 (batch,kv,wu) 类调优
4. 裸 __builtin_mxc_mma_16x16x16bf16, 64线程/block, grid=(ns, batch*nkv)

## 真实 OJ case 表 (SPJ 确认, 勿再用错!)
case: (batch, cap, kv)
1:(1,1,4) 2:(4,2,8) 3:(16,17,4) 4:(64,64,8) 5:(16,141,4) 6:(16,362,8)
7:(64,2048,8) 8:(16,4096,4) 9:(32,4096,8) 10:(1,8192,4) 11:(16,12251,4)
12:(8,32768,8) 13:(1,58966,8) 14:(1,61519,4)

## agentG_v2 OJ 每 case (tk_ms, score)
1:(0.007,84) 2:(0.007,84) 3:(0.009,83) 4:(0.021,74) 5:(0.016,74)
6:(0.027,64) 7:(0.238,54) 8:(0.079,58) 9:(0.237,57) 10:(0.048,57)
11:(0.203,55) 12:(0.397,59) 13:(0.194,55) 14:(0.142,54)

## 关键教训 (极其重要)
1. **本地 benchmark 的 cache_seqlens 分布假设 (uniform) 与 OJ 严重不符!**
   - 本地预测 case8 82->71, case11 179->164 (split ns=22 改善)
   - OJ 实测: case8 81->79 (小改善), case11 198->203 (反而+5us 退步!)
   - => 基于本地 uniform 的 split 微调在 OJ 上无效甚至反向
2. **split 微调在 OJ 是噪声级** (case8/9 微改善来自 ns 变化, 但 case11 退步)
3. OJ baseline (tb) 每次提交微波动 (case7 tb: 0.298/0.305/0.281)
4. tk 本地与 OJ 量级一致但分布敏感

## OJ 真实分布反推 (用 tk + ~1.35TB/s 校准)
大 case 推断的"平均填充率":
- case 7 (batch64): 59% | case 8: 79% | case 9: 60% | case 11: 67%
- case 12 (batch8): 50% | case 13 (batch1): ~108% (近全满) | case 14: ~152%(受限)
含义: 大 batch case 平均填充中等 (50-80%), batch=1 case 近全满
这不像 uniform! 更像: cs[0]=满, 其余 skew 到较长, 或混合固定长度

## 下一步方向 (按优先级)
1. **[进行中] playwright 自动提交** - 减少手动操作。chromium 已下载, install-deps 完成, 待验证启动
2. **针对 OJ 真实分布优化** - 不再用本地 uniform。方法: 每次提交的 tk 当 profile,
   反推该 case 实际负载, 针对性调 split (不是盲调)
3. **大头 case 结构优化** - OJ tk 最大: case12(397us,59分) case7(238,54) case9(237,57)
   case11(203,55) case13(194,55)。这些是大头, 提升它们最有效
4. 2 kv_head/block 填 16 MMA 行 (未完成, agentH 任务, 无 GPU 难验证正确性)

## 提交操作提示
- 每个提交返回完整 14 case tk/tb/speedup - 这是最准 profile
- 用户手动复制提交 (痛点在复制麻烦, 自动化进行中)
- 账号: muxi2026C2032@example.com / 密码在用户消息里

## 文件索引
- submission_agentG_v2.cu: 当前最佳 65.14
- agentG_v2.cu: 带 env override 的实验版 (本地更好但 OJ 无效)
- SOTA:*.txt: 各版 OJ 记录
- OJ_BASELINE.md: merged 的 OJ 数据
- bench_cuda_event.py: 本地测 (注意分布假设不准!)
- .opencode/skills/.../SKILL.md: 教学 skill

## 注意
- 勿再花大量时间做依赖本地 uniform 分布的 split 微调
- 无 GPU 验证的结构改动风险高 (正确性无法本地确认)
- 推进前先确认同事 SOTA (git pull)

## 自动化提交探索结论 (2026-09-04 更新)
- playwright + chromium 已装 (headless shell 可跑)
- 登录成功 (storage_state 保存于 xpuoj_state.json, 但该文件含敏感 cookie 勿提交)
- Monaco 代码注入可行 (window.monaco.editor.getModels().forEach(m=>m.setValue(code)))
- **提交被 Cloudflare Turnstile 人机验证拦截** (console: "Captcha acquisition failed",
  "Failed to initialize Cloudflare Turnstile")
- 真实 API 在 volceapi.com (apigateway-cn-beijing), 有 Cloudflare 防护
- => 纯自动提交不可行, 需人工过 CAPTCHA。手动提交仍是唯一方式
- 脚本: xpuoj_login.py / xpuoj_submit.py (登录可用, 提交被 captcha 挡)
