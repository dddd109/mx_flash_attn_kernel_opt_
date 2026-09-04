# WORKFLOW - 无 GPU 环境下的详细操作流程

> 适用：没有本地 GPU（无法跑 profiler/benchmark），只能靠 xpuoj 提交验证。
> 核心循环：**改代码 → 编译验证 → 你手动提交 → 读 14 case tk/tb → 分析 → 再改**

---

## 角色分工（重要）
- **用户/同事**：负责在 xpuoj 网页手动提交（Cloudflare Turnstile 拦自动化，无法绕过）。一次提交给一个 .cu 文件。
- **Agent（你）**：负责分析 OJ 返回的 tk/tb、决定改什么、产出新 .cu、编译验证。
- 提交返回的是 **14 个 case 各自的 (tk_time_ms, tb_time_ms, speedup, score)** —— 这就是无 GPU 时最准的 profile。

---

## 一、一次优化循环（从 OJ 结果到新提交）

### Step 0: 起点
当前最佳：`submission_agentG_v2.cu`（65.14）。下一版从它复制改名，如 `submission_vN.cu`。

### Step 1: 用户提交当前版本，把 14 case 的 tk/tb/score 发来
保存到 `SOTA:submission_XXX_cu_YY.YY.txt`（YY.YY=分数）。

### Step 2: Agent 分析 OJ 数据（重点！）
对每 case 算 `speedup = tb/tk`，找 **speedup 最低 / score 最低** 的 case：
- 这些是提升空间最大的（score 公式单调依赖 speedup）
- 参考历史分数：62.21(原版)→64.57→64.93→65.14。第一名 72.71。

**关键判断：区分"可调"与"瓶颈"**
- 若某 case 的 tk 已接近 tb（speedup~1.0x）但 tb 本身大 → 该 case 是瓶颈，重点看
- 对比**本次 vs 上次**同一 case 的 tk 变化：改了 split 应看到 tk 变（本次 65.14 的教训：本地 uniform 预测的改善在 OJ 上无效，所以**只信 OJ tk 对比**）

### Step 3: Agent 决定改什么
无 GPU 时**只做低风险改动**（改错了能编译但不影响正确性）：
- **split 数/策略**：改 host 端 run_kernel 里 ns 计算（最常见有效杠杆）
- 每 (batch, kv, pages/work_units) 类的 ns 常量
- 不要做需要 GPU 验证正确性的结构改动（2 kv_head/block 之类）——除非你有信心。

改法参考 `agentG_v2.cu` 的 split 决策块（注释标了每类的测量最优值）。

### Step 4: 编译验证（无 GPU 也能做）
```
/opt/maca/mxgpu_llvm/bin/mxcc -std=c++17 -shared -fPIC submission_vN.cu -o /tmp/vN.so -I/opt/maca/include -I/opt/maca/tools/cu-bridge/include
```
- 编译过 = 语法对（无 GPU 也能确认）
- **无法本地验证正确性/性能** —— 所以改动要保守

### Step 5: 交给用户提交
告诉用户：提交 `submission_vN.cu`，预期变化（哪些 case 该变快多少）。用户提交后把新 14 case 发回 → 回到 Step 1。

---

## 二、OJ 数据分析方法（无 profiler 的替代）

### 判断 case 是否带宽饱和（能否再快）
某 case 若 speedup 低（~1.0-1.2x）可能是：
1. **kernel 已到带宽极限**（参考 ~1.3-1.5TB/s 实测上限）→ 难提升，别浪费
2. **split/并行不足** → 可调 ns 改善
3. **launch/延迟受限**（短 case）→ 接近 launch floor 就别动

区分方法：看该 case 的 tk 随你改 split 是否变化。变 = 可调；不变 = 其他瓶颈。

### 跨版本对比
维护一张表：每 case 的 tk（gen11b=64.57, merged=64.93, agentG_v2=65.14 各有）。tk 下降的改动保留，上升的回退。

---

## 三、当前待办/方向（按上次分析）
1. agentG_v2 = 65.14。它相对 merged 的改动（case8 ns22, case9 ns11）在 OJ 上：
   - case8: 81→79 ✓ 有效
   - case9: 240→237 ✓ 有效
   - case11: 198→203 ✗ 反而慢了（ns 从? 改成 22 在 case11 上过细）
   - **下一步**：case11 单独调回更粗的 ns（它是 kv4 batch16 pages766，22 splits 每块 35 页可能太多/太少，试中间值）
2. 低分 case 优先：case7(54), case13(55), case11(55), case14(54), case9(56), case10(57), case8(58)
3. 第一名 72.71 —— 我们 65.14 还差 7.6 分，主要在 54-59 分的中大 case

---

## 四、提交注意事项
- xpuoj 自动提交被 Cloudflare Turnstile 挡（`Captcha acquisition failed`）→ 必须人工
- 每次提交前 `git pull` 看同事是否更新了 SOTA
- 提交后把结果存 SOTA 文件并 commit + push（同事可看到）
- **勿提交**：xpuoj_state.json（已 gitignore）、任何 API key

---

## 五、如果 GPU 回来了
- 本地 benchmark 用 `bench_cuda_event.py`（真实 case 表已内置）
- **警告**：本地 uniform cache_seqlens 分布 ≠ OJ 分布（65.14 教训：本地预测 case11 改善 15us，OJ 反而慢 5us）。本地只能做相对粗调，最终以 OJ 为准
- 正确性验证：`python3 full_verify.py <lib.so>`（14 case vs flash_attn）

---

## 六、常用命令速查
```bash
# 编译
/opt/maca/mxgpu_llvm/bin/mxcc -std=c++17 -shared -fPIC in.cu -o out.so -I/opt/maca/include -I/opt/maca/tools/cu-bridge/include

# （有GPU时）测单 case
python3 bench_cuda_event.py opt <case> <lib.so> <reps>

# （有GPU时）全量正确性
python3 full_verify.py <lib.so>

# git 同步
git pull origin main && git add -A && git commit -m "..." && git push origin main
```
