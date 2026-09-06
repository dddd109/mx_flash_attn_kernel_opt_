# NOTES.md — agentD_async: 权威复查 async global->smem copy (bsm) on CURRENT toolchain

Date: 2026-09-06. Toolchain: mxcc 1.0.0 (d9102a1572), MACA 3.7.1.5, KMD driver 3.8.30,
xcore1000 (MetaX C500, 104 SM, smem 64KB/SM, torch 2.8.0+metax3.7.1.3). Workspace:
/tmp/agent_ws/agentD_async/.

## AUTHORITATIVE VERDICT

**cp.async/bsm 引擎在 *当前* 工具链上本身存在且能用，但对本 kernel 的 padded smem tile
结构上不可用，且即便能用也**快 1.7x**。方向 CLOSED。**

Two independent reasons, either alone kills it:

1. **正确性 (关键新发现):** bsm 引擎的 dst 写规则与 baseline 的 padded smem tile
   (K 264B / V 272B 行 pitch) 不兼容：
   - **每个 16-lane "quarter" 的 dst 基址被量化到 256B 边界**，无视真实 pitch。
     64-lane (或 32-lane) 单指令写 4 (或 2) 个 padded token 行时, 各 quarter 落到
     256B-quantized slot → V tile 行 1..3 被写偏 (数据整体前移 4 words) → 真实 kernel
     stage_page 形状 (async_sb: 每 (round) 64-lane 连续 instr) 全部错位 → garbage。
   - **只有 "leading 16 lanes" (0..15) 单行指令才在 padded tile 上正确** (数据字完全
     正确; 该行 4 个 pad word 会被引擎清零 — bsm 零填充 dst 区段尾部)。lane 16..63
     的非 prefix 子集 (如活跃 =16..31) 直接**什么都不写**。
   - 多 CTA 并存 (不同 smem 基址) 时即使 "正确" 形式也出现残留错误 (multi_cta probe:
     每 CTA 错误数随机 210..278/2176 word), 说明引擎对 smem 基址/并置还额外敏感。
   - 这精确解释了历史 "probes pass / real kernel garbage": 旧探针读 buf[lane] (dense,
     256B 对齐) 所以过; 真实 padded tile 走 64-lane 多行 instr → 全部 shift。

2. **性能:** 用唯一正确形式 (leading-16-lane 逐行, 每页 32 instr) 重建 async kernel
   (async_sb.cu/.so, stage_page 替换, 其余同 baseline):
   case13 176→310us, case14 119→196us (**B/A=1.66-1.76x 更慢**, interleaved A/B)。
   原因: 16-lane 每指令只发 1 行, 比 baseline 的 64-lane 每指令 4 行少 4x 并行,
   async "non-blocking" 的收益被 issue 串行化吃掉。

## 逐配置 readback 结果 (全部 clean full-width word-exact 比对)

- **dense 布局** (dst = lane*16B, leading-16/32/64 lanes 全部活跃, 每 256B 对齐):
  - wait = `__builtin_mxc_arrive(64+0)` + `__builtin_mxc_barrier_inst()` (mctlass
    arrive_gvmcnt(0)): **16/32/64 lanes x 1 load, 及 16x4 / 64x4 / 64x8 loads:
    0 mismatches over 4 reps**。多 CTA (8..104) x 多轮 (2..8) x 每轮 wait: 0 bad /
    up to 53248 words。→ dense bsm 在当前工具链**完全可靠**。
  - 其它 wait 编码: arrive_bsmcnt(0) / barrier_inst-only / barrier_warp-only → 数据未落
    地 (读回 poison) = 错误 wait。`gvmcnt(0)+bsmcnt(0)` 也过。**wait 必须含 gvm arrive**
    (cp.async 的 LDG 侧计数), 与 mctlass/flashinfer 用法一致。
- **V tile (16x136, 272B pitch)**: baseline 的 64-lane 4-row-per-round issuing →
  data shift (见上), 无论 wait 形式。leading-16 逐行: 数据字正确但 pad 被 0-fill;
  在真实 K+V 并置的 smem (V base 非 256B 对齐) 下, V rows 0-2 甚至全 0 (错位+清零)。
- **K tile (16x132, 264B pitch)**: 同 V; 奇数行 8B 对齐 (非 16B) 触发编译期
  "LDG_DEFAULT_PREDICATOR_BSM: globalOffset value and use Saddr conflict!" 警告 ×16
  (b64/b128 对齐冲突), 但 K 数据实际正确 (仅 pad 0)。改用 136 对齐 pitch 警告消失。

## Flag gate 测试

`strings mxcc` 里的 async pass (metaxgpu-process-memcpy-async / -async-bsm-write /
-insert-arrivecnts-forceasync / metaxgpu-async) 经 `-mllvm` 传递时:
process-memcpy-async 与 forceasync **不识别** (Unknown argument); 其余静默接受但
产物 .so 无行为/大小差异。**`__builtin_mxc_ldg_b128_bsm` 在默认 -O3 路径直接下译为
`llvm.mxc.ldg.predicator.bsm.v4i32` (async_sb.so 二进制内确认), 无需任何 flag。**
无宏 gate (MACA_CP_ASYNC_ACTIVATED=0 只关 mctlass 的 SASS cp.async 内联 asm, 无关 bsm
builtin)。

## 语法确认 (勿再考古)

- `__builtin_mxc_ldg_b128_bsm(dst_smem, src_global, 0, -1, true, true, false, true)`
  (后 4 参 = 0, -1, Ret0=false, ...; flashinfer/mctlass 均此形式; 见
  /opt/maca/include/mcflashinfer/cp_async.cuh load_128b_bsm 与
  mctlass/arch/maca_memory.h maca_cp_async_zfill<16>)。
- wait = `__builtin_mxc_arrive(64 + N)` (arrive_gvmcnt(N)) + `__builtin_mxc_barrier_inst()`。
  bsmcnt arrive = `4096 + 128*N` (arrive_bsmcnt)。N=0 等全部。
- 无 commit_group/wait_group 概念 (非 SASS cp.async); 全部经 gvm/bsm counter + barrier。
- intrinsic 本身 **无 macro gate、无 arch gate**、默认开启。

## 文件

- bsm_probe.cu/.so + bsm_driver.py (dense 16/32/64-lane、multi-load、V/K padded tile、
  wait-mode sweep; 读回覆盖实际写入区, 逐 word 比对, smem 预毒化)
- stride_probe / lanegroup_probe / mech_probe / vsub_probe / exact_stage_probe(.2) /
  multi_cta_probe / dense_multi_probe + 各自 run 脚本 (机制隔离)
- async_sb.cu/.so (真实 kernel 变体: stage_page→bsm, leading-16 逐行正确形式)
- baseline.so (本地参照, ALL-14 PASS, case13 176 / case14 118)
- ab.py (交错 A/B)

## 建议

主 session 接受 67.36 为本地最优。async 方向无 upside:
- 唯一让 padded tile 正确的 issuing (leading-16 逐行) 即引入 4x issue 串行化, 测出
  1.7x 变慢; 引擎设计 (256B-quarter 量化 + 只认 prefix lane + 多 CTA 不稳定) 排除了
  用 64-lane 多行 instr 直接灌 padded tile 的路径。
- 若未来仍想 async: 必须改成 dense 256B-aligned 中间 smem 再二次搬运 (adds smem traffic)
  或 smem 行 pitch 全部 256B (VSTRIDE 272→? 破坏 conflict-free read), 均为负收益。
- OJ 政策验证可并行推进。
