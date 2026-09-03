# Gen9 Results - H1 applied, marginal gain, honest ceiling

## What was done
Gen9 applied teacher hint H1: split the page sweep into a steady-state loop over FULL
pages (no per-page ntok clamp, no -INF padding masks, straight __expf) plus a single
separate tail path for the one partial page per sequence. Body duplicated verbatim
(no lambdas) to keep register count at 152 (gen6=154).

## Results (best-of-N A/B)
- case 9: 256.9 -> 254.9
- case 11: 363.1 -> 361.1
- case 12: 574.0 -> 571.5
- Total: 1980 -> 1966us. Score ~53.2 (from 53.1).

## Honest finding (gen9's caveat, teacher agrees)
H1's premise - that per-page masking is hot-path waste - is only worth ~3us on 573us
because case 12 is DRAM-LATENCY-bound, not mask-bound. The masks are branch-predictable,
near-free ALU. The reference's historical H1 win (-65%) applied to a kernel whose
steady-state codegen was ALREADY latency-overlapped. gen6's serial
stage->barrier->compute chain needs software pipelining, but THIS mxcc build punishes
pipelining with register bloat (154->184+ regs, 400-1700us cost).

## Structural conclusion
Reaching 62 (ref 494us on case 12 = 1.09 TB/s vs gen6's 0.94 TB/s) requires either:
(a) a register-ring/software pipeline that this compiler doesn't penalize (not found),
(b) a different load mapping giving deeper per-thread MLP, or
(c) accepting that mxcc's codegen on gen6-style kernels caps ~53-54.

## Best student kernels preserved
- libgen6.so / agent_gen6_kernel.cu (~53.1)
- libgen9.so / agent_gen9_kernel.cu (~53.2, H1)
