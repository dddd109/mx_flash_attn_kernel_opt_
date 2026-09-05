# Gen2 concrete experiments (dispatch to subagents)

Context: v66.71, MMA-throughput bound (probes: 85-90% MMA), structural limits decoded.
Cases 4-14 need ~1.25x for 72.7. Big cases verified local==OJ (reliable bench).

## Candidate experiments (each = one subagent)
E1. CTA shape / occupancy vs MMA latency (is 8 blocks/SM enough to hide MMA latency?
    Test 32-thread or 128-thread blocks; test smaller smem tile; test 2 warps/block.)
E2. MMA operand setup reduction: the QK loop reads K_b smem + builds bv per st; the
    PV loop gathers V per st. Count non-MMA instructions in the inner loop; try
    preloading all 8 st operands, or wider smem reads, to cut issue slots between MMAs.
E3. Investigate if a 16x16 MMA can be issued with k=32 worth of useful work by holding
    A/B across 2 MMAs without reload (accumulate c) — i.e., is the 2nd MMA free of
    operand-reload stalls? (may already be true)
E4. Two-kernel split for kv8: since gqa=4 only 4 heads, is a NON-MMA (scalar FMA)
    QK for 4 heads faster than the 4/16-useful MMA? Probably not, but the PV might
    benefit from scalar when only 4 p-rows needed.
E5. Re-examine the ns/policy under the MMA-bound model: with MMA-issue the limit,
    optimal splits = fill SMs with exactly the right # CTAs so no MMA pipe idles.
    Model the SM count (104) x MMA issue rate and confirm ns policy matches.
