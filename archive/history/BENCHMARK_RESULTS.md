# Correct Benchmark Results (cuda_event timing, 2026-08-31)

## Important Finding
**torch.profiler severely underreports GPU kernel time on MetaX C500** (~400x off).
Use `torch.cuda.Event` timing or wall-clock instead.

## Data

| case | batch | seqlen | nkv | flash(us) | opt_agent(us) | ratio(opt/flash) |
|------|-------|--------|-----|-----------|---------------|------------------|
| 1 | 4 | 8 | 4 | 31.5 | 12.0 | 0.38x (faster) |
| 2 | 4 | 2 | 8 | 31.7 | 11.9 | 0.37x |
| 3 | 16 | 17 | 8 | 38.2 | 13.7 | 0.36x |
| 4 | 64 | 64 | 4 | 38.6 | 25.4 | 0.66x |
| 5 | 16 | 141 | 8 | 38.9 | 22.6 | 0.58x |
| 7 | 64 | 2048 | 4 | 162.0 | 148.9 | 0.92x |
| 8 | 16 | 4096 | 4 | 90.9 | 84.0 | 0.92x |
| 9 | 32 | 8 | 4 | 31.8 | 12.0 | 0.38x |
| 10 | 1 | 8192 | 4 | 57.4 | 64.1 | 1.12x (slower) |
| 11 | 16 | 12251 | 4 | 206.1 | 194.4 | 0.94x |
| 12 | 8 | 32768 | 4 | 320.7 | 298.1 | 0.93x |
| 13 | 1 | 58966 | 4 | 153.4 | 168.2 | 1.10x (slower) |

## Observations
- Short sequences (case 1-5, 9): agent kernel 0.36-0.66x (significantly faster)
- Long sequences with batch>1 (case 7,8,11,12): agent kernel 0.92-0.94x (slightly faster)
- batch=1 long sequences (case 10,13): agent kernel 1.10-1.12x (SLOWER than flash!)

## Scoring
- Baseline flash_attn = 50 points
- optimized target = 63.x points
- OJ formula: S=100/(1+((1/0.5)-1)*(Tk-Th)/(Tb-Th))

## Notes
- Agent copied optimized_c500_flash_attn.cu structure (violates constraint)
- Need to verify original optimized_c500_flash_attn.cu performance (compile separately)
- case 10/13 weak spots: batch=1 long seq - likely split-KV tuning needed
