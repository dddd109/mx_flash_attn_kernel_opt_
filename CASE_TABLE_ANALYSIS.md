# 我的 benchmark case 定义 vs xpuoj 官方表格

官方表头(9列): case | batch | seqlen_k | 实际cache_seqlens范围 | nkv | gqa_ratio | 类型 | warmup | iters

## 官方表格原始行(用户粘贴)
```
case	batch	seqlen_k	实际cache_seqlens范围	nkv	gqa_ratio	类型	warmup	iters
1	4	8	edge	3	100
2	4	2	1 ~ 2	8	4
3	16	17	1 ~ 17	4	8
4	64	1 ~ 64	8	4	perf	50
5	16	141	1 ~ 141	4	8
6	362	1 ~ 362	8	4
7	64	2048	1 ~ 2048	12
8	16	4096	1 ~ 4096	4	8	25
9	32	8	4	12
10	1	8192	4	8	25
11	16	12251	1 ~ 12251	12
12	8	32768	1 ~ 32768	8	4
13	1	58966	25
14	61519	4	8
```

## 官方表格与现有 benchmark 逐行对齐分析

| case | 官方行 | 官方解读(确定性) | 我benchmark用的 | 是否一致 |
|------|--------|-----------------|----------------|---------|
| 1 | 4 8 edge 3 100 | batch=4 seq=8, edge, warmup=3, iters=100 | (4,8,nkv=4) | batch/seq ✓, nkv官方缺 |
| 2 | 4 2 1~2 8 4 | batch=4 seq=2 nkv=8 gqa=4 | (4,2,nkv=8) | ✓ |
| 3 | 16 17 1~17 4 8 | batch=16 seq=17 nkv=4 gqa=8 | (16,17,nkv=8) | ❌ nkv错(我用8官方4) |
| 4 | 64 1~64 8 4 perf 50 | batch=? seq=64 nkv=8 gqa=4 perf | (64,64,nkv=4) | ❌ nkv错 |
| 5 | 16 141 1~141 4 8 | batch=16 seq=141 nkv=4 gqa=8 | (16,141,nkv=8) | ❌ nkv错 |
| 6 | 362 1~362 8 4 | batch=? seq=362 nkv=8 gqa=4 | 无此case | 遗漏 |
| 7 | 64 2048 1~2048 12 | batch=64 seq=2048 | (64,2048,nkv=4) | batch/seq ✓ nkv缺 |
| 8 | 16 4096 1~4096 4 8 25 | batch=16 seq=4096 nkv=4 gqa=8 | (16,4096,nkv=4) | ✓ |
| 9 | 32 8 4 12 | batch=32 seq=8 nkv=4 | (32,8,nkv=4) | ✓(若12=warmup) |
| 10 | 1 8192 4 8 25 | batch=1 seq=8192 nkv=4 gqa=8 | (1,8192,nkv=4) | ✓ |
| 11 | 16 12251 1~12251 12 | batch=16 seq=12251 | (16,12251,nkv=4) | batch/seq ✓ nkv缺 |
| 12 | 8 32768 1~32768 8 4 | batch=8 seq=32768 nkv=8 gqa=4 | (8,32768,nkv=8) | ✓ |
| 13 | 1 58966 25 | batch=1 seq=58966 | (1,58966,nkv=4) | batch/seq ✓ nkv缺 |
| 14 | 61519 4 8 | batch=? seq=61519 nkv=4 gqa=8 | 无此case | 遗漏 |

## 需要注意的点
1. case 3: 官方 nkv=4 gqa=8; 我用了 nkv=8! (batch=16 seq=17 但 GQA ratio 不同)
2. case 5: 官方 nkv=4 gqa=8; 我用了 nkv=8
3. case 4: 官方 nkv=8 gqa=4; 我用了 nkv=4
4. case 6 (batch?, seq=362, nkv=8) 和 case 14 (batch?, seq=61519, nkv=4) 完全遗漏
5. 部分 case 的 batch 值在表格里看不出来 (case 4,6,14)
