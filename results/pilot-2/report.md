# adapter_bits report: pilot-2

## Runs

W = trainable adapter parameters, M = items (random label bits). I = M(1 - H2(err)) is the Fano estimate of stored label information. Knee = fewest bits per weight keeping error within 1 point of full precision (on the gradient sample for the predicted columns).

| adapter | size | lr | W | M | M/W | err | I/W (bits/param) | knee | knee, typical-noise model | knee, worst-case (l-inf K) | median worst-case bits/item | K_inf | K_2 | tangent rank (1e-2) init / trained | top Hessian eig | counting bound held |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe | 128 | 0.05 | 128 | 128 | 1.0 | 0.21875 | 0.2421 | 4 (sample 4) | 8 | 16 | 9.62 | 329.1398 | 35.9215 | None / None | 185.91745 | yes |
| probe | 128 | 0.05 | 128 | 256 | 2.0 | 0.30469 | 0.2261 | 7 (sample 7) | 8 | 16 | 10.45 | 344.5617 | 37.366 | None / None | 222.79016 | yes |
| lora:gate | 1 | 0.003 | 4096 | 2048 | 0.5 | 0.43457 | 0.0062 | 6 (sample 10) | 7 | None | 11.03 | 345.2538 | 16.2519 | None / None | 49.24608 | yes |
| lora:gate | 1 | 0.01 | 4096 | 2048 | 0.5 | 0.41357 | 0.0108 | 8 (sample 8) | 10 | None | 11.71 | 440.7298 | 28.7624 | None / None | 113.7477 | yes |
| lora:gate | 1 | 0.03 | 4096 | 2048 | 0.5 | 0.39746 | 0.0153 | 7 (sample 8) | 10 | None | 12.0 | 453.1068 | 26.9127 | None / None | 56.7259 | yes |

## Information per parameter after rounding

Each cell is I_b / W in bits per parameter. The counting bound caps it at b (plus 32 bits per tensor for the scale).

| adapter | size | lr | M | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 | b=8 | b=10 | b=12 | b=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe | 128 | 0.05 | 128 | 0.0 | 0.0 | 0.0 | 0.228 | 0.03 | 0.1131 | 0.2144 | 0.2144 | 0.2421 | 0.2421 | 0.2421 |
| probe | 128 | 0.05 | 256 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0286 | 0.265 | 0.1433 | 0.2752 | 0.2355 | 0.2261 |
| lora:gate | 1 | 0.003 | 2048 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0053 | 0.0057 | 0.0062 | 0.0062 | 0.0059 | 0.0061 |
| lora:gate | 1 | 0.01 | 2048 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0015 | 0.0004 | 0.007 | 0.01 | 0.011 | 0.011 | 0.0108 |
| lora:gate | 1 | 0.03 | 2048 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0077 | 0.0132 | 0.0154 | 0.0147 | 0.0154 | 0.0153 |

## Stored bits against the gradient matrix's rank

Cover's theorem caps a linearized adapter near twice the rank of its items-by-parameters gradient matrix. Rank is counted at two relative thresholds on a sample of items, so it cannot exceed the sample size.

| adapter | size | lr | M | I (bits) | sample rows | 2 x rank at 1e-2 | 2 x rank at 1e-3 | participation ratio |
|---|---|---|---|---|---|---|---|---|

## Capacity scaling

- probe (lr 0.05): best stored bits by W = 128: 31
- lora:gate (lr 0.003): best stored bits by W = 4096: 25
- lora:gate (lr 0.01): best stored bits by W = 4096: 44
- lora:gate (lr 0.03): best stored bits by W = 4096: 62
