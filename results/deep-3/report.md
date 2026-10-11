# adapter_bits report: deep-3

## Runs

W = trainable adapter parameters, M = items (random label bits). I = M(1 - H2(err)) is the Fano estimate of stored label information. Knee = fewest bits per weight keeping error within 1 point of full precision (on the gradient sample for the predicted columns).

| adapter | size | lr | W | M | M/W | err | I/W (bits/param) | knee | knee, typical-noise model | knee, worst-case (l-inf K) | median worst-case bits/item | K_inf | K_2 | tangent rank (1e-2) init / trained | top Hessian eig | counting bound held |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deep-gate | 24:4 | 0.03 | 16384 | 2048 | 0.125 | 0.0 | 0.125 | 8 (sample 8) | 8 | 16 | 11.71 | 3541.1099 | 106.8463 | 37 / 509 | 20.12085 | yes |
| deep-gate | 24:4 | 0.03 | 16384 | 8192 | 0.5 | 0.31287 | 0.0518 | 8 (sample 10) | 8 | None | 11.83 | 1161.2224 | 26.5249 | 40 / 385 | 5.37455 | yes |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 16384 | 2048 | 0.125 | 0.0 | 0.125 | 7 (sample 7) | 7 | 16 | 10.56 | 13851.4561 | 342.7023 | 35 / 463 | 8.30103 | yes |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 16384 | 8192 | 0.5 | 0.14575 | 0.2004 | 8 (sample 10) | 8 | 16 | 11.23 | 7733.9429 | 188.1188 | 38 / 423 | 50.7095 | yes |

## Information per parameter after rounding

Each cell is I_b / W in bits per parameter. The counting bound caps it at b (plus 32 bits per tensor for the scale).

| adapter | size | lr | M | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 | b=8 | b=10 | b=12 | b=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deep-gate | 24:4 | 0.03 | 2048 | 0.0 | 0.0 | 0.0 | 0.0003 | 0.0011 | 0.0076 | 0.0909 | 0.123 | 0.125 | 0.125 | 0.125 |
| deep-gate | 24:4 | 0.03 | 8192 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0038 | 0.0178 | 0.0409 | 0.047 | 0.0524 | 0.0516 | 0.0517 |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 2048 | 0.0 | 0.0 | 0.0 | 0.0021 | 0.0165 | 0.0606 | 0.1199 | 0.1225 | 0.125 | 0.125 | 0.125 |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 8192 | 0.0 | 0.0 | 0.0 | 0.0005 | 0.025 | 0.1036 | 0.1661 | 0.1898 | 0.2004 | 0.2001 | 0.2004 |

## Stored bits against the gradient matrix's rank

Cover's theorem caps a linearized adapter near twice the rank of its items-by-parameters gradient matrix. Rank is counted at two relative thresholds on a sample of items, so it cannot exceed the sample size.

| adapter | size | lr | M | I (bits) | sample rows | 2 x rank at 1e-2 | 2 x rank at 1e-3 | participation ratio |
|---|---|---|---|---|---|---|---|---|
| deep-gate | 24:4 | 0.03 | 2048 | 2048 | 512 | 1018 | 1022 | 15.39 |
| deep-gate | 24:4 | 0.03 | 8192 | 848 | 512 | 770 | 1022 | 7.9 |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 2048 | 2048 | 512 | 926 | 1022 | 12.7 |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 8192 | 3284 | 512 | 846 | 1022 | 12.85 |

## Capacity scaling

- deep-gate (lr 0.03): best stored bits by W = 16384: 3284
