# adapter_bits report: deep-2

## Runs

W = trainable adapter parameters, M = items (random label bits). I = M(1 - H2(err)) is the Fano estimate of stored label information. Knee = fewest bits per weight keeping error within 1 point of full precision (on the gradient sample for the predicted columns).

| adapter | size | lr | W | M | M/W | err | I/W (bits/param) | knee | knee, typical-noise model | knee, worst-case (l-inf K) | median worst-case bits/item | K_inf | K_2 | tangent rank (1e-2) init / trained | top Hessian eig | counting bound held |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deep-gate | 27:4 | 0.03 | 16384 | 512 | 0.0312 | 0.13477 | 0.0134 | 8 (sample 8) | 12 | None | 13.46 | 2827.7078 | 132.384 | 22 / 16 | 1297.38696 | yes |
| deep-gate | 27:4 | 0.03 | 16384 | 2048 | 0.125 | 0.3667 | 0.0065 | 8 (sample 8) | 10 | None | 12.15 | 1061.6202 | 46.7995 | 22 / 16 | 139.72103 | yes |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 16384 | 512 | 0.0312 | 0.0 | 0.0312 | 7 (sample 7) | 7 | 16 | 10.51 | 6943.2266 | 160.0515 | 35 / 282 | 16.63505 | yes |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 16384 | 2048 | 0.125 | 0.0 | 0.125 | 7 (sample 7) | 7 | 12 | 10.2 | 10271.874 | 261.5492 | 35 / 503 | 10.05135 | yes |
| deep-gate | 26:2,27:2 | 0.03 | 16384 | 512 | 0.0312 | 0.0 | 0.0312 | 10 (sample 10) | 12 | 16 | 12.56 | 9948.5859 | 541.8307 | 29 / 26 | 1535.50948 | yes |
| deep-gate | 26:2,27:2 | 0.03 | 16384 | 2048 | 0.125 | 0.15332 | 0.0477 | 10 (sample 10) | 10 | None | 12.33 | 5317.3687 | 157.3011 | 28 / 207 | 141.15275 | yes |

## Information per parameter after rounding

Each cell is I_b / W in bits per parameter. The counting bound caps it at b (plus 32 bits per tensor for the scale).

| adapter | size | lr | M | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 | b=8 | b=10 | b=12 | b=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deep-gate | 27:4 | 0.03 | 512 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0038 | 0.0016 | 0.0136 | 0.0089 | 0.0126 | 0.0134 |
| deep-gate | 27:4 | 0.03 | 2048 | 0.0 | 0.0 | 0.0 | 0.0 | 0.005 | 0.0052 | 0.0042 | 0.0065 | 0.0064 | 0.0066 | 0.0065 |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 512 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0011 | 0.0191 | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0312 |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 2048 | 0.0 | 0.0 | 0.0 | 0.001 | 0.0125 | 0.093 | 0.1225 | 0.1242 | 0.125 | 0.125 | 0.125 |
| deep-gate | 26:2,27:2 | 0.03 | 512 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0001 | 0.0091 | 0.003 | 0.0306 | 0.0312 | 0.0312 |
| deep-gate | 26:2,27:2 | 0.03 | 2048 | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 | 0.024 | 0.0343 | 0.033 | 0.0476 | 0.0477 | 0.0477 |

## Stored bits against the gradient matrix's rank

Cover's theorem caps a linearized adapter near twice the rank of its items-by-parameters gradient matrix. Rank is counted at two relative thresholds on a sample of items, so it cannot exceed the sample size.

| adapter | size | lr | M | I (bits) | sample rows | 2 x rank at 1e-2 | 2 x rank at 1e-3 | participation ratio |
|---|---|---|---|---|---|---|---|---|
| deep-gate | 27:4 | 0.03 | 512 | 220 | 512 | 32 | 338 | 1.03 |
| deep-gate | 27:4 | 0.03 | 2048 | 106 | 512 | 32 | 314 | 1.02 |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 512 | 512 | 512 | 564 | 1022 | 3.57 |
| deep-gate | 24:1,25:1,26:1,27:1 | 0.03 | 2048 | 2048 | 512 | 1006 | 1022 | 14.2 |
| deep-gate | 26:2,27:2 | 0.03 | 512 | 512 | 512 | 52 | 564 | 1.3 |
| deep-gate | 26:2,27:2 | 0.03 | 2048 | 782 | 512 | 414 | 938 | 4.37 |

## Capacity scaling

- deep-gate (lr 0.03): best stored bits by W = 16384: 2048
