# adapter_bits report: pilot-3

## Runs

W = trainable adapter parameters, M = items (random label bits). I = M(1 - H2(err)) is the Fano estimate of stored label information. Knee = fewest bits per weight keeping error within 1 point of full precision (on the gradient sample for the predicted columns).

| adapter | size | lr | W | M | M/W | err | I/W (bits/param) | knee | knee, typical-noise model | knee, worst-case (l-inf K) | median worst-case bits/item | K_inf | K_2 | tangent rank (1e-2) init / trained | top Hessian eig | counting bound held |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe | 128 | 0.05 | 128 | 128 | 1.0 | 0.0 | 1.0 | 4 (sample 4) | 4 | 7 | 5.59 | 184.3129 | 21.2523 | 124 / 124 | 0.07323 | yes |
| probe | 128 | 0.05 | 128 | 256 | 2.0 | 0.17578 | 0.6585 | 5 (sample 5) | 6 | 16 | 6.64 | 237.9354 | 27.3872 | 128 / 128 | 0.49242 | yes |
| probe | 128 | 0.05 | 128 | 384 | 3.0 | 0.25781 | 0.5297 | 5 (sample 5) | 5 | 12 | 6.38 | 237.9354 | 27.3872 | 128 / 128 | 0.39583 | yes |
| lora:gate | 1 | 0.03 | 4096 | 128 | 0.0312 | 0.20312 | 0.0085 | 5 (sample 5) | 8 | 16 | 10.11 | 45.5215 | 4.587 | 20 / 16 | 3.02215 | yes |
| lora:gate | 1 | 0.03 | 4096 | 512 | 0.125 | 0.10938 | 0.0627 | 8 (sample 8) | 12 | None | 12.53 | 1644.3243 | 146.8735 | 21 / 13 | 2061.17708 | yes |
| lora:gate | 1 | 0.03 | 4096 | 2048 | 0.5 | 0.354 | 0.0312 | 10 (sample 10) | 10 | None | 11.88 | 479.3029 | 40.5933 | 21 / 15 | 230.51484 | yes |

## Information per parameter after rounding

Each cell is I_b / W in bits per parameter. The counting bound caps it at b (plus 32 bits per tensor for the scale).

| adapter | size | lr | M | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 | b=8 | b=10 | b=12 | b=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe | 128 | 0.05 | 128 | 0.3561 | 0.338 | 0.8839 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| probe | 128 | 0.05 | 256 | 0.0911 | 0.0855 | 0.3652 | 0.4561 | 0.6412 | 0.5912 | 0.6076 | 0.6242 | 0.6585 | 0.6585 | 0.6585 |
| probe | 128 | 0.05 | 384 | 0.1147 | 0.1801 | 0.3299 | 0.4392 | 0.5179 | 0.5179 | 0.5297 | 0.5179 | 0.5297 | 0.5297 | 0.5297 |
| lora:gate | 1 | 0.03 | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 0.008 | 0.0076 | 0.009 | 0.0071 | 0.0085 | 0.0085 | 0.0085 |
| lora:gate | 1 | 0.03 | 512 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0002 | 0.0613 | 0.0627 | 0.0627 | 0.0627 |
| lora:gate | 1 | 0.03 | 2048 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.016 | 0.0091 | 0.0242 | 0.019 | 0.0325 | 0.0283 | 0.0306 |

## Stored bits against the gradient matrix's rank

Cover's theorem caps a linearized adapter near twice the rank of its items-by-parameters gradient matrix. Rank is counted at two relative thresholds on a sample of items, so it cannot exceed the sample size.

| adapter | size | lr | M | I (bits) | sample rows | 2 x rank at 1e-2 | 2 x rank at 1e-3 | participation ratio |
|---|---|---|---|---|---|---|---|---|
| probe | 128 | 0.05 | 128 | 128 | 128 | 248 | 254 | 57.07 |
| probe | 128 | 0.05 | 256 | 84 | 256 | 256 | 256 | 78.23 |
| probe | 128 | 0.05 | 384 | 67 | 384 | 256 | 256 | 91.25 |
| lora:gate | 1 | 0.03 | 128 | 34 | 128 | 32 | 202 | 1.02 |
| lora:gate | 1 | 0.03 | 512 | 257 | 512 | 26 | 290 | 1.01 |
| lora:gate | 1 | 0.03 | 2048 | 127 | 512 | 30 | 298 | 1.01 |

## Capacity scaling

- probe (lr 0.05): best stored bits by W = 128: 128
- lora:gate (lr 0.03): best stored bits by W = 4096: 257
