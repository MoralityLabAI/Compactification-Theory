# adapter_bits report: fu-a1-eta0.1

## Runs

W = trainable adapter parameters, M = items (random label bits). I = M(1 - H2(err)) is the Fano estimate of stored label information. Knee = fewest bits per weight keeping error within 1 point of full precision (on the gradient sample for the predicted columns).

| adapter | size | lr | W | M | M/W | err | I/W (bits/param) | knee | knee, typical-noise model | knee, worst-case (l-inf K) | median worst-case bits/item | K_inf | K_2 | tangent rank (1e-2) init / trained | top Hessian eig | counting bound held |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora:gate | 1 | 0.1 | 4096 | 256 | 0.0625 | 0.37891 | 0.0027 | 5 (sample 5) | 4 | 16 | 7.97 | 157.117 | 11.2262 | 21 / 23 | 19.35279 | yes |

## Information per parameter after rounding

Each cell is I_b / W in bits per parameter. The counting bound caps it at b (plus 32 bits per tensor for the scale).

| adapter | size | lr | M | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 | b=8 | b=10 | b=12 | b=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora:gate | 1 | 0.1 | 256 | 0.0 | 0.0 | 0.0 | 0.001 | 0.0036 | 0.0032 | 0.0028 | 0.002 | 0.0028 | 0.0027 | 0.0027 |

## Stored bits against the gradient matrix's rank

Cover's theorem caps a linearized adapter near twice the rank of its items-by-parameters gradient matrix. Rank is counted at two relative thresholds on a sample of items, so it cannot exceed the sample size.

| adapter | size | lr | M | I (bits) | sample rows | 2 x rank at 1e-2 | 2 x rank at 1e-3 | participation ratio |
|---|---|---|---|---|---|---|---|---|
| lora:gate | 1 | 0.1 | 256 | 10 | 256 | 46 | 310 | 1.06 |

## Capacity scaling

- lora:gate (lr 0.1): best stored bits by W = 4096: 10
