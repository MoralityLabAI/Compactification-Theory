Selection (development data): shots 0, block [0, 22], d* = 8, ensemble size m(d*) = 8. Development future-correct@12: 49 at depth 1, 63 at the selection.

## Arms on the 168 test positives

| Arm | future@12 | accepted@12 | future@1 | greedy valid | greedy accepted | novel future@12 | negatives accepted | collision pairs same trace | GPU s | end-to-end s | p50 / p95 ms | PFLOP | future@12 per e2e s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| loop@1 | 44 | 53 | 10 | 0 | 0 | 25 | 0 | 6 | 147.2 | 147.6 | 743 / 897 | 7.003 | 0.2982 |
| loop@2 | 65 | 72 | 6 | 0 | 0 | 37 | 0 | 6 | 286.8 | 287.2 | 1454 / 1767 | 13.997 | 0.2263 |
| loop@4 | 76 | 85 | 11 | 0 | 0 | 45 | 0 | 6 | 562.3 | 562.7 | 2852 / 3396 | 27.985 | 0.1351 |
| loop@8 | 71 | 83 | 8 | 0 | 0 | 46 | 0 | 6 | 1119.1 | 1119.4 | 5673 / 6954 | 55.961 | 0.0634 |
| open@8 | 55 | 63 | 11 | 0 | 0 | 28 | 0 | 6 | 151.0 | 151.3 | 767 / 958 | 7.003 | 0.3634 |
| shuffle@8 | 62 | 74 | 5 | 0 | 0 | 39 | 0 | 1 | 1266.4 | 1266.6 | 6506 / 7548 | 62.955 | 0.0490 |
| ensemble@8 | 51 | 60 | 9 | 0 | 0 | 28 | 0 | 6 | 1159.6 | 1160.0 | 5941 / 7278 | 52.206 | 0.0440 |
| controller | 46 | 52 | - | - | - | 28 | 0 | 6 | 2087.1 | 2089.3 | 12512 / 14016 | 98.580 | 0.0220 |
| symbolic:certified_mass12 | 102 | 111 | - | - | - | 64 | 0 | 6 | 0.0 | 0.4 | 2 / 4 | 0.000 | 273.5809 |
| symbolic:certified_mass_full | 159 | 168 | - | - | - | 92 | 0 | 6 | 0.0 | 0.5 | 2 / 5 | 0.000 | 340.9566 |
| symbolic:exhaustive_64 | 157 | 168 | - | - | - | 91 | 0 | 6 | 0.0 | 0.6 | 1 / 11 | 0.000 | 273.6052 |
| symbolic:frozen_ff | 122 | 132 | - | - | - | 66 | 0 | 6 | 0.0 | 0.6 | 3 / 5 | 0.000 | 221.4857 |
| symbolic:frozen_gru | 129 | 138 | - | - | - | 68 | 0 | 6 | 0.0 | 3.3 | 17 / 27 | 0.000 | 38.7455 |

## Paired comparisons, future-correct at 12 checks

| A vs B | A | B | A only | B only | A - B | cluster 95% CI (per case) | McNemar p |
|---|---:|---:|---:|---:|---:|---|---:|
| controller vs loop@1 | 46 | 44 | 15 | 13 | +2 | [-0.067, +0.086] | 0.8506 |
| loop@1 vs symbolic:certified_mass12 | 44 | 102 | 21 | 79 | -58 | [-0.444, -0.244] | 0.0000 |
| loop@1 vs symbolic:frozen_ff | 44 | 122 | 13 | 91 | -78 | [-0.563, -0.360] | 0.0000 |
| loop@1 vs symbolic:frozen_gru | 44 | 129 | 8 | 93 | -85 | [-0.611, -0.399] | 0.0000 |
| loop@2 vs loop@1 | 65 | 44 | 58 | 37 | +21 | [+0.000, +0.256] | 0.0396 |
| loop@4 vs loop@1 | 76 | 44 | 66 | 34 | +32 | [+0.068, +0.312] | 0.0018 |
| loop@8 vs ensemble@8 | 71 | 51 | 48 | 28 | +20 | [+0.012, +0.230] | 0.0286 |
| loop@8 vs loop@1 | 71 | 44 | 50 | 23 | +27 | [+0.057, +0.268] | 0.0021 |
| loop@8 vs loop@2 | 71 | 65 | 50 | 44 | +6 | [-0.099, +0.173] | 0.6063 |
| loop@8 vs open@8 | 71 | 55 | 49 | 33 | +16 | [-0.024, +0.215] | 0.0970 |
| loop@8 vs shuffle@8 | 71 | 62 | 10 | 1 | +9 | [+0.016, +0.096] | 0.0117 |
| loop@8 vs symbolic:certified_mass12 | 71 | 102 | 14 | 45 | -31 | [-0.273, -0.095] | 0.0001 |
| loop@8 vs symbolic:certified_mass_full | 71 | 159 | 0 | 88 | -88 | [-0.614, -0.429] | 0.0000 |
| loop@8 vs symbolic:exhaustive_64 | 71 | 157 | 0 | 86 | -86 | [-0.598, -0.421] | 0.0000 |
| open@2 vs loop@1 | 48 | 44 | 5 | 1 | +4 | [+0.000, +0.051] | 0.2188 |
| open@4 vs loop@1 | 58 | 44 | 19 | 5 | +14 | [+0.034, +0.137] | 0.0066 |
| open@8 vs loop@1 | 55 | 44 | 17 | 6 | +11 | [+0.012, +0.122] | 0.0347 |
| shuffle@2 vs loop@1 | 69 | 44 | 63 | 38 | +25 | [+0.019, +0.281] | 0.0165 |
| shuffle@4 vs loop@1 | 70 | 44 | 61 | 35 | +26 | [+0.039, +0.267] | 0.0103 |
| shuffle@8 vs loop@1 | 62 | 44 | 42 | 24 | +18 | [+0.016, +0.200] | 0.0356 |

## Decision rules

- H1_depth: **met**
- H2a_feedback: **not met**
- H2b_state: **met**
- H2b_state_original_shuffle: **met**
- H3a_repeat: **met**
- H3b_ff: **not applicable**
- H3c_efficiency: **not met**
- H4_symbolic: **not met**
- mechanism_claim: **not met**

## Depth curve (loop, test positives)

| step | future@12 | future@1 | accepted@1 | greedy valid | greedy accepted | median relative residual | median reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44 | 10 | 12 | 0 | 0 | 23.787 | - |
| 2 | 65 | 6 | 10 | 0 | 0 | 5.828 | 0.245 |
| 3 | 63 | 8 | 9 | 0 | 0 | 0.813 | 0.140 |
| 4 | 76 | 11 | 13 | 0 | 0 | 0.961 | 1.183 |
| 5 | 80 | 7 | 7 | 0 | 0 | 0.850 | 0.885 |
| 6 | 60 | 10 | 13 | 0 | 0 | 1.277 | 1.503 |
| 7 | 68 | 7 | 7 | 0 | 0 | 0.628 | 0.492 |
| 8 | 71 | 8 | 8 | 0 | 0 | 0.688 | 1.094 |

First step with a future-correct top-1 answer: {"1": 10, "2": 6, "3": 8, "4": 9, "6": 9, "8": 4, "None": 122}. Found only after step 1: 36; still correct at the last step: 8.

| depth vs step 1 | lost | gained |
|---|---:|---:|
| at12@2 | 37 | 58 |
| at12@4 | 34 | 66 |
| at12@8 | 23 | 50 |
| at1@2 | 10 | 6 |
| at1@4 | 10 | 11 |
| at1@8 | 10 | 8 |

## Perturbation (kick 0.25 x RMS after step 1)

| readout step | future@12 | clean future@12 | top-1 agrees with clean | median distance / kick |
|---:|---:|---:|---:|---:|
| 1 | 45 | 44 | 111 | 1.000 |
| 2 | 66 | 65 | 135 | 2.849 |
| 4 | 74 | 76 | 73 | 3.888 |
| 8 | 62 | 71 | 69 | 6.855 |

## Answer-site patching (donor site states, loop to 8)

| patch step | top-1 kept | top-1 moved to donor's | top-1 changed | future@12 | median KL to clean | p95 KL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 143 | 10 | 25 | 72 | 0.0118 | 0.0548 |
| 4 | 94 | 55 | 74 | 67 | 0.1020 | 0.4779 |
| 8 | 86 | 82 | 82 | 63 | 0.1632 | 0.7038 |

## Linear probes (block output at the last prompt token, step 0 = block input; train 96, select 36, test seen 72, novel 96)

**outer_op**

| step | novel acc | novel majority | novel perm. p | seen acc | seen majority | k | lam |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.406 | 0.406 | 0.602 | 0.264 | 0.264 | 8 | 0.001 |
| 1 | 0.375 | 0.406 | 0.050 | 0.347 | 0.264 | 16 | 0.01 |
| 2 | 0.344 | 0.406 | 0.328 | 0.319 | 0.264 | 8 | 0.1 |
| 3 | 0.375 | 0.406 | 0.189 | 0.417 | 0.264 | 8 | 0.1 |
| 4 | 0.354 | 0.406 | 0.070 | 0.417 | 0.264 | 32 | 0.01 |
| 5 | 0.375 | 0.406 | 0.015 | 0.389 | 0.264 | 32 | 0.01 |
| 6 | 0.354 | 0.406 | 0.199 | 0.347 | 0.264 | 8 | 0.001 |
| 7 | 0.333 | 0.406 | 0.109 | 0.403 | 0.264 | 32 | 0.001 |
| 8 | 0.385 | 0.406 | 0.025 | 0.403 | 0.264 | 32 | 0.001 |

**uses_R**

| step | novel acc | novel majority | novel perm. p | seen acc | seen majority | k | lam |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.500 | 0.500 | 1.000 | 0.528 | 0.528 | 8 | 0.001 |
| 1 | 0.542 | 0.500 | 0.164 | 0.653 | 0.528 | 32 | 1.0 |
| 2 | 0.490 | 0.500 | 0.657 | 0.681 | 0.528 | 16 | 1.0 |
| 3 | 0.521 | 0.500 | 0.239 | 0.639 | 0.528 | 16 | 1.0 |
| 4 | 0.500 | 0.500 | 0.517 | 0.750 | 0.528 | 16 | 1.0 |
| 5 | 0.490 | 0.500 | 0.697 | 0.722 | 0.528 | 16 | 1.0 |
| 6 | 0.479 | 0.500 | 0.701 | 0.625 | 0.528 | 32 | 0.001 |
| 7 | 0.542 | 0.500 | 0.254 | 0.653 | 0.528 | 32 | 1.0 |
| 8 | 0.427 | 0.500 | 0.995 | 0.639 | 0.528 | 8 | 1.0 |

## Future-correct@12 by source family

| family | loop@1 | loop@8 | symbolic:certified_mass_full | symbolic:frozen_gru |
|---|---:|---:|---:|---:|
| lacunary | 7 | 8 | 24 | 22 |
| product | 9 | 10 | 24 | 24 |
| rational | 7 | 11 | 23 | 15 |
| signed | 5 | 11 | 24 | 22 |
| sparse | 5 | 4 | 19 | 15 |
| squared | 3 | 10 | 21 | 17 |
| twisted_product | 8 | 17 | 24 | 14 |

## ARG-01 external.py cross-check (top-1 loop answers)

- depth 1: ARG-01 exact accepted 12 = ARG-02 12; future 10 = 10
- depth 2: ARG-01 exact accepted 10 = ARG-02 10; future 6 = 6
- depth 4: ARG-01 exact accepted 15 = ARG-02 15; future 12 = 12
- depth 8: ARG-01 exact accepted 8 = ARG-02 8; future 8 = 8

## Peak allocated VRAM per stage (GiB)

test_controller 8.79, test_ensemble 8.41, test_loop 8.57, test_open 8.41, test_patch 8.65, test_perturb 8.58, test_shuffle 8.52
