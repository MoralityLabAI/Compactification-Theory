# Follow-ups: optimizer cap, seeds, saturation

Written and pushed **before** any of these runs, on 2026-10-12, after a calibration run (`gd-calib-2`, below). Each prediction is labelled:
- **P**: follows from a proof in the note.
- **E**: established empirically elsewhere (cited from memory).
- **J**: conjecture.

Results will be added under each prediction without editing it.

## Calibration that shaped the design (`gd-calib-2`)

Setup: full-batch plain gradient descent, 4,000 steps, last-layer gate LoRA rank 1 (W = 4,096), M = 256 random labels, seed 0.

- **η = 0.1.** Sharpness rose to 2/η = 20 and stayed there: η·λ was 1.70 to 2.35 at every tracked point from step 500. The loss oscillated, and the adapter stored 11 bits.
- **η = 0.01 and 0.001.** The runs were far from the stability limit (η·λ = 0.20 and 0.007 at the end). They were simply slow, storing 5 and 3 bits.
- **For contrast, Adam** (lr 0.03, 3,000 steps) memorized all 256 labels on the same setup, ending at sharpness about 1,400 (`results/sweep-1`).

So comparisons across step sizes are made at equal gradient-flow time T = η·steps = 400.

## A. Does the optimizer's stability cap limit what an adapter can store?

The note's §4.3 conjecture is this: gradient descent at step η can only settle where sharpness is at most about 2/η, so it cannot reach sharp high-capacity solutions. On this model, last-layer solutions that store labels are sharp (λ ≈ 10^3 under Adam). Layer-24 solutions are flat (λ ≈ 5 to 50).

**A1.** Layer 27, gate LoRA rank 1, M = 256, full batch, T = 400. Step sizes and steps:

| η | steps |
|---|---|
| 0.1 | 4,000 |
| 0.03 | 13,333 |
| 0.01 | 40,000 |
| 0.003 | 133,333 |

- **P-A1a (E: Cohen et al. 2021, edge of stability).** In any run whose tracked sharpness reaches 1.5/η, every later tracked value stays within [1.5, 2.6]/η.
- **P-A1b (J).** Stored bits at T = 400 do not increase with η: I(0.003) ≥ I(0.01) ≥ I(0.03) ≥ I(0.1), each within 10 bits of tolerance.
- **P-A1c (J).** No run with η ≥ 0.01 memorizes all 256 labels. Adam's memorizing solution has λ ≈ 1,400, above 2/η = 200.

**A2.** Layer 24 alone, gate LoRA rank 4 (`24:4`), M = 256, full batch, T = 400, η ∈ {0.1, 0.03}.

- **P-A2 (J).** At each η, layer 24 stores at least 4 times the bits that layer 27 stores in A1 at the same η, because flat high-capacity solutions exist there.

## B. Seeds 1 and 2

New random keys, labels and initializations per seed. Adam with the original settings.

**B1.** Single layer, pool 4,096, 3,000 steps:
- probe k = 64 at M = 128;
- probe k = 512 at M = 1,024;
- gate LoRA rank 1 at M = 256;
- gate LoRA rank 8 at M = 512.

**B2.** Depth, pool 2,048, M = 2,048, 4,000-step cap, specs `27:4`, `24:4` and `24:1,25:1,26:1,27:1`.

- **P-B1 (P).** The counting bound I_b ≤ W·b + 32·tensors holds on every run at every rounding level.
- **P-B2 (E: Cover's theorem).** Probe k = 64 stores at least 1.5 bits per parameter at M = 128. Probe k = 512 stores at least 0.8 bits per parameter at M = 1,024.
- **P-B3 (J, replicating sweep-1).** Gate LoRA rank 8 (W = 32,768) stores less than 3 times the bits of rank 1 (W = 4,096) on each seed.
- **P-B4 (J, replicating deep-2/3).** At M = 2,048, `24:4` and the four-layer adapter each store at least 5 times the bits of `27:4`. The four-layer adapter stores at least as much as `24:4`, minus 50 bits.
- **P-B5 (J).** The typical-noise model predicts every probe run's knee within 2 bits.

## C. Where the four-layer adapter saturates

Setup: last-token gate LoRA rank 1 in each of layers 24 to 27 (W = 16,384), M = 16,384, Adam lr 0.03, cap of 12,000 steps. Gradient rows are sampled on 2,048 items, so the rank estimate can reach 2,048.

- **P-C1 (J, the spectral-capacity reading).** Stored bits are at most 1.3 times twice the trained gradient matrix's rank at relative threshold 1e-3. If that rank reaches the 2,048-row cap, the test is inconclusive and is reported as such.
- **P-C2 (J).** The knee is at most 9 bits per weight, and the top Hessian eigenvalue is at most 100. That is, high capacity here still comes without precision or sharpness.

## Queue

`scripts/followups.cmd` runs A1, B1, B2, A2 and C in that order, one model-loading job at a time. Each job waits for 12 GB of free commit memory before loading the model.
