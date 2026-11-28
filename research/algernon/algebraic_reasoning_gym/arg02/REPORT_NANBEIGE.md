# ARG-02 extension N1 report: Nanbeige4.2-3B, a natively looped model

**Executed 2026-11-13** on the RTX 5080 Laptop GPU under Extension N1 of [PROTOCOL.md](PROTOCOL.md), registered before any Nanbeige output on development or test rows. Development selection ran once and the test stage ran once, all from commit `65039a5`. Numbers are from [`results_nanbeige/arg02_receipt.json`](results_nanbeige/arg02_receipt.json) and [`arg02_profile.json`](results_nanbeige/arg02_profile.json), with tables in [`arg02_tables.md`](results_nanbeige/arg02_tables.md). `analyze replay` regrades the committed answer tape to the receipt (31 arms).

Nanbeige4.2-3B (rev `451ed48c`) runs its 22 layers plus the final norm as one loop and was trained with 2 loops. Depth here is the number of loops; the adapter's loop-t readout equals the model's own forward with `num_loops = t` exactly on the real weights.

## Answer

**Recurrence helps this model, but the registered mechanism claim is not met.**

- **The gain over one pass.** At the selected depth of 8 loops the model recovered **71 of 168** test programs with correct withheld futures within 12 exact checks, against **44** for one pass: +27, cluster 95% CI [+0.057, +0.268] per case, McNemar p = 0.002.
- **What the gain survives.**
  - It beats an equal-compute ensemble of 8 single-loop forwards: 51, giving +20, CI [+0.012, +0.230], p = 0.029.
  - It beats the donor-shuffled recurrent state: 62, giving +9, CI [+0.016, +0.096], p = 0.012.
- **Where it fails.** The open-loop control comes within the interval: 55, giving +16, CI [-0.024, +0.215], p = 0.097. That is the one registered rule that fails.
- **Most of the gain is the trained second loop.**
  - Loop 2 vs loop 1: +21, p = 0.040.
  - Loop 8 vs loop 2: +6, CI [-0.099, +0.173], p = 0.61.
  - At depth 2, swapping in another case's recurrent displacement changes only 6 of 168 outcomes.
- **Still far below exact search.** The best Nanbeige arm stays below the 12-check class-size prior (102) and the complete certified class search (159).

## Development selection

| Choice | Development result (132 rows, future-correct@12) | Selected |
|---|---|---|
| shots, at loop 1 | 0 shots: 49; 4 shots: 48 | 0 shots |
| depth d* in {2, 4, 8} | loop 1: 49; loop 2: 45; loop 4: 53; loop 8: 63 | d* = 8 |

## Test results (168 positives; 12 negatives and 6 collision pairs as controls)

| Arm | future@12 | accepted@12 | top-1 future | novel generators (/96) | GPU s | PFLOP | p50 / p95 ms per case |
|---|---:|---:|---:|---:|---:|---:|---:|
| loop 1 (initial pass) | 44 | 53 | 10 | 25 | 147 | 7.0 | 743 / 897 |
| loop 2 (trained depth) | 65 | 72 | 6 | 37 | 287 | 14.0 | 1,454 / 1,767 |
| loop 4 | 76 | 85 | 11 | 45 | 562 | 28.0 | 2,852 / 3,396 |
| **loop 8 = d\*** | **71** | 83 | 8 | 46 | 1,119 | 56.0 | 5,673 / 6,954 |
| open loop, 8 | 55 | 63 | 11 | 28 | 151 | 7.0 | 767 / 958 |
| donor-shuffled state, 8 | 62 | 74 | 5 | 39 | 1,266 | 63.0 | 6,506 / 7,548 |
| ensemble of 8 single-loop forwards | 51 | 60 | 9 | 28 | 1,160 | 52.2 | 5,941 / 7,278 |
| external verifier-feedback controller (loop 1, <= 12 rounds) | 46 | 52 | - | 28 | 2,087 | 98.6 | 12,512 / 14,016 |
| class-size order, 12 checks (ARG-01) | 102 | 111 | - | 64 | CPU | - | 2 / 4 |
| complete certified class search (ARG-01) | 159 | 168 | - | 92 | CPU | - | 2 / 5 |
| frozen RRC-19 GRU (ARG-01) | 129 | 138 | - | 68 | CPU | - | 17 / 27 |

GPU seconds, PFLOP and latency cover all 192 test rows. The unconstrained argmax decode was never a valid three-operator program at any depth. With 0 shots, the model's top next token after "Program:" is never an operator, so every program above comes from the constrained 64-program ranking.

## Registered decisions (d* = 8)

| Rule | Comparison | A - B | 95% CI per case | McNemar p | Outcome |
|---|---|---:|---|---:|---|
| H1 depth | loop 8 vs loop 1 | +27 | [+0.057, +0.268] | 0.002 | met |
| H2a feedback | loop 8 vs open loop 8 | +16 | [-0.024, +0.215] | 0.097 | not met |
| H2b state | loop 8 vs donor-shuffled 8 | +9 (10 vs 1 discordant) | [+0.016, +0.096] | 0.012 | met |
| H3a repeat | loop 8 vs ensemble of 8 forwards | +20 | [+0.012, +0.230] | 0.029 | met |
| H3b feedforward | no same-family feedforward checkpoint on this machine | | | | not applicable |
| H3c efficiency | future@12 per end-to-end second: loop 8 0.063, loop 1 0.298, ensemble 0.044 | | | | not met |
| H4 symbolic | loop 8 vs complete certified class search | -88 | [-0.614, -0.429] | < 1e-4 | not met |
| **Recurrent mechanism claim** | H1, H2a, H2b, H3a | | | | **not met (H2a)** |

**What the open-loop control is for this model.** The loop body ends in the final norm, and the readout applies the LM head to that state directly. So the open loop's state after t steps is `t*h_1 - (t-1)*x`. Here `h_1` is about 24 times the norm of the embedding `x` (median relative step at loop 1: 23.8), so its logits are nearly `t` times the loop-1 logits. For Nanbeige the registered open-loop arm is therefore close to a temperature-sharpened single pass. That sharpening alone moves loop 1 from 44 to 55 under joint-log-probability ranking (+11, p = 0.035). The real loop's margin over it (+16 at depth 8, +17 at 2, +18 at 4) is consistent but not significant at any one depth.

## What the intermediate states show

- **The trained second loop carries most of the gain.** Accuracy over loops 1-8 was 44, 65, 63, 76, 80, 60, 68, 71.
  - Loop 2 vs loop 1: +21, CI [0.000, +0.256], p = 0.040, with heavy churn: 58 gained, 37 lost.
  - Beyond the trained depth the gains are not significant: loop 4 vs loop 2 +11, p = 0.080; loop 8 vs loop 2 +6, p = 0.61.
  - Loop 5 (80) was the best step, but it was not a registered depth.
- **The second loop's effect is nearly case-agnostic.** The shuffle adds another case's loop displacement `h_t - h_1` to this case's own `h_1`. At depth 2 it scores 69 vs the loop's 65, differing on only 6 of 168 cases. At depth 4 it scores 70 vs 76 (20 discordant). Only at depth 8 does the case's own displacement matter measurably: 71 vs 62, 10 vs 1 discordant. Most of what loop 2 does to the readout is a shift common to all cases.
- **It beats repeated single-pass compute.** Eight single-loop forwards over prompt variants reach only 51; two reach 44, the same as one. The loop's gain is not something extra single-pass compute buys.
- **No convergence.** After loop 3 each loop changes the state by 63-128% of its norm, with per-loop ratios oscillating between 0.49 and 1.50. There is no fixed point.
- **Early answer-site states get repaired; late ones carry the answer.**
  - Patching the five answer-site positions with a donor's states after loop 1 left 143 of 168 top-1 answers unchanged (median KL 0.012). The remaining loops largely overwrite an early corruption, which the untrained Qwen loop did not.
  - The same patch after loop 4 moved 55 answers to the donor's; after loop 8, 82.
- **Perturbations grow.**
  - A 0.25 x RMS kick after loop 1 grows to 2.8x the kick at loop 2, 3.9x at loop 4 and 6.9x at loop 8.
  - Top-1 agreement with the clean run falls from 135 (loop 2) to 69 (loop 8).
  - Accuracy at loop 8 drops from 71 to 62.
- **Probes see operator class only on seen generators.**
  - The outer operator of the canonical class representative is decodable above majority on seen-generator test rows (0.32-0.42 vs 0.26 at loops 1-8). It is not decodable on novel generators (at or below the 0.406 majority at every loop).
  - "Uses R" behaves the same way: 0.63-0.75 vs 0.53 on seen generators, 0.43-0.54 vs 0.50 on novel ones.
  - The loop does not build a generator-independent operator representation that a linear probe can read.

## Compared with the Qwen3.5 run (descriptive only: different models, tokenizers and training)

| Model and arm | future@12 /168 |
|---|---:|
| Qwen3.5-0.8B, one pass (4 shots) | 68 |
| Qwen3.5-0.8B, untrained block loop at its d* = 16 | 67 |
| Qwen3.5-2B, one pass | 79 |
| Nanbeige4.2-3B, loop 1 / loop 2 (trained) / loop 8 (d*) | 44 / 65 / 71 |
| class-size prior, 12 checks | 102 |
| complete certified class search | 159 |

An untrained loop added nothing to Qwen. Nanbeige's trained loop is needed just to reach the level of a single Qwen pass: its one-pass readout is weak, and the trained second loop restores it. Running past the trained depth adds a small, case-specific gain that the shuffle detects (+9 at 8 loops), but not a significant one over loop 2.

## Validity checks

- **Negatives and collision pairs.** No arm accepted any of the 12 negatives. Every arm gave identical traces for all 6 collision pairs except the shuffle (1 of 6), whose pair members have different donors by construction.
- **Row consistency.** Rows that share a prefix agree exactly (row spread 0) in every arm.
- **Cross-check against ARG-01.** ARG-01's `external.py` scorer agrees with ARG-02's top-1 grading at loops 1, 2, 4 and 8.
- **Provenance and resources.** All three stages ran from clean commit `65039a5`. Peak allocated VRAM was 8.4-8.8 GiB per stage. Loop 8 took 7.6x the GPU time of one loop (1,119 vs 147 s over 192 rows).
- **Same inputs as Qwen.** The exported public rows hash identically to the Qwen run's.

## Scope

- No non-looped Nanbeige checkpoint was available, so a same-family feedforward control (H3b) is missing; nothing was downloaded.
- Development chose 0 shots (49 vs 48). The unconstrained readout never produces the answer format, so the constrained ranking is the only answer channel.
- Probes were trained on 96 rows.
