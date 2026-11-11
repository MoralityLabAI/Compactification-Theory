# ARG-02 report: recurrent depth in Qwen3.5-0.8B on the Algernon algebraic gym

**Executed 2026-11-11** on the RTX 5080 Laptop GPU under the protocol registered on 2026-11-10 ([PROTOCOL.md](PROTOCOL.md)), with one post-run amendment to the donor-shuffle control (Amendment 1). Development selection ran once, the test stage ran once, and the corrected shuffle arm ran once more. All numbers below are from [`results/arg02_receipt.json`](results/arg02_receipt.json) and [`results/arg02_profile.json`](results/arg02_profile.json). Full tables are in [`results/arg02_tables.md`](results/arg02_tables.md) and per-case grades in `results/arg02_traces.jsonl.gz`. `analyze replay` regrades the committed answer tape to the committed receipt on CPU.

## Answer

**No.** At the pre-registered depth d* = 16, the block loop recovered **67 of 168** test programs with correct withheld futures within 12 exact checks. The unmodified single pass recovered **68**. The difference is -1: 32 cases gained and 33 lost, cluster-bootstrap 95% CI [-0.085, +0.072] per case, McNemar p = 1.0. The development gain that selected this setting (+16 of 132) did not replicate. Extra recurrence reshuffled which programs were found without finding more. It also drove the unconstrained readout off the operator grammar and amplified small state perturbations about tenfold, at 3.5x the GPU time.

Every equal-compute control did at least as well, and the mechanism claim fails. Qwen3.5-2B in one pass recovered 79, using 0.76x the loop's FLOPs and 0.36x its GPU time. Every Qwen arm stayed far below the exact symbolic baselines: the complete certified class search recovered 159 in 0.14 CPU-seconds for the whole 192-row test set.

## Development selection

| Choice | Rule | Result |
|---|---|---|
| shots | depth-1 future-correct@12 on 132 development rows | 4 shots (52) over 0 shots (46) |
| block, d* | best over 4 blocks x depths 2/4/8/16 | `[8,12)` at d = 16: 68 vs 52 at depth 1 |

The selected block's development curve was 52, 48, 48, 49, 68 at depths 1, 2, 4, 8, 16. The other blocks at depth 16 scored 57 (`[12,16)`), 55 (`[16,20)`) and 44 (`[4,8)`). On the test rows the same block gave 68, 60, 60, 58 and 67 at those depths. d* was the maximum of 16 development cells, and the test run shows it was a selection artefact.

## Test results (168 positives; 12 negatives and 6 collision pairs as controls)

| Arm | future@12 | accepted@12 | top-1 future | greedy valid | novel generators future@12 (/96) | GPU s | PFLOP | p50 / p95 ms per case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loop(1) = base | 68 | 78 | 14 | 146 | 42 | 310 | 3.56 | 1,485 / 2,446 |
| loop(2) | 60 | 70 | 11 | 167 | 37 | 357 | 4.16 | 1,713 / 2,933 |
| loop(4) | 60 | 66 | 19 | 154 | 36 | 462 | 5.34 | 2,254 / 3,790 |
| loop(8) | 58 | 63 | 14 | 47 | 36 | 663 | 7.72 | 3,249 / 4,787 |
| **loop(16) = d\*** | **67** | 77 | 8 | 0 | 46 | 1,075 | 12.46 | 5,349 / 8,018 |
| open(16), no feedback | 46 | 57 | 8 | 0 | 23 | 302 | 3.56 | 1,474 / 2,238 |
| shuffle(16), corrected (A1) | 66 | 76 | 9 | 0 | 42 | 1,221 | 14.24 | 5,825 / 10,193 |
| shuffle(16), original | 67 | 77 | 9 | 0 | 42 | 1,237 | 14.24 | 6,032 / 9,242 |
| ensemble, 4 prompt variants | 69 | 79 | 15 | - | 41 | 1,207 | 13.94 | 6,006 / 8,520 |
| Qwen3.5-2B, one pass | 79 | 88 | 15 | 167 | 42 | 389 | 9.44 | 1,971 / 2,485 |
| external verifier-feedback controller, <= 12 rounds | 69 | 76 | - | - | 44 | 3,097 | 36.69 | 19,038 / 26,032 |
| class-size order, 12 checks (ARG-01) | 102 | 111 | - | - | 64 | CPU | - | 1 / 1 |
| complete certified class search (ARG-01) | 159 | 168 | - | - | 92 | CPU | - | 1 / 2 |
| exhaustive 64-program search (ARG-01) | 157 | 168 | - | - | 91 | CPU | - | 0 / 3 |
| frozen RRC-19 GRU (ARG-01) | 129 | 138 | - | - | 68 | CPU | 53k MACs/case | 5 / 7 |
| frozen RRC-19 FF (ARG-01) | 122 | 132 | - | - | 66 | CPU | 6k MACs/case | 1 / 1 |

The ARG-01 arms reproduce their committed counts exactly (159, 157, 102, 129, 122). Accuracy columns count the 168 positives; GPU seconds, PFLOP and latency cover all 192 test rows. "greedy valid" counts cases whose unconstrained argmax decode is three operator tokens. Latency is GPU time for that arm's own computation plus verifier time; the GPU was shared with another session's job for part of the test stage.

## Registered decisions (d* = 16)

| Rule | Comparison | A - B | 95% CI per case | McNemar p | Outcome |
|---|---|---:|---|---:|---|
| H1 depth | loop(16) vs loop(1) | -1 | [-0.085, +0.072] | 1.000 | not met |
| H2a feedback | loop(16) vs open(16) | +21 | [+0.023, +0.226] | 0.019 | met |
| H2b state | loop(16) vs corrected shuffle(16) | +1 | [-0.077, +0.082] | 1.000 | not met (original shuffle: +0, not met) |
| H3a repeat | loop(16) vs ensemble(4) | -2 | [-0.097, +0.070] | 0.902 | not met |
| H3b feedforward | loop(16) vs Qwen3.5-2B | -12 | [-0.166, +0.018] | 0.182 | not met |
| H3c efficiency | future@12 per end-to-end second | loop(16) 0.062, loop(1) 0.219, ensemble 0.057 | | | not met |
| H4 symbolic | loop(16) vs complete certified class search | -92 | [-0.608, -0.488] | < 1e-4 | not met |
| **Recurrent mechanism claim** | H1, H2a, H2b, H3a, H3b all met | | | | **not met** |

H2a is met only because removing the feedback is destructive: open(16) falls to 46, 22 below the base (CI [-0.224, -0.041], p = 0.013). Repeating the block's first update sixteen times without recomputing it damages the readout, while the closed loop returns to roughly the base level. That is not an advantage of recurrence over the single pass.

## What the intermediate states show

- **Churn, not refinement.** Accuracy over loop steps 1-16 was 68, 60, 54, 60, 54, 50, 49, 58, 70, 72, 80, 78, 74, 69, 72, 67. Between step 1 and step 16, 32 programs were gained and 33 lost at 12 checks; at top-1 the counts were 7 gained and 13 lost. 43 cases reached a future-correct top-1 answer only after step 1, but only 7 of them still had it at step 16.
  - The peak at step 11 (80) was not a registered depth. Development had no readout there, and against the base it gives +12 with CI [-0.023, +0.173], p = 0.18, before any correction for 16 steps. It is not evidence of a useful depth.
- **The readout leaves the grammar.** The unconstrained argmax decode was a valid three-operator program for 146 cases at step 1 and 167-168 at steps 2-3. It then fell to 47 at step 8, 1 at step 9 and 0 from step 10 onwards. Only the constrained 64-program ranking still produces programs at depth. The 2B model's greedy decode was valid for 167 cases.
- **No fixed point.**
  - The relative size of each block step is 0.75 at step 1 (the block's ordinary update) and shrinks to 0.45, 0.33 and 0.28.
  - It then stays near 0.20 from step 10 to 16, with a per-step reduction ratio of 0.97-1.03.
  - The iteration keeps moving the state by about a fifth of its norm per step and never converges.
- **Perturbations grow.** A kick of 0.25 x RMS after step 1 shrinks to 0.77 of its size by step 2. It then grows to 2.0x by step 8 and 10.9x by step 16. Top-1 agreement with the unperturbed run falls from 144/168 (step 1) to 81 (step 8) and 22 (step 16). Aggregate accuracy stays the same (69 vs 67 at step 16): deep-loop answers wander among similarly ranked candidates.
- **The shuffle tests case-specificity.** With the corrected alignment, replacing a case's own recurrent displacement `h_t - h_1` with another case's cost 8 programs relative to the loop at depths 2 and 4 (52 vs 60, and 16 below the base), so early steps do carry case-specific state. At depth 16 it changed nothing (66 vs 67). By then the recurrent displacement carries no recoverable advantage.
- **Patching the answer site.**
  - Replacing the last five positions with a donor's states at the final step moved 84 of 168 top-1 answers to the donor's answer (median KL to the clean run 5.15). At depth 16 the answer is largely carried at the readout positions.
  - The same patch at step 8 moved 28 to the donor and changed 154.
  - At step 1 it moved 4 to the donor, changed 91 and left 77 unchanged. Early-site states are mostly overwritten, but the trajectory stays sensitive to them.
- **Probes find nothing that grows with depth.**
  - Linear probes on the block output at the last prompt token were trained on 96 development rows and tuned on 36 calibration rows.
  - For the outer operator of the canonical class representative, accuracy on novel generators was at or below the 0.406 majority rate at 15 of 17 steps (above it only at steps 6 and 8). The best was 0.458 at step 8 (permutation p = 0.005), one of 34 uncorrected tests.
  - For "uses R", novel accuracy was 0.50-0.62 against 0.50, already 0.56 at step 1, and 0.52 at step 16.
  - Nothing here indicates that the loop builds a linearly readable operator class beyond the single pass.

## Compute

The 0.8B arms peaked at 3.9-4.9 GiB allocated and the 2B control at 6.0 GiB. The base forward took 1.5 s per case median with a 4-shot prompt (about 1,090 tokens x 16 candidate rows).

| Arm | future@12 (168 positives) per GPU-hour (192 rows) |
|---|---:|
| loop(1) | 790 |
| loop(16) | 224 |
| ensemble(4) | 206 |
| Qwen3.5-2B | 730 |
| external controller | 80 |

The external controller spent 10x the base GPU time and gained 1 program. Exact symbolic search needs no GPU: the complete certified class search ran 1,985 two-demonstration checks over all 192 test rows in 0.14 s of CPU time, against the loop's 1,790 checks after 1,075 GPU-seconds.

## Validity checks

- **Negatives and collision pairs.** No arm accepted any of the 12 inconsistent negatives. Every arm gave identical traces for all 6 collision pairs, except the shuffle arms (0 of 6). That is by construction: the two members of a pair have different donor rows.
- **Cross-check against ARG-01.** ARG-01's own `external.py` scorer, run on ARG-02's top-1 answers, agrees with ARG-02's grading at depths 1, 2, 4 and 8 (exact-accepted 18/16/24/15, future 14/11/19/14).
- **Row consistency.** Rows that share a prefix agree exactly (row spread 0) in every arm after Amendment 1. Before the fix, 40% of the original shuffle records disagreed by up to 5.4 nats. The corrected arm's result (66) and the original (67) lead to the same decision.
- **Provenance.** All stages ran from clean commits. Development and the main test stage ran at `deea6c6`; the 2B control at `547f190` (same runner code); the corrected shuffle at `5b7d2bc`. Every part is keyed on the code, config, checkpoint lock and selection digests.

## Scope

- **The recurrence was untrained.** No looped-layer Qwen checkpoint exists locally, so the loop re-applies pretrained layers the model never learned to iterate. A model trained with recurrence (Huginn-style, or a Qwen with a trained loop) is the stronger test of the question and is not covered here.
- **The 0.8B base is weak on this task even in one pass.** At 68/168 it is below the 12-check class-size prior (102), so effects near the floor are hard to see.
- **Setup choices.** Answers come from a constrained ranking over all 64 programs with one prompt format; probes used 96 training rows.

## Reproduce

```sh
PYTHONPATH=.:../EvalMath/src python -m research.algernon.algebraic_reasoning_gym.arg02.analyze replay \
  --tape research/algernon/algebraic_reasoning_gym/arg02/results/arg02_answer_tape.jsonl.gz \
  --check research/algernon/algebraic_reasoning_gym/arg02/results/arg02_receipt.json
```

The GPU chain is `run_arg02.ps1` (README), followed by `runner test --arms test_shuffle_a1` for Amendment 1.
