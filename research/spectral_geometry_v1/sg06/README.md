# SG-06: measured CPU reasoning allocation

[Executable campaign, pinned source](https://github.com/MoralityLabAI/DataGoblin/blob/23b25df7f2efd26954e1b85076f1bfe5b761807d/training/rlt_gym/sg06_spectral_allocation/sg06.py) · [Protocol](https://github.com/MoralityLabAI/DataGoblin/blob/23b25df7f2efd26954e1b85076f1bfe5b761807d/training/rlt_gym/sg06_spectral_allocation/README.md).

48 training roots and 48 separate exposed evaluation roots; persistent, rotating and mixed utility regimes. 304 eligible imitation labels, 240 dual. Tied tanh recurrent allocator (313 parameters), feedforward baseline (169), and shuffled recurrent (313), plus five deterministic/random controls. This is a recurrent RNN, not a canonical TRM. One policy initialization per architecture; 120 Adam steps. All arms select the best-bound state, then choose expansion or a zero-start reverse coordinate sweep. Shared initial incumbent, 2048 transitions, 24 proposals, 80 decisions; fall back to the sole feasible action. All 768 episodes replayed using the existing DataGoblin verifier and independently checked against the exact SG-03 objective; zero violations.

## Work caps
| Policy | CPU ms | Mean transitions | Mean proposals | Mean gap | Certified /48 |
|---|---:|---:|---:|---:|---:|
| Expansion priority |18.4070|565.000|0.208|0.2292|47|
| Root dual |17.3318|541.958|1.208|0.2292|47|
| Always dual |13.5504|265.208|7.521|0|48|
| Heuristic |14.3847|298.792|6.250|0|48|
| Random |16.1711|371.333|4.063|0|48|
| MLP |15.1320|269.875|6.125|0|48|
| Recurrent |15.6952|307.000|6.000|0|48|
| Shuffled recurrent |14.7748|270.167|6.688|0|48|

Always-dual saves 26.38% mean process CPU versus expansion priority, wins the paired time comparison 34/48, and certifies one additional task. Heuristic wins time 37/48. Recurrent wins 28/48. One host and one timing repetition, no inferential claim.

## Soft 20 ms CPU deadline
| Policy | Mean CPU ms | Mean gap | Mean regret | Certified /48 | Optimal /48 |
|---|---:|---:|---:|---:|---:|
| Expansion priority |15.0195|3.1875|1.6042|32|35|
| Root dual |13.7531|3.0208|1.6042|33|35|
| Always dual |12.8015|1.8750|1.1458|36|38|
| Heuristic |13.3810|1.8333|1.1458|36|38|
| Random |14.0453|2.5833|1.2708|35|37|
| MLP |13.3419|2.0625|1.1458|36|38|
| Recurrent |13.5364|2.2292|1.1458|35|38|
| Shuffled recurrent |13.1286|1.8750|1.1458|37|38|

Deadline checked between atomic actions, hence not a hard cap. Max overshoot 24.20 ms (random); all overshoots retained in original full receipt. Timed computation includes policy inference, initialization, verifier actions and final receipt, excludes offline teacher/training, exact reference and post-hoc replay. Training CPU 3.276 seconds. Torch 2.10.0+cpu, one thread. Imitation loss MLP 0.258594, recurrent 0.174524, shuffled 0.295799. Lower training loss does not imply better fresh-root allocation. Analytic dual allocation is the recommended default for this finite workload. Use new roots and equal-capacity controls for any new learned hypothesis.

## Evidence custody
Executed source Git blob: 8bb3f55840722b52cfcfd3ff0633ab22c0f2db1c (verified against DataGoblin commit above). Exact local results.json SHA256: 1bf90a72ded675e81bd8e84e27cb51ed50ab642ba5df3347ad7d7a70007db474. Raw 2,562,315 bytes. Full receipt (including every event and timing) is provided as the session's downloadable archive sg06_cpu_campaign.zip, SHA256 00e3ed4e1f8f037f23add4a5d905ff1ae7bd0b76a4d1a19ba6969906c5a819b0. Only aggregate report and exact executable are committed, not original raw timings; regenerating creates a new timing run. Full repo tests/Ruff and native causal confirmation pending. No GPU or paid inference.

Next: investigate lazy bound refinement and amortization of exact suffix-state caches, with paired measured-time baselines; do not increase learned controller size on these exposed roots.
