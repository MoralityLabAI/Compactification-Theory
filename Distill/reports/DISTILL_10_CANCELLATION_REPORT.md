# DISTILL-10 / Counterfactual cancellation and risk-aware research opportunities

Tested commit: `5fa63c7048e45923c4fb0da6a4e1aa5d9db9ccfc`. [Cancellation workflow](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46038835173) passed four targeted tests and the full campaign. [Offline suite](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46038835044) passed 69 tests. Full root rows and certificate receipts are in [artifact 15218697693](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46038835173/artifacts/15218697693) (30-day retention). Report ID: `d47e3cc954c2af8d9fc4073f412485ea6c8d2cb1ddc9b965c6ee56f98ce71427`.

## Frozen campaign

Eight calibration roots; 36 fresh evaluation roots (12 ordinary, 12 shifted, 12 scarcity); all 64 worlds per root, 13 arms, two trajectory work caps. 451,008 replayed world transitions, 4,856 audited utility/risk intervals and 396 explicit priced action replays. Counts include repeated caps, legacy arms and forced controls. Actor-only process CPU: 14.633 s over all evaluation arms, with 0 descriptive envelope overruns. Logical work is a declared accounting abstraction.

### Mean priced-utility difference from the budget rule

| Policy | Ordinary 8192 | Shifted 8192 | Scarce 8192 | Ordinary 32768 | Shifted 32768 | Scarce 32768 |
|---|---:|---:|---:|---:|---:|---:|
| budget_rule | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| legacy_late | 0.04059 | 0.02200 | -0.11265 | 0.04059 | 0.02200 | -0.11265 |
| legacy_bounded | -0.22166 | -0.30730 | -0.62650 | -0.22166 | -0.30730 | -0.62650 |
| paired_late | -0.02778 | -0.10592 | -0.14109 | -0.02778 | -0.10592 | -0.14109 |
| split_late | -0.02778 | -0.10592 | -0.14109 | -0.02778 | -0.10592 | -0.14109 |
| forced_late | -0.15564 | -0.10592 | -0.21373 | -0.15564 | -0.10592 | -0.21373 |
| paired_tail | -1.01803 | -1.00736 | -0.77945 | -1.01803 | -1.00736 | -0.77945 |
| split_tail | -1.01803 | -1.00736 | -0.77945 | -1.01803 | -1.00736 | -0.77945 |
| forced_tail | -1.01803 | -1.00736 | -0.85208 | -1.01803 | -1.00736 | -0.85208 |
| paired_broad | -1.50814 | -1.50521 | -1.53273 | -2.00146 | -2.00342 | -1.95017 |
| split_broad | -1.50814 | -1.50521 | -1.53273 | -2.00146 | -2.00342 | -1.95017 |
| forced_broad | -1.50814 | -1.50521 | -1.53273 | -2.00146 | -2.00342 | -1.95017 |
| calibrated | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

## Main findings

1. Signed-state cancellation is exact on the targeted algebraic fixture and the interrupted expansion preserves its previous sound interval. In the actual registered candidate searches there are zero cancellations, zero saved expansions, and zero paired-minus-split work or utility differences at all caps and cohorts. Dominated terminal-to-terminal comparisons are screened out; no common nonterminal state was reached within the searched frontier. This is a negative mechanism result, not a speedup.
2. The risk-aware late policy achieves prior-weighted acceptance 10.4167% ordinary, 0% shifted, and 5.5556% scarce at either cap. Minimum accepted net margins are +0.61572 ordinary and +1.02158 scarce. Its mean incremental expected false recommendations are -0.01632 ordinary, 0 shifted, and -0.00741 scarce. No certified actor increases root-conditional expected false recommendations anywhere in evaluation. The legacy late actor raises risk on two ordinary and two shifted roots, with positive cohort priced deltas +0.04059 and +0.02200. Risk-aware late loses -0.02778, -0.10592 and -0.14109 priced utility across the cohorts, winning 2/12, 0/12 and 1/12 roots. It is closer to break-even on ordinary than earlier search, but the fresh roots preclude an across-version causal claim.
3. The tail policy accepts 5.5556% prior-weighted trajectories in scarcity, with minimum net margin +0.45029; broad accepts none. Forced-incumbent controls preserve raw outcome aggregates. All calibration candidates lose mean priced utility; both caps select no optional search. Keep the budget rule.

## Interpretation and follow-up

Exact-state cancellation requires identical public sufficient states, including the experimental budget and posterior mask. Different investigative actions commonly change both, and bounded search can end before reconvergence. Avoid treating the fixture success as evidence that the mechanism pays for itself. The next target is an equivalence quotient for continuations with distinct but decision-irrelevant fields, paired with a cheap final-stage risk certificate and explicit rejected-search cost. Any coarsening must preserve the incumbent action, conditional reward/risk, and successor equivalence; prove these conditions with exhaustive finite replay before testing performance. A separate diagnostic should count complete reachable counterfactual reconvergences outside the bounded actor, distinguishing absence of overlap from failure to reach it.

No provider calls, GPU training, native DataGoblin weight update, or downstream training admission occurred. The benchmark certifies finite-model expected incremental risk and conditional utility, not individual-world or real-world safety. The full artifacts are time-limited; the printed aggregate is preserved beside this report.
