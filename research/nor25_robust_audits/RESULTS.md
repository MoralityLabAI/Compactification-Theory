# NOR-25 first prospective outcome: registered PASS, with measured robustness cost

**First held-out run:** [GitHub Actions #43885131079](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43885131079). Its source-only fixture was written and SHA-locked before constructing NOR-25 attack sets or audit schedules. The first preregistration commit is [da075255](https://github.com/MoralityLabAI/Compactification-Theory/commit/da075255e2d5e188d5a84f3febbcc7b162d08b79). See [registration](REGISTRATION.md) for all four fixed primary gates.

The first full calculation and separate-process independent replay both passed. The replay reconstructed 12 original source identities, 15 fixed blind attack sets per source, all pseudorandom hybrid plans, theoretical worst-case floors and **4,423,680** paired detection decisions. The original output was preserved with exact raw file hashes in [results/nor25_original_first_receipt.json](results/nor25_original_first_receipt.json); its [original CI artifact, ID 14233747860](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43885131079), stores the source-only fixture and complete outcome JSON.

## Primary registered k=16, radius=4

| Policy | In-template weighted detection | Off-template fixed blind attack |
| --- | ---: | ---: |
| Uniform | 6.22% | 6.15% |
| Shuffled prior | 5.99% | 6.00% |
| Static trained prior | 38.92% | 0.00% |
| Sequential posterior | 39.11% | 0.00% |
| Hybrid, 4 uniform / 12 targeted | 33.82% | 1.56% |
| **Hybrid, 8 uniform / 8 targeted** | **27.03%** | **3.17%** |
| Hybrid, 12 uniform / 4 targeted | 17.90% | 4.32% |
| Privileged support oracle | 47.97% | 8.39% |

The registered hybrid_50 policy improved on uniform for **11/12 source pairs**, with +0.208194986979 absolute mean in-template gain. Off-template it beat static ranking for **12/12 source pairs**, with 0.0316569 mean detection versus static's zero. All four registered primary gates passed.

Measured cost: hybrid_50 retained approximately **69.5%** of static prior's weighted in-template detection (27.03% versus 38.92%). Its off-template detection was roughly half uniform's 6.15%. This is a tradeoff under a fixed witness budget, and the sampling stream is public for reproducibility.

## Source-level paired primary effects

The source represents the independent unit. The 1,024 audit challenge ordinals within a source share that source's fixed blind attack positions.

| Replicate | Degree | Source family | Hybrid_50 minus uniform, pp | Hybrid_50 minus static, pp | Off-template hybrid_50 |
| --- | ---: | --- | ---: | ---: | ---: |
| 4 | 512 | dense | -3.69 | +4.07 | 3.32% |
| 4 | 512 | sparse | +15.63 | -12.59 | 4.79% |
| 4 | 512 | rational | +20.99 | -16.01 | 4.49% |
| 4 | 1024 | dense | +18.01 | -15.55 | 1.27% |
| 4 | 1024 | sparse | +17.74 | -13.13 | 1.76% |
| 4 | 1024 | rational | +19.13 | -13.91 | 2.54% |
| 5 | 512 | dense | +16.64 | -11.04 | 3.81% |
| 5 | 512 | sparse | +14.77 | -13.79 | 4.39% |
| 5 | 512 | rational | +16.06 | -13.67 | 3.42% |
| 5 | 1024 | dense | +77.47 | -8.44 | 2.93% |
| 5 | 1024 | sparse | +18.45 | -15.16 | 3.13% |
| 5 | 1024 | rational | +18.64 | -13.43 | 2.15% |

One unusually large +77.47 pp source effect contributes to the mean. The median source gain is **+17.88 pp**. Excluding that single maximum source as a retrospective sensitivity check gives about **+15.67 pp** over the remaining eleven; their paired win count is ten of eleven. The registration retained all twelve sources.

## Per-family observation

At k=16, radius=4, hybrid_50 detected 36.91% of tail attacks, 33.19% of magnitude attacks, 9.94% of lag-window attacks, and 3.21% of uniform-background attacks. The static prior detected 54.59%, 51.39%, 8.33% and **0.00%**, respectively, on these same source-world constructions. Hybrid allocation therefore restores some coverage against both the explicitly off-template attack and the declared uniform background family.

## Exact worst-case blind floor

For a fixed radius-four attack chosen before the audit seed, the hybrid_50 two-stage sampling design guarantees at least

| Query output length | Exact minimum detection probability |
| ---: | ---: |
| 769 coefficients | 4.147204% |
| 1537 coefficients | 2.078527% |

These are theoretical probabilities for idealized independent uniform sampling, not guaranteed minimum frequencies over 1,024 trials. Each source's finite empirical off-template frequency can fall below its exact probability floor by sampling fluctuation. A realized-schedule-informed adversary again evaded every partial audit, as preregistered.

## Original exact byte receipts

- 12-source public-only JSON: **2,187,225 bytes**, SHA-256 \`39f69654ed17b970088f405d8b8db171b5042459307d923fb3c7cc32b1b3b6bf\`.
- Full original outcome: **859,176 bytes**, SHA-256 \`901bd3b8ac9ac3335fc5e26cd69ee8d0ea9c81d7ba8fae3a508c7e9bcc14958f\`.
- Evidence archive: Actions #43885131079, artifact ID **14233747860**. The two full original files remain obtainable as the CI artifact until retention expiry and are deterministically regenerable by the source-pinned experiment.
- The separate-process auditor passed on the original artifact independently of the first evaluator, with exact source and attack replay.

**Scope.** Synthetic fixed blind attacks, an independent source-only prior, and an authentic target-dependent exact-coefficient witness. Reported challenge averages are correlated within twelve source units. The fully informed adversary is a separate threat model with a zero-detection counterexample. A higher posterior in-template mean was observed, and the incremental adaptation effect remains small relative to the cost of global exploration.
