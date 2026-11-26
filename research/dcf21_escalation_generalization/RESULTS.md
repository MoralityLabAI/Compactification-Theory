# DCF-21 results: escalate-on-abstention outside the DCF-20 world

The run used 444 fresh instances of the ten DCF-14 graph families, at n = 8 to 256, with 407 verified by an independent exact path. Budgets were 2^10, 2^14 and 2^18 counted operations, penalties 4 and 9, giving 438 conditions (family × n × budget × penalty). The protocol and code were committed before any instance was generated (`PRE_RUN_SHA256`).

## Gates

| Gate | Result |
|---|---|
| F1: BP abstention captures ≥ 80% of BP/exact decision changes | **FAIL**: recall .727 (202 of 278); precision .166 (202 of 1,218 abstentions) |
| E1: escalate beats exact-else-mini-bucket at λ = .05 | **FAIL**: 45 wins, 365 losses, 28 ties |
| E2: escalate beats the best alternative at some declared λ | **FAIL**: 0 wins at λ = .05, .2 and .5 (384, 394 and 402 losses) |

## Policies (mean over conditions)

| λ | escalate | exact else mbe | exact else BP | exact or abstain | always BP |
|---:|---:|---:|---:|---:|---:|
| .05 | .2903 | **.3188** | .3139 | .2859 | .2558 |
| .2 | .2157 | **.2864** | .2825 | .2714 | .1823 |
| .5 | .0665 | .2216 | .2198 | **.2423** | .0354 |

Mean counted operations: escalate 13,834, always BP 13,223, exact else mbe 6,792, exact else BP 4,341, exact or abstain 1,965. Expected utility before cost: escalate .315, exact else mbe .330. Escalation loses in every family at every λ.

Flag recall by family:

| Family | Recall | Decision changes |
|---|---:|---:|
| xor_noiseless | 1.00 | 71 |
| tree | .82 | 17 |
| grid | .82 | 28 |
| dense | .71 | 17 |
| sparse | .70 | 33 |
| chain | .69 | 13 |
| hyper | .57 | 28 |
| ktree | .57 | 42 |
| star | .40 | 25 |
| weak_dense | 0 | 4 |

## Secondary: DCF-16 held-out records (reused data, descriptive)

| Predictor | Verified EU | Mean ops |
|---|---:|---:|
| EvalMath symbolic (cheapest exact within budget, else mbe) | **.352** | 8,009 |
| escalate-on-abstention | .300 | 15,473 |
| always BP | .229 | 14,927 |

## Why it fails here

DCF-20B's win rested on two premises: the approximate operator is cheap relative to exact, and its failures announce themselves. Both fail on the DCF-14 families.

- **Cost order is inverted.** In counted operations, loopy BP spends up to its iteration budget on every query. Exact elimination on these graphs is usually cheaper, and its cost is known symbolically before execution. Running BP first therefore doubles the work of policies that go straight to exact when it fits.
- **Errors are less self-flagged.** On the mesh's aligned XOR queries, BP failed by collapsing to p ≈ .5. On general graphs 27% of BP/exact decision changes are confident BP errors (BP acts and is wrong in the expected-utility sense), which an abstention flag cannot catch. Abstentions are also mostly harmless: 83% of them are cases where exact abstains too.

## Verdict

Escalate-on-abstention is not a general routing rule. It wins only where the approximate operator is much cheaper than exact and fails by under-confidence, as in the DCF-20 mesh at high compute prices. Where exact cost is symbolically known and often small, the DCF-14 rule (exact when it fits the budget, otherwise mini-bucket) remains the strongest policy measured. The series' durable routing principle is to price operators symbolically before running them, not to probe with a cheap operator first.
