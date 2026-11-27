# DISTILL-13 hosted history-memory result

Tested commit e4f5436e2283e38c84b512e21466a0c6dc777b86. [Workflow 46073089166](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46073089166) and full offline exam checks passed. [Artifact 15234091040](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46073089166/artifacts/15234091040). Two targeted tests passed. The initial failed run had an aggregation bug; the corrected run retained the frozen protocol.

66 disjoint public root specifications, 470,588 enumerated literal reachable states, 35,732 quotient classes (sum of within-root classes), 9,325 selected public records, 3,168 verified triplets. Eight 2,612-parameter RNN fits, each 648 updates; 5,184 total. 1,152 training triplets; 1,728 evaluation triplets across 36 held-out roots. Training only on the synthetic public histories and exact labels.

Held-out root/seed macro means:

| Arm | Triplet accuracy | Positive distance | Negative distance | Belief TV | Action accuracy | Value MAE | Risk MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sorted no pair | .683738 | .100357 | .186121 | .389604 | .401924 | 3.360873 | .111011 |
| Sorted pair | .690972 | .053657 | .110270 | .386204 | .393451 | 3.462882 | .107264 |
| History no pair | .849537 | .078064 | .326519 | .332141 | .600055 | 2.469769 | .125369 |
| History pair | .868056 | .029520 | .245266 | .318303 | .582379 | 2.436463 | .114451 |

Pair loss improves ranking by 1.85 percentage points and reduces positive distance by 62.2%, also shrinks negative distances. Action accuracy falls 1.77 points. Scarce triplet accuracy improves from .744792 to .782118 but action accuracy remains .388197. Temporal order yields the larger effect than the pair loss. These are descriptive paired means, not confidence intervals or online policy gains. The selected triplets oversample nontrivial quotient classes and are not an estimate of on-policy state prevalence.

Retain the auxiliary representation result. No learned certificate, online actor promotion, native DataGoblin model update or training admission. No provider calls. Next test bounded within-root retrieval with exact verification and explicit exact-hash baseline.
