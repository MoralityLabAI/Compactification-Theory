# COG-10: exact local information and generalization gap

Executed at 49a443538a70c6ac671c1e518b51a3e65056e202. [Passing CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45741392764). The independent JavaScript enumeration and Python test both checked 65 paired depth witnesses (d=0 through 64). The source-pinned corpus pass replayed 256 roots, 2,453 distinct teacher-reached states and 66,893 candidate checks. Full depth 0..16 output is in [the artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45741392764/artifacts/15081890327), expires 2026-12-26T02:17:25Z. The adjacent JSON file preserves the six selected depth readouts printed in the workflow log and the complete output hash.

## Deeper partition (33,237 candidates, 2,117 positive)

| Passes | Minimum empirical signature-only errors | Training-majority errors | Unseen signatures (examples) | Bounded goal propagation false negatives |
|---:|---:|---:|---:|---:|
| 0 | 158 | 2003 | 25857 | 2051 |
| 1 | 0 | 2073 | 28592 | 1792 |
| 2 | 0 | 2085 | 28726 | 1322 |
| 4 | 0 | 2106 | 28864 | 234 |
| 8 | 0 | 2116 | 28924 | 3 |
| 16 | 0 | 2116 | 28924 | 0 |

The exact feature signature has no conflicting labels on the observed deeper split after one pass, but this oracle representability result uses those labels and does not imply transfer. At depth four, the training majority decoder sees 28,864 of the 33,237 examples as unseen signatures and makes 2,106 errors. At depth sixteen it misses 2,116 positives (a constant negative baseline misses 2,117). Increasing depth refines signature identity, encouraging a lookup decoder to memorize. Conversely, the shared Boolean goal-propagation rule reaches every true ancestry label by depth sixteen, with zero false positives throughout. At depth four, 234 useful candidates remain beyond that bounded reach, spread across 44 roots; depth eight leaves three across three roots. The symbolic propagation is an algorithmic control and requires the public graph. The table measures a finite sample, not a fitted transformer.

The locality construction shows why full ancestry must be charged as part of the shared feature extractor in COG-07. For a first-decision rule-local head with d message passes and no other nonlocal information, there is a pair of graphs with opposite labels for the same ready rule and identical radius-d signatures. Its 50% pairwise classification lower bound does not constrain later verifier decisions with carry or global attention. The next neural gym should directly supervise the bounded propagation state at each pass and test transfer to longer dependency chains with equal neural work. This report supplies an explicit mathematical proof and finite replay, not a Lean-checked theorem or a model parameter lower bound.
