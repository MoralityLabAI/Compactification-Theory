# SA-02 first registered baseline receipt

Source head d68567237775245f133d1b8ed27a00a151c9b4c3. [Native baseline run 45325139838](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45325139838), job 138212807805, completed successfully. Lean 4.19.0. [Archived full source, logs and baseline JSON](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45325139838/artifacts/14888913045), ZIP SHA256 6058fc68d58376c66a31e873cfde0ff399986b9f98a070fc54191911dfa09eca.

Native: train 32 families / 160 axiom checks; calibration 12 / 60; depth test 20 / 100; mechanism test 16 / 64. All 80 families, 384 axiom checks passed. The 128-read gate asserted all 320 cases correct under each of the five policies. No wrong verdicts permitted by the independent checks. This is a generated graph acquisition task with native source corroboration, not a general dependency extraction result.

Held-out verdict totals copied from the first registered job log:

| Split | Budget | Policy | Correct | Abstained | Axiom axis | Semantic axis | Joint families |
|---|---:|---|---:|---:|---:|---:|---:|
| test_depth | 16 | random | 0/80 | 80 | 0 | 0 | 0/20 |
| test_depth | 16 | lexical | 3/80 | 77 | 3 | 0 | 0/20 |
| test_depth | 16 | proof_first | 0/80 | 80 | 0 | 0 | 0/20 |
| test_depth | 16 | semantic_first | 0/80 | 80 | 0 | 0 | 0/20 |
| test_depth | 16 | alternating | 0/80 | 80 | 0 | 0 | 0/20 |
| test_depth | 32 | random | 5/80 | 75 | 5 | 0 | 0/20 |
| test_depth | 32 | lexical | 17/80 | 63 | 21 | 1 | 0/20 |
| test_depth | 32 | proof_first | 40/80 | 40 | 76 | 0 | 0/20 |
| test_depth | 32 | semantic_first | 0/80 | 80 | 0 | 0 | 0/20 |
| test_depth | 32 | alternating | 0/80 | 80 | 0 | 0 | 0/20 |
| test_depth | 64 | random | 36/80 | 44 | 69 | 1 | 0/20 |
| test_depth | 64 | lexical | 30/80 | 50 | 44 | 9 | 0/20 |
| test_depth | 64 | proof_first | 40/80 | 40 | 80 | 0 | 0/20 |
| test_depth | 64 | semantic_first | 20/80 | 60 | 0 | 40 | 0/20 |
| test_depth | 64 | alternating | 40/80 | 40 | 76 | 0 | 0/20 |
| test_mechanism | 16 | random | 12/64 | 52 | 21 | 0 | 0/16 |
| test_mechanism | 16 | lexical | 16/64 | 48 | 19 | 4 | 0/16 |
| test_mechanism | 16 | proof_first | 30/64 | 34 | 62 | 0 | 0/16 |
| test_mechanism | 16 | semantic_first | 0/64 | 64 | 0 | 0 | 0/16 |
| test_mechanism | 16 | alternating | 8/64 | 56 | 16 | 0 | 0/16 |
| test_mechanism | 32 | random | 34/64 | 30 | 55 | 12 | 0/16 |
| test_mechanism | 32 | lexical | 32/64 | 32 | 41 | 20 | 0/16 |
| test_mechanism | 32 | proof_first | 33/64 | 31 | 64 | 5 | 0/16 |
| test_mechanism | 32 | semantic_first | 31/64 | 33 | 5 | 62 | 0/16 |
| test_mechanism | 32 | alternating | 31/64 | 33 | 62 | 5 | 0/16 |
| test_mechanism | 64 | random | 64/64 | 0 | 64 | 64 | 16/16 |
| test_mechanism | 64 | lexical | 64/64 | 0 | 64 | 64 | 16/16 |
| test_mechanism | 64 | proof_first | 64/64 | 0 | 64 | 64 | 16/16 |
| test_mechanism | 64 | semantic_first | 64/64 | 0 | 64 | 64 | 16/16 |
| test_mechanism | 64 | alternating | 64/64 | 0 | 64 | 64 | 16/16 |

At 32 reads proof-first leads the deeper split (40/80), all correct verdicts on extra-axiom arms because it has zero completed semantic axes. At 64 reads it remains 40/80. Semantic-first reaches 20/80 on that split, while all methods saturate the unseen-mechanism split at 64. This exposes a scheduling problem, with distinct completion costs for proof dependence and correspondence. Results do not yet establish an advantage of a learned policy; proof-first is the strongest observed deeper-split control at 32 reads. The later oracle diagnostic is separate from the first registered run. No test oracle trajectories may be used for training.

Next preregistration must compare a target-only semantic refutation search and adaptive proof-frontier search, since the current semantic-first control also resolves the source expression before finishing the target. This is a post-hoc observation and does not change the registered results.
