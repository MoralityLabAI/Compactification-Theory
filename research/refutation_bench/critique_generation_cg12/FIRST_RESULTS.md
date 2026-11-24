# CG-12 first native results

[Run 45377072030](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45377072030), job 138386079764, source head ec523f65300bf283ec3b54d0830a3129e366401f. Pinned original MathCan 6d02b611c7358216379cb36952512a104e0bdb8e. Native gate PASS: four unchanged source modules, nine original theorem roots, eighteen matched hypothetical foundation-policy cases. Independent graph closure, native instrumented root reports and original imported module axiom sets agree for every selected root.

| Original theorem | Axiom set | Graph nodes | Summary boundaries |
|---|---|---:|---:|
| MathCan.ObservableQuotient.factor_iff_fiber | Classical.choice | 15 | 13 |
| MathCan.ObservableQuotient.factor_unique | Quot.sound | 11 | 10 |
| MathCan.AuthorizationQuotient.classifier_exists_iff_fiber | Classical.choice | 16 | 14 |
| MathCan.AuthorizationQuotient.collision_forces_error | none | 14 | 13 |
| MathCan.AuthorizationQuotient.coarse_compatible_implies_fine | none | 5 | 3 |
| MathCan.NOR19StreamedVerifier.streamed_exactly_iff_materialized_equal | propext | 53 | 47 |
| MathCan.NOR19StreamedVerifier.streamed_rejection_iff_unequal | propext | 64 | 57 |
| MathCan.MatchedStepMargin.positive_margin_stable | Quot.sound, propext | 62 | 60 |
| MathCan.MatchedStepMargin.negative_margin_stable | Quot.sound, propext | 65 | 63 |

Under the standard classical basis all nine proofs pass. Under the hypothetical basis excluding Classical.choice, seven pass and two have verified excluded-axiom dependency certificates: factor_iff_fiber and classifier_exists_iff_fiber. This establishes a property of the particular source proofs relative to the policy, not that their statements are false or lack constructive alternatives. The original proof bytes were preserved; the instrumentation copy adds an import and command, so only axiom-set equivalence is established, not term identity. The large fraction of imported summary boundaries limits the internal resolution of paths across modules.

[Artifact 14911971530](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45377072030/artifacts/14911971530), ZIP SHA256 784eafd4a1a1d3ab9c55910ffaba2b034a9c734f3fcc8506a187163a2ab5617c. Root receipts include original/instrumented source hashes and graph hashes. All cases are development-only. No learned model evaluated.