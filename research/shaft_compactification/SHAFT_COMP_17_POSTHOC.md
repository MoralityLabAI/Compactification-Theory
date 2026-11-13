# SHAFT-COMP-17 post-hoc 16-output capacity control

**Declared after viewing the registered COMP-17 result.** This is an exploratory ablation. Its results cannot replace the preregistered comparison or the original permutation p-value.

The primary experiment compares a four-output linear probe trained on one-step behavioral signatures against a sixteen-output probe trained on sixteen two-step final-output signatures.

To separate output dimension from history content, fit a **sixteen-output duplicate-immediate control** with the same number of parameters, fitting states, BCE objective, optimizer, 400-step budget and learning rate as the two-step probe. Its targets are the native four one-step response bits repeated four times, in 4×4 query-pair ordering. Use independent probe initialization seed model_seed+92000; the history-aware probe uses model_seed+91000.

Preserve the original models and state pairs; reproduce the five-token diagnostic and 32-token outcome input streams exactly. Evaluate both readouts through identical recurrent Jacobian cocycles and equal eligibility. Do not tune thresholds, feature orientation or eligibility on heldout labels.

Report heldout AUC overall, per family and per model, and label all findings *post-hoc*. The control addresses trivial output duplication and parameter count. Independent probe initialization and fit quality remain possible confounders. Nothing here concerns covert persistence or reconstruction of erased model weights.
