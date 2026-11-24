# RB-CG-02: local compositional critique execution

Registered after CG-01 failed both widths on [run 45328809443](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45328809443). All learned decoders had zero complete dependency paths on deeper proofs. The failure is retained in CG-01 FIRST_RESULTS.md.

## Hypothesis and explicit boundary

A reusable transition cell with explicit local declaration binding can extrapolate proof length. It can generate a critique certificate by composing small operations until the proof leaf and all Boolean obligations are resolved. This is a hybrid, grammar-restricted program executor; it is not a free-form mathematical reasoner. The syntax frontend extracts declaration names, immediate textual RHS identifier if present, candidate declarations, explicit theorem type identifiers, and Boolean ASTs. It does not traverse dependencies, classify the root as clean/tainted, calculate a truth table, or select counterexamples. The output is the model-executed chain and valuation, with separate exact certificate verification. Bridge lookup and exact type identity remain deterministic. Each primitive has an exact symbolic control.

## Prospective data

Use unchanged CG-01 generator with fresh seeds 101801/101802/101803/101804 and 128/32/48/48 families. Train on fresh train only, eight arms per family; report calibration and both prospective test splits separately. Increase depth transfer aliases to 7–11 as in CG-01 and keep mechanism holdout vacuity/unrelated. Compile all 256 fresh families and every original native witness; require disjoint raw source hashes against the earlier 384. Gold remains public synthetic evaluation. No selection on test.

## Tied transitions

1. A small terminal classifier sees local keyword/shape features and predicts continue, clean, or axiom.
2. A shared identifier projection scores the local bare RHS identifier against all preceding theorem/axiom declaration names using the 32 raw identifier bits. This is explicit local binding: no exact equality bit, candidate target label, or hidden reachability is supplied. The model's selected pointer determines the next line. A zero-training dot-product baseline is included to expose the strength of this representation. Read the next declaration only after the pointer is chosen. Stop at predicted terminal or fail after 64 steps; no semantic repair.
3. A tied Boolean MLP receives one-hot not/and/or opcode and Boolean operands. Execute its predictions recursively on the syntax AST with explicit variable binding and deterministic call expansion. Its training rows are the complete primitive truth table, not held-out expression labels. Produce the first predicted falsifying valuation. If none, use explicit matching type or named bridge, else UNLINKED. A wrong gate prediction may produce a rejected semantic certificate.

Train two seeds 61/67 for 120 fixed AdamW full-batch updates (lr 0.01) on a capped deterministic sample of 2048 local training declarations and all gate rows. Projection width 16, terminal/gate hidden width 16. Record parameter counts, weights, hashes, loss and primitive accuracy. Evaluate learned chain+learned Boolean, learned chain+exact Boolean, exact chain+learned Boolean, untrained bit dot-product chain+exact Boolean, and exact parser reference. No ground truth or parsed verifier verdict enters the learned policy; verifier can inspect the entire raw source only after generation.

## Gates

Primary architectural gate: both learned seeds produce at least 95% independently verified complete certificates on both fresh held-out splits, with at most 5% proposed false accusations among supported controls. Report dependency/semantic axes, supported/unlinked and counterexample coverage, actual output lengths and local steps. Symbolic comparison is mandatory. This is a feasibility test of compositional execution, with no claim that a learned primitive is needed for a decidable grammar. The untrained exact-bit baseline may make learning unnecessary. Failure must be recorded. A pass motivates more difficult nontrivial local actions (unknown identifier meanings, branching dependencies, theorem/type normalization and native verifier interactions), not scaling this trivial binding task to millions of parameters.
