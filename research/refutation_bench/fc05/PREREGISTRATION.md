# FC-05 preregistration: model-proposed, kernel-checkable positive certificates

**Date: 2026-11-13.** The protocol is committed before FC-05 corpus generation,
model training, checkpoint evaluation or test inspection. Amendments after the
first evaluation must preserve the original run and be called exploratory.

## Objective and falsifiable claim

FC-04 releases a refutation only after exact checking of its proposed Boolean
countermodel. Its positive verdicts are provisional. FC-05 tests a symmetric
release rule: *an entailment is certified only when a bounded model-guided
forward proof is independently checked*. The model proposes an ordered trace
of rule IDs; a separate checker validates every inference from the supplied
premises and a Lean 4.19.0 kernel run compiles all accepted positive proofs.
The model may abstain if the trace fails. No oracle may repair its proposed
trace during evaluation.

This remains bounded propositional Horn reasoning. It does not establish
correctness of any real research manuscript without an independent verified
semantic encoding.

## Frozen data, neural trunk and calibration

- FC-02 training generator commit
  `7aca2bc3fc5586c5b7997fba868bd179db761121`,
  canonical corpus SHA-256
  `8e86a41dd93ff162072799332ebbef4c7134951798eb4af7ebb4ab044bebd56f`.
- LR-01 tied graph model `LoopReach(4,width=24)`, **11,521 base
  parameters**, loaded from the FC-04 model-reproduction recipe: seeds
  **17,29**, 350 AdamW updates, batch 48, lr 0.002, weight decay 0.01,
  grad clip 1, two CPU threads. Pretrained external checkpoints are
  excluded. The same train/dev partition as FC-04 is used.
- Inference checkpoints **4,6,8,10,12**. Threshold grid
  `[0.55, 0.70, 0.85, 0.93, 0.98]` and the FC-04 validation-only
  utility determine θ separately for each seed; no FC-05 data are
  available during training or calibration.
- **Positive proof-head training** uses FC-02 train *established* tasks
  only. A deterministic shortest-action BFS supplies one trace over
  legal forward-chaining Horn rules; BFS is strictly training-only.
  Each prefix of that trace gives a supervised legal-next-rule state.
  Freeze the recurrent trunk while fitting a two-layer rule readout
  with width **48**, GELU activation and one scalar score. Inputs are
  concatenated 24-dimensional current rule, head-atom and averaged
  premise-atom embeddings plus a 24-dimensional goal-atom embedding.
  Train head with CE over currently legal rules; **250 AdamW updates**,
  batch 48, lr 0.003, weight decay 0.01, seed as above.
  Permuting names and shuffled source rule order prevent position
  decoding. Training labels and BFS traces never appear in model input.
- During inference, the learned head ranks currently legal rules.
  At most **10 legal rule applications** are attempted per positive
  proof. After each action, the known set and legal mask change and
  the original frozen recurrent cell is re-evaluated for the chosen
  goal with four tied passes. The model decides rule order; neither
  the verifier nor an oracle may choose an action for it.

## Real early stopping and safety

At each FC-04 recurrent checkpoint, first attempt exact negative
countermodel verification. On success, immediately release
`refuted_certified`. If goal confidence has exceeded the validation
threshold at two consecutive checkpoints, attempt the 10-step positive
proof search. Only if its **model-selected action trace** passes the
independent verifier may the system release `established_certified`.
On proof-search failure it **abstains**, with an explicit failure
reason; it may not invoke the exact solver to synthesize missing steps.
If neither criterion fires by 12 passes, it abstains.

A successful positive certificate includes rule indices in their
actual selected order and a generated Lean proof term. The verifier
replays each step: rule exists; antecedents are already known; the
consequent is newly derived; and the target goal is ultimately known.
The Lean proof term is rendered from that **same checked trace**, and
Lean 4.19.0 compiles every accepted term as a separate parameterized
theorem. The exact checker is the authority for the local propositional
logic; Lean compilation is a second independent proof-kernel check.
The student never emits an unverified theorem certificate.

Report all computation: recurrent passes used for the FC-04 stopping
decision; additional tied-cell calls from proof-step proposals;
rule actions attempted; Python checker calls/CPU time; total elapsed
CPU time; and mean proof trace plus certificate size in bytes. Training
BFS, reference Lean proofs and source gold truth must not enter the
inference path.

## Fresh family-held-out test

Generate **384 fresh** cases, in `test_same` at depth 2–4 and
`test_deep` at depth 6–8, with 192 each. All four entirely new graph
families `split_relay`, `cross_gate`, `detour_join` and
`layered_cycle` contribute exactly 24 valid and 24 invalid cases to
each split. Root seed **20261113**, offsets **81011** and **92033**.
Use deterministic attempt-index seeding, 6–12 atoms, <=20 Horn rules,
and enforce disjoint conservative WL fingerprints against **all**
FC-02 (1536), FC-03 (384), FC-04 (384) and FC-05 records.
No earlier positive or negative test tasks may join the training set.

Every source label is independently confirmed by Horn closure and
exhaustive Boolean truth tables. Every negative source task has an exact
countermodel. Every positive source task has a reference Lean theorem
compiled in source CI. All model-generated positive proofs must be
compiled again from their own action traces in the dependent **evaluation**
job.

Training and stopping calibration run in a GitHub Actions job that can
check out only the pinned FC-02 source. Freeze and SHA256-check trained
trunk and head weights plus the threshold ledger. A dependent evaluation
job downloads those frozen artifacts and **then** checks out the pinned
FC-05 generator and emits all per-case results. CI may be repaired for
mechanical failures before any test metrics appear, but those repairs
must be documented.

## Registered baselines and measures

All policies use the same frozen FC-04 recurrent positive/negative
eligibility thresholds:

- `learned_trace`: train-only learned rule priority, <=10 actions.
- `random_legal`: deterministic uniformly random choice among
  currently applicable novel rules, <=10 actions, seed 2917 + a stable
  hash of the task ID.
- `immediate_goal_random`: take an applicable goal-producing rule
  immediately, otherwise random legal choice, same rule budget.
- `exact_oracle`: exact forward-chaining certificate generation from
  the formal graph, clearly identified as the non-neural 100%-correct
  reference and **excluded from the neural inference comparison**.
- `fc04_provisional`: original unverified positive releases and
  exact negative certificates at identical stopping stages, as a
  non-certified coverage benchmark.

Per family/depth/seed report **positive certified recall on 96 valid**,
**negative certified recall on 96 invalid**, combined certified coverage,
provisional claims suppressed, abstentions, zero-false-certificate checks,
model-forward plus verifier CPU time, proof-head calls, recurrent
cell invocations, rule-action counts, and average/p95 certificate bytes.
Record all failed traces. Head and trunk parameter counts are separate.
Use paired per-instance results for learned versus random and
goal-first baselines, and report differences with numerator/denominator.

**Primary success criterion:** The new system must produce at least one
independently verified, Lean-compiled positive proof on *each* fresh test
split for *both* seeds, while never releasing a false positive or false
negative certificate. Report selective coverage and compute even if this
criterion is met; learning advantage requires higher positive certificate
recall than both non-oracle baselines on both splits and seeds. Otherwise
the proof channel is operational but the evidence for learned proof
selection is inconclusive or negative.

The new test is public and preregistered, not a private sealed benchmark.
The finite Horn closure oracle has 100% exact performance on this
fragment. No result from this experiment decides RH, Collatz, Fermat or
P versus NP.
