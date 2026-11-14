# FC-12 typed Lean tactic campaign | first run, 2026-11-14

FC-12 completed the registered original-MathCan held-out experiment. The learned
four-pass, 62,401-parameter TypedLoop **matched** the fixed Lean-compiler
symbolic baseline on proof coverage and consumed more wall time per certified
original theorem. The registered learning and compute criteria both failed.

## Frozen design and training gate

- [Preregistration](../fc12/PREREGISTRATION.md) frozen at `d6fae0ddb9ce60036cff09e5ec93820ed3879af8`.
- [Train-only pre-test amendment](../fc12/PRETEST_AMENDMENT_A.md) preserved the initial exact-coverage failure; it permitted local hypothesis application before held-out selection.
- Pinned untouched source: `patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189` and Lean4.35.0.
- 27 training theorems, 51 true single-goal Lean compiler states; 961 candidate compiler probes.
- Compiler-observed progressing training examples: `exact_local` 1, `rw_local` 13, `rcases_local` 1, `induction_list` 30. These were scratch transitions, not completed certified proofs.
- Train-state SHA256 `403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0`; effect-semantic SHA256 `bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec`.
- Exact train coverage [CI 43547041634](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43547041634), [model freeze 43547762837](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43547762837).
- After the training freeze, the 16 held-out original theorems came from Hodge, TOEDressedMatter, TOELorentzBivector, and TOETensorCompletion, pinned at source revision `b9b2f48fc4cc3d921ddac5d90ca7c12995ff7f9b`.

## Held-out scores

| Seed | Learned theorem certificates | Native symbolic theorem certificates | Learned wall ms / certificate | Native wall ms / certificate |
| --- | ---: | ---: | ---: | ---: |
| 17 | 5/16 | 5/16 | 40,273.84 | 32,827.72 |
| 29 | 5/16 | 5/16 | 33,283.64 | 32,056.10 |

All ten learned certified theorem instances coincide with the corresponding
native certificates. No learned proof used the newly parameterized actions.
The oracle proofs preserved from original MathCan source compiled 16/16.
Positive model proofs were separately compiled fully without `sorry`.

The registered native comparator used the exact same instantiated tactic
grammar in fixed family order, up to 3 goal transitions and 12 candidate
compiler probes. It received each model's actual time allowance as a scalar.
Its process-level deadline can overrun between subprocess calls; logs show
91.66 ms total overrun for seed17 and 5,648.99 ms for seed29. Two positive
native certificates exceeded the deadline by under 30 ms each, so an exact
hard-deadline clipping rule would make a **post-hoc** 5/16 versus 4/16
comparison per seed. This is sensitive to compiler-runtime jitter and is
excluded from the preregistered main score.

The separate 20-second source-only native ceiling also certified 5/16:
[CI 43548372735](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43548372735).

## Independent replay

[Original paired run CI 43548536370](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43548536370)
stored all model/native attempts and original theorem compiler outputs.
[Separate audit CI 43550665086](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43550665086)
regenerated all 64 model/native chosen proofs from the immutable original
MathCan source through a separate renderer, then freshly compiled them with
Lean4.35. All 64 reported success/failure verdicts reproduced, with zero
disagreements.

The [independent replay receipt](fc12_independent_replay_audit_20261114.json)
pins the original scored report SHA256
`9a9f106e41487ae250f1880bb010e4c0ebeeb8fd50b94a204e6ec52b90793d0a`
and replayed case JSONL SHA256
`d6a38aa336e2b9aa21fad2395ad8629077e28858aa3d172535d217f4e5cfd8e6`.
The raw per-case compiler audit is preserved as an artifact of the replay CI.

The detailed [model-side run report](https://github.com/MoralityLabAI/LoopedTransformers/blob/feat/lr01-fc12-typed-tactic-policy/experiments/refutation_transfer/FC12_FIRST_RUN_20261114.md)
retains the action choices and individual theorem names.
