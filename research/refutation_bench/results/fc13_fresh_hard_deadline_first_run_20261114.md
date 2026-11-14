# FC-13: frozen first-run original Lean transfer | November 14, 2026

**Registered outcome: negative.** The frozen FC-12 recurrent TypedLoop, transferred
without retraining to 16 source-family held-out original MathCan theorems,
certified fewer proofs than native search in seed 17, and matched native
in seed 29.

## Pinned experiment

- [FC-13 preregistration](../fc13/PREREGISTRATION.md) committed in `309fed65caf67da46db8cb981757a9326bc59fbc` before held-out evaluation.
- Exact original MathCan source `c84e2025334e9a967c1297425a490f98d5b3d189`, Lean 4.35.0.
- Source selector: first four original theorem declarations each from
  HookMultiTick, Settlement, TOEJordan, TOENonAbelian.
- Original-source proof oracle: **16/16** compiled. The instrumented selector
  produced 18 real compiler goal states; source [CI 43554159459](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43554159459);
  goal JSONL SHA256 `3df327272bc4768eac2fdfaa60a2cdbead06443e48153781683ee03944568554`.
- Frozen FC-12 model training run `43547762837`, 62,401 parameters per seed,
  twelve local-context tactic families, 51 original train goal states, 961 actual
  Lean-probed action examples. The same train-effect SHA256
  `bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec`
  was used.
- **Both** controllers received 20,000 ms per theorem, a twelve-Lean-subprocess
  total budget including scratch and final compiler calls, and up to three
  successfully advancing transitions. The native baseline used the same generated
  scripts in the frozen family order. Every released proof required a separate
  completely closed original Lean theorem compilation with no `sorry` before
  its hard deadline.

## Registered result

| Policy seed | TypedLoop certified | Native certified | TypedLoop elapsed ms | Native elapsed ms | TypedLoop ms/proof | Native ms/proof |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 3/16 | 6/16 | 97,304.87 | 95,204.14 | 32,434.96 | 15,867.36 |
| 29 | 6/16 | 6/16 | 88,289.66 | 95,399.49 | 14,714.94 | 15,899.91 |

Seed17 had three native-only proofs and no model-only discoveries.
Seed29 had matching theorem coverage. Per-theorem compiler calls were
133 model / 132 native in seed17 and 121 model / 132 native in seed29.
The ceiling was **twelve subprocess calls per individual theorem**, rather
than an aggregate per-seed cap.

The registered **learning** criterion (strictly more verified theorems than
native in both seeds) and **compute** criterion (learning improvement plus
lower milliseconds per certified theorem in both seeds) were both **false**.

The primary result is the immutable original scored run,
[LoopedTransformers CI 43554219466](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43554219466),
with compiler traces for every positive and negative theorem. Its
`FC13_report.json` SHA256 is
`f8e677e8e5f2d86820e718be885d23b2e1d2c9d50dd80728d3166a17b1bc05c9`.

## Independent compiler replay

[Independent replay CI 43554487985](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43554487985)
passed, compiling all 64 recorded model/native final chosen tactic prefixes
against the original immutable MathCan proof context through a separate source
renderer. It found **zero false positive certificates** and zero
unscored semantically valid final chosen proofs. The replay JSONL SHA256 was
`2ea167295cc33d7d1fefb2b422015d0cd9187e9649262c8f31c282869569d97c`.

The first FC-12 result was separately preserved, including 64/64 correct
independent compiler replays. FC-13 adds a genuine source-family transfer test
and a strict process-level timeout; it neither reselects FC-12 cases nor
changes the existing FC-12 score.

Remaining limitation: the fixed 12-family recurrent policy cannot advance
through two-goal induction proof branches because multi-goal states trigger
an explicit abstention. The FC-13 result motivates a separately preregistered
multi-goal search campaign.
