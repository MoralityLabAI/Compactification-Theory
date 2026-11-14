# FC-11 post-hoc: fully allotted Lean-native tactic search

This independent control was added after the first registered FC-11
model comparison. The original registered result is preserved unchanged.

[Successful original-source CI #43543948775](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43543948775)
compiled each of the **14 original held-out MathCan theorems** using the
**same preregistered eight tactic choices** in a Lean-native `first`
search with up to three goal transitions. The source was pinned to
`patrickdugan/MathCan` commit
`c84e2025334e9a967c1297425a490f98d5b3d189`,
toolchain Lean 4.35.0. All candidate source scripts were sorry-free.

Every theorem received a generous **20-second** native Lean
compiler time allowance. The resulting coverage was:

| Source family | Certified under full-time native search | Total original proofs |
| --- | ---: | ---: |
| Intervention | 1 | 6 |
| HookState | 0 | 4 |
| GraphAuthorization | 4 | 4 |
| **Total** | **5** | **14** |

There were **zero timeouts**. Full native search required
approximately **13.56 seconds** total compiler wall time.
The original reference theorem proofs still compile 14/14.

## Exact source agreement with the original first test

The first registered evaluation was [model CI #43543404745](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43543404745),
using two previously frozen 62,401-parameter LR-01 actor checkpoints.
Under each model's own measured CPU time allowance,
the native search certified 3/14 (seed17 budget)
and 4/14 (seed29 budget), with **two** and **one**
native timeout respectively. The two models certified
4/14 and 1/14 original theorems, all using a single
arithmetic tactic.

The post-hoc evaluator used an independent Python module
that regenerated the native `first` scripts from the
original Lean theorem source and the same fixed eight-action bank.
An independent archive cross-check compared the SHA256 digest of
every native source script in the first registered evaluation
against the corresponding full-budget script:

**All 28 of 28 source SHA256 digests matched exactly.**

Thus the only intended difference was the per-theorem
Lean compiler time limit, with normal runtime fluctuation.

The first registered proof-count predicate is formally true
because seed17's 4 certificates exceeded the budget-limited
native baseline's 3. The fully allotted native search,
with **5/14** accepted proofs, outperforms **both**
learned model seeds. There is **no robust learned
proof-search advantage** on this original imported
Lean theorem corpus. The registered compute
efficiency criterion was already false.

## Records, trust and next stage

The full control source, original theorem-specific
generated Lean scripts, accepted/failed compiler
results and raw timings are in artifact
`fc11-posthoc-unbounded-native-43543948775`,
with canonical JSONL SHA256
`d36475c4c46e0f0bf13a384130aa96ad0cca0d633de1289e0c28ce068a584698`.

The post-hoc control includes no source reference proof
tactic hints, gold labels or model predictions.
Only the original theorem source and the eight
registered tactic scripts determine its search.
The kernel is the authority for its five accepted
proofs. Some original theorems require parameterized
rewrites and exact hypotheses beyond this fixed
eight-action, three-step search. The next
benchmark should support typed parameterized
tactic actions, longer intermediate states
and warmed-up Lean-native search that avoids
timing artifacts from separate compiler launches.
