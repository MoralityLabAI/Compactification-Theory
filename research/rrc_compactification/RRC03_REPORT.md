# RRC-03: adaptive coefficient-query pilot

**Status (2026-11-09):** finite exact oracle and supervised CPU student run,
with frozen checkpoints and portable replay. Reproduces the adaptive query
capacity theorem on 33 pairwise prefix-indistinguishable partition systems;
executes held-out policy generalization to other hypothesis counts.

## What changes from RRC-02

RRC-02 found an information-theoretic failure: a late extra part type of
weight `K>64` is invisible when only coefficients through degree 64 are
observed. RRC-03 gives the agent a paid query interface to selected larger
degrees, so the evidence can change during recurrent decision-making.
Every query reveals one binary comparison with the known Rogers–Ramanujan
baseline. The family and interface are explicit; the experiment measures
finite active discrimination rather than arbitrary mathematical discovery.

The derivation and its **sharp** lower/upper query bounds are written in
[RRC03_THEOREMS.md](RRC03_THEOREMS.md). The exact independent verification
and model split are registered in [RRC03_PROTOCOL.md](RRC03_PROTOCOL.md).

## Results: uniform 33-hypothesis family

| Queries | Exact adaptive | Trained RNN, 353 parameters | Trained full-history FF, 353 parameters | Fixed prechosen thresholds | Recurrent with feedback hidden |
|---:|---:|---:|---:|---:|---:|
| 0 | 1/33 | 1/33 | 1/33 | 1/33 | 1/33 |
| 2 | 4/33 | 4/33 | 4/33 | 3/33 | 3/33 |
| 3 | 8/33 | 8/33 | 8/33 | 4/33 | 4/33 |
| **4** | **16/33** | **16/33** | **16/33** | **5/33** | **5/33** |
| 5 | 32/33 | 28/33 | 23/33 | 6/33 | 6/33 |
| 6 | 33/33 | 31/33 | 32/33 | 7/33 | 6/33 |

The analytic adaptive-policy accuracy is `min(2^B,33)/33`; the static
capacity is `min(B+1,33)/33`. Both are exactly attained by the supplied
stdlib reference programs, and the independent Bellman DP confirms the
optimal adaptive number for all budgets 0–6. The gap at B=4 is a factor
`16/5=3.2` in expected exact identification success for adaptive feedback.
The neural student is an **imitation** of a known midpoint strategy. Its
feedforward comparator matches it at the 4-query gate.

## Held-out hypothesis counts (disjoint sizes)

| Unseen H | Query budget | Exact adaptive upper bound | Recurrent | Feedforward | Fixed queries |
|---:|---:|---:|---:|---:|---:|
| 13 | 4 | 13/13 | 12/13 | 13/13 | 5/13 |
| 25 | 4 | 16/25 | 16/25 | 16/25 | 5/25 |
| 49 | 4 | 16/49 | 16/49 | 16/49 | 5/49 |
| 13 | 6 | 13/13 | 13/13 | 13/13 | 7/13 |
| 25 | 6 | 25/25 | 25/25 | 25/25 | 7/25 |
| 49 | 6 | 49/49 | 40/49 | 42/49 | 7/49 |

The models were trained with `H in {9,17,33,65}`. Heterogeneous random
query histories (including off-midpoint actions) were used for imitation,
so evaluation uses unseen candidate counts and its own on-policy feedback
histories. The underlying defect type and many candidate degrees were
already represented in the training pool; this is candidate-count transfer
rather than a family-disjoint discovery test. Both learners generalize to the exact four-query information
capacity for `H=25,49`. The feedforward network is more accurate at six
queries on the largest holdout. Neither architecture shows an exclusive
ability to use feedback.

The feedback-censoring ablation replaces each response available to the
policy with zero, while preserving real responses for exact posterior
scoring. With H=33 and B=4, the RNN falls from `16/33` to `5/33`. With
H=49 and B=4, it falls from `16/49` to `5/49`. This supports causal use
of previous observations in query selection, within the declared interface.

## Training and resources

The models each have 353 trainable FP32 parameters (`11,296` weight bits).
The sequential student keeps 16 FP32 hidden values (`512` mutable bits).
The feedforward control reads the entire 6-slot public query history and
uses exactly the same parameter budget. For four decisions, the dense
multiply-accumulate accounting is approximately **1,984 recurrent MACs**
with state recomputation from history, or **1,344 feedforward MACs**;
streaming the recurrent state once per new feedback reduces repeat work.
These numbers exclude `tanh`, `sigmoid`, exact coefficient generation,
query oracle service, posterior MAP decoder, Python overhead, model loading
and training. Query count and returned one-bit observations are tracked
separately. The student did not learn to compute Rogers–Ramanujan
coefficients.

The training used PyTorch with one CPU thread, 1,500 Adam updates of 256
examples per model from the same 19,604 teacher-labeled histories.
`rrc03_recurrent.json` and `rrc03_feedforward.json` contain full weights,
training metadata and exact training-data hash; the portable evaluator
needs only Python 3.10+ standard library. `rrc03_pilot.json` binds the
models and source by SHA-256, contains all evaluation cells and full trace
hashes. The exact audit separately cross-checked 3,201 coefficient values
and 1,056 binary comparisons for the 33-hypothesis family.

## Reproduce

From the root of Compactification-Theory:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_active03.py' -v
python research/rrc_compactification/active03.py --check --out research/rrc_compactification/results/rrc03_exact.json
python research/rrc_compactification/bench_active03.py --check
```

With PyTorch installed, to **retrain** rather than replay:

```bash
python research/rrc_compactification/train_active03.py --steps 1500 --batch-size 256 --per-size 1000
python research/rrc_compactification/bench_active03.py --check
```

The final command fails if weights or source changed. Recompute and review
a replacement receipt before treating a new run as registered evidence.
Training consumes CPU and time; the CI replay consumes neither model
inference services nor accelerator capacity.

## Limits and next experiments

The interface reveals an oracle-computed binary comparison to a *known*
periodic generating function, and hypothesis count is public. The queried
comparison is stronger than observing a raw unlabeled sequence with
unknown mathematical origin. An unrestricted numeric coefficient query
would supply more than one bit and may change the theoretical comparison.
The trained RNN and the feedforward comparator share the same optimal
four-query success; six-query performance leaves room for improved policy
training but does not establish a general recurrent advantage.

The remaining end-to-end objective is a raw-coefficient student: consume
normalized `f_0..f_T` directly, learn to propose product grammars without
Euler-exponent features, and use exact verification under held-out local
rule families and modulus-disjoint products. The RRC-03 active-query
theorem provides a bounded evidence-acquisition subtask for that future
experiment.
