# CB01: CanonizationBench v0.1

**Goal:** test whether independently written Lean proofs can be canonized into a
shared lemma with identical theorem statements and downstream reuse. This is a
small proof-engineering prototype for Compactification Theory, with a bounded
deterministic history-equivalence fixture. It is not an automated discovery of
new theorems, and no learned model is involved.

## Ownership and formal statement

This package is additive. The exact stochastic, timed predictive quotient
remains with [IRC-02 / EvalMath](../../experiments/interface-relative-compactification/README.md).
CB01's deterministic machine only tests proof/library reuse, finite witnesses
and source-level metrics. Other research packages and existing workflows stay
untouched.

For a history type H, a future experiment type W and an observable output O:

~~~lean
def SameFuture (F : H → W → O) (x y : H) : Prop :=
  ∀ word : W, F x word = F y word
~~~

`CB01/Core.lean` proves reflexivity, symmetry, and transitivity. Its toy
machine records timed event signals; future equivalence entails one-step event
agreement and is preserved under synchronized transitions. The theorem assumes
equality under every declared future experiment. It does not imply that a
learned compressed state meets that assumption.

## Matched proof challenge

The baseline consists of two independently authored special-case proofs:

- `BaselineA.prefix_context`: future agreement under a fixed action prefix.
- `BaselineB.translated_context`: future agreement under an action grammar translation.

`Canonical.precompose` is one generic proof that subsumes both. The
declarations in `Consumers.lean` repeat their exact baseline statements,
then derive them using the common lemma. `Canonical.postcompose` preserves
agreement under observation normalization. `Canonical.pair` combines
output channels with separately established agreement.

The held-out `Heldout.lean` file imports only the canonical library and
proves three independently specified clients: repeated actions, trace-length
summaries and joint reports.

## Run locally

Python 3.10+ and standard library only:

~~~bash
python -m unittest discover -s research/canonization_bench_cb01 -p 'test_*.py' -v
python -O -m unittest discover -s research/canonization_bench_cb01 -p 'test_*.py' -v
python research/canonization_bench_cb01/runner.py --json /tmp/cb01-receipt.json
~~~

With the pinned Lean 4.15.0 release and Lake installed:

~~~bash
cd research/canonization_bench_cb01
lake build CB01
python runner.py --require-lean --json /tmp/cb01-verified.json
~~~

The path-scoped GitHub Actions workflow builds the Lean library and runs the
finite tests on the PR. The runner explicitly reports
`not_run_no_lake` when Lean is unavailable. It refuses to claim kernel
acceptance when `--require-lean` is specified without a successful build.

## Metrics and toy fixture

| Measurement | Expected v0.1 result |
|---|---|
| Independent baseline proofs | 2 |
| Canonical shared precomposition proof | 1 |
| Canonical calls, paired consumers | 2 |
| Canonical calls, held-out clients | 3 |
| Timed history classes, three-tick budget | [0,2], [1,3], [4], [5] |
| Fixed-length class index with public clock | 2 bits |
| Read-only grammar classes | 3 |
| Minimum distinguishing action word for 0 vs 4 | switch, read |
| After mutating state 2's read duration, witness for 0 vs 2 | read |

A clock-bound enumeration executes every feasible word of length at most the
budget; the finite fixture has only positive-duration actions. Denials,
timeouts, rewards, durations and public modes are observable. Because an
action may cost two ticks, histories 0 and 4 agree through budget two and
separate after switch then read at budget three.

The runner counts nonblank Lean lines in the baseline and canonical files and
reports actual call sites. It does **not** infer successful proof compression
from a decrease in source size. In the initial small experiment, the more
general canonical library costs additional source lines; reuse and semantic
coherence are the primary measurements.

## Verification scope

The six local Python tests verify the finite deterministic oracle, replay
witnesses, duration mutation, action-grammar monotonicity and reproducible
receipts. They cannot establish Lean kernel acceptance. That requires
`lake build CB01` in the PR workflow; the lexical placeholder guard
supplements this check. Exact theorem-statement preservation is established by
compiling both the baseline and replacement declarations with the same type.
No Mathlib-wide proof-reuse gain, probabilistic posterior equality, actual
cross-repository integration, or model-training result is claimed.

A next version can ingest pinned, statement-only tasks from EvalMath and
record blinded refactor attempts, dependency change, readability review,
kernel time and held-out consumer success.
