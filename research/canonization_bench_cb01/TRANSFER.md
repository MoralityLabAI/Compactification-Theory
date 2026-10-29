# CB02: pinned EvalMath canonization transfer

**Parent**: [CB01](README.md), branch `research/canonization-bench-cb01-20261029`, draft PR #9.
**Status**: development phase; held-out clients will be added only after this
phase builds and its canonical library is frozen by exact Git blob SHA.

## Source ownership

All mathematics is traced to [EvalMath IRC-02/T](https://github.com/MoralityLabAI/EvalMath/blob/2616092a2c73c083378e508abfd966fdf5e84814/paper/interface_relative_compactification.md).
Pinned commit: `2616092a2c73c083378e508abfd966fdf5e84814`.
The paper Git blob is `1036fcb7c9a7c9caaf07b5afca87c816938c16fd`.
The reference Python Git blob is `db1f9b9872baef0215e00734faa52f748a896f9d`.
CI checks out EvalMath at this commit read-only and verifies both source blobs.

The new code is in `CB01/Transfer/` in the Compactification-Theory Lean
package. No EvalMath code is vendored or changed.

## Pre-registered target contracts

| Target | EvalMath source | Lean specialization |
|---|---|---|
| T01_MENU | §4: adding public actions refines all-futures equivalence | Precomposing future tests with a legal-menu translation preserves equivalence |
| T02_CODE | §2: any sufficient code separates distinguishable histories | Equal exact codewords imply equal future laws, and any differing test forces unequal codes |
| T03_STEP | §2: successor congruence | Deterministic full-trace equivalence implies equal immediate events and equivalent successor states |

The paper's theorem is about rational stochastic kernels with adaptive
feedback, positive branch durations, and graded posteriors. CB02 formally
proves **only these abstract and deterministic specializations**.
It leaves stochastic conditioning, finite successor closure, 11-class
cardinality, and the full bit lower bound with EvalMath.

## Freeze and blinded-client protocol

1. Register statement types in `Statements.lean` and exact source anchors in
   `transfer_manifest_v1.json`.
2. Produce independently proved `Baseline.lean`; produce shared
   `Canonical.lean` and derived `Candidate.lean`.
3. Compile both against the exact same statement aliases in `Audit.lean`.
4. Freeze all five development files with their Git blob SHA-1 values and
   record the successful CI commit before constructing held-out clients.
5. Introduce four held-out clients that **import only the frozen canonical
   library**; never import the baseline in a held-out test.
6. Run pinned-source verification and Lean kernel compilation in CI, plus
   Python integrity and mutation tests. Report failures, changes after
   freeze, and compile verdicts.

This is a **protocol-level holdout**, meaning the candidate files are fixed
before the evaluator files exist. The authoring model has read the EvalMath
paper and can anticipate its themes; this is not statistical blindness, a
new theorem-generation result, or an autonomous-agent benchmark. A future
multi-agent blinded test should isolate solver prompts from hidden targets.

## Metrics

The receipt records theorem counts, baseline/candidate proof-body bytes,
the extra canonical library cost, full local import closure, six
statement-identity proof checks, and kernel success. The ratio of baseline
to candidate *client* bytes measures refactoring efficiency; the total
`canonical + candidate` cost is also reported to expose shared-library
overhead. Build timing is a warm runner diagnostic with hardware
dependence, and semantic fidelity is limited to stated specializations.

Run with Python 3.10+:

```bash
python -m unittest discover -s research/canonization_bench_cb01 -p 'test_*.py' -v
cd research/canonization_bench_cb01
lake build CB01
python transfer_bench_v1.py --require-lean --source-paper ../../external-evalmath/paper/interface_relative_compactification.md --source-reference ../../external-evalmath/src/spiral_ln/interface_relative_compactification.py
```

The last command's source paths assume a sibling pinned EvalMath checkout
at repository root. A local source-free run reports
`not_replayed_external_source` and thereby keeps that evidence separate.
