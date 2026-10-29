# CB02: pinned EvalMath canonization transfer

**Parent**: [CB01](README.md), branch `research/canonization-bench-cb01-20261029`, draft PR #9.
**Status**: four held-out consumers added after successful development CI run
[40886559993](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40886559993).
Frozen development commit: `d216fb87623ec396a0b59538908a422c5db66e6a`; the subsequent
manifest-only commit does not alter the candidate. CI for the held-out
phase independently compiles all four new consumers.

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

## Held-out challenge log (CB02/H)

The four client tasks in `CB01/Transfer/Heldout.lean` were authored after
successful development CI and import `CB01.Transfer.Canonical` only:

- **H01**: future-law invariance across a fixed public interface mode/deadline
  and an independently supplied action translation.
- **H02**: postprocessed observations remain equal when exact encoded states
  collide under a correct decoder.
- **H03**: two feedback channels remain jointly equal after a common action
  translation.
- **H04**: code equality induces deterministic immediate-event and successor
  equivalence.

The evaluation manifest pins the held-out source blob. This test checks proof
reuse and compiler acceptance on mathematically related statements. The
results do not measure autonomous research novelty or model generalization.

## Verified CB02 v1 result (2026-10-29)

Evidence: [held-out CI run 40886987513](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40886987513),
head `885cc118740051f1727f832437dbf46e4c8e00bf`. The GitHub
artifact `cb01-cb02-receipts` contains `cb02-transfer-receipt.json`,
including Git blob checks and the Lean build verdict.

| Measurement | Measured |
|---|---:|
| Registered source-derived target propositions | 3 |
| Independently compiled baseline proof bodies | 577 UTF-8 bytes |
| Refactored candidate proof bodies | 112 UTF-8 bytes |
| Shared canonical proof bodies | 604 UTF-8 bytes |
| Candidate plus shared library | 716 UTF-8 bytes |
| Common-target statement identity checks | 6 |
| Held-out Lean consumer theorems compiled | 4 |
| Normal Python tests in first held-out CI | 11 passed |
| EvalMath paper and reference Python blob checks | Both passed |
| Lean package build | Passed |

The client proof-body saving is `(577-112)/577 = 80.6%`, measured
lexically, while the one-off library-inclusive total is
`716/577 = 1.241` times the baseline. This is a positive *reuse*
result with measured general-library overhead. The four held-out clients
also compile without touching the frozen five-file development set.
The complete Lean elaborated proof terms, elaboration time under controlled
hardware, and comparability of disparate mathematical claims are outside
these initial source-byte measurements.

Frozen development commit:
[`d216fb87623ec396a0b59538908a422c5db66e6a`](https://github.com/MoralityLabAI/Compactification-Theory/commit/d216fb87623ec396a0b59538908a422c5db66e6a).
Held-out source blob: `3d684e4e6c53975503080a257b1959cef0486d1f`.
An additional CI regression replays the Python tests with `python -O`.
