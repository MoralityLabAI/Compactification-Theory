# SCM-02: implementation and synthetic validation

**Run date:** 2026-11-13. **Execution:** local CPU with Python 3.13.5,
PyTorch 2.10.0+cpu and SymPy 1.14.0. The original SCM-01 tests are included.

Command:

    python -m unittest discover -s research/seifert_activation/tests -v

Results: **20/20 passing unit tests**. The new tests include:

- Exact independent pairwise winding and Artin linking, for positive/negative
  Hopf braid presentations, cancellation pairs and repeated twists.
- Four explicit pure 3-strand braid cases and 40 seeded pure 3-strand words
  assembled as products of conjugated even powers (seed 1407).
- Rejection of nonpure named-endpoint closure, actual pairwise PL segment
  collisions, and near-collisions.
- A sufficient externally verified derivative-bound checker, with positive,
  negative and missing-contract fixtures.
- A mock `AnalysisModel.run(tokens, hook)` with downstream token responses to a
  cyclic injection after visit 0. The fixture produces pairwise winding 0 at
  visit 1 and +1 at visits 2-4, with unchanged final answer, so winding alone
  is *not* evidence of a causally relevant answer mechanism.
- Agreement under 32-to-64 angular sampling in the mock, fixed injection axes
  under a changed readout chart, hash stability, and invalid chart rejection.

**No trained RMP checkpoint was available in this environment.** No neural
activations from LOOP-T/FF-U, angular-response statistics, causal effect,
compactification gain, or four-dimensional surface invariants were measured.
Gate A remains passed for synthetic tests. Gate B is *unexecuted* on trained
models, and Gates C-D await later work.

## Files

- `winding.py`: independent polygon winding, pure-link crossing-number check,
  optional *externally supplied* continuous derivative bound.
- `rmp_capture.py`: standalone RMP adapter, hash-verified artifact input,
  deterministic selection of 64 calibration-selection examples, decoder and
  random-orthogonal chart modes with fixed injection axes, explicit invalid
  observations and optional angle doubling.
- `tests/test_winding.py`, `tests/test_rmp_capture.py`: exact and mock tests.
- `SCM02_MATH.md`: configuration-space theorem and certificate limits.
- `SCM02_REGISTRATION.md`: exploratory protocol, controls and decision gates.

## Example command (on a machine with the original RMP artifacts)

    python research/seifert_activation/rmp_capture.py \
      --rmp-root /path/to/LoopedTransformers \
      --runs /path/to/LoopedTransformers/experiments/recurrence_monitorability_v0/runs \
      --artifacts /path/to/rmp_v0/artifacts \
      --monitor-bundle /path/to/rmp_v0/bundle \
      --dictionary /path/to/rmp_v0/phase2/dictionaries/D-shared.pt \
      --model 'LOOP-T s1' --features 10 20 \
      --inject-token 4 --observed-tokens 16 32 \
      --radius 0.05 --angles 32 --double-sampling \
      --out /path/to/scm02/unique-run-id.jsonl

The dictionary basename and feature indices above are placeholders: inspect
RMP `phase2/dictionaries.json` and lock the actual registered D-shared artifact
and two feature IDs before using this command. The script rejects unknown
models, wrong dictionary kind, hidden-size mismatches, missing artifacts and
existing output paths. Each run uses the calibration selection half only.
