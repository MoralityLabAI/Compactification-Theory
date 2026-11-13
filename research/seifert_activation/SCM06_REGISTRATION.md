# SCM-06 source-locked RMP derivative gate

**Date:** 2026-11-13, prior to accessing trained RMP artifacts for this track. Scope: a new standalone checker over the SCM-02/04/05 *selection-half exploratory capture*, not a modification of sealed RMP data or existing experimental registrations.

## Locked source and samples

- RMP source reference: `MoralityLabAI/LoopedTransformers` on `rmp-v0-phase2`, with exact git blob hashes in `scm06_audit.PINNED_RMP_SOURCE`. Changed source code is a hard refusal requiring a fresh proof review.
- Cells: `LOOP-T s1` and `FF-U s3`; use eight visits and checkpoint/dictionary hashes in each capture receipt. RMP's input re-injection order stays authoritative.
- Examples: the SCM-02 fixed group-disjoint 64 examples, evenly across pointer chase, gated pointer chase, reachability, and sudoku4, drawn from the **selection half** of `monitor_calibration` only. The reporting half and sealed test split remain untouched.
- Intervention: token position 4, two frozen `D-shared` decoder directions, radius 0.05 times clean post-visit-0 token RMS; observed positions 16, 32, 48. Controls keep intervention axes fixed and change only the output chart. Angles are the original 32 and 64 uniform samples.
- New capture fields: `injection_baseline_rms`, `sequence_length`, `clean_chart_by_visit`. Old receipts with missing fields produce an explicit refusal; no synthetic defaults are allowed.

## Check and report

1. Verify checkpoint, dictionary, chart and intervention-axis SHA-256 against the record; verify all four pinned source blob hashes. Run CPU inference only for loading the immutable model, not for retraining.
2. For every used block, calculate exact dyadic spectral-norm upper bounds, global LayerNorm and attention derivatives, exact-erf GELU bound, then multiply per visit. Record the full mathematical factor chain.
3. For each pair and visit, check SCM-05's rational polygon clearance against the derived global bound using `44/(7N)` as a strict safe angular-gap upper. Provide the direct clean-radius disk null-braid certificate too.
4. The primary output is a **certified derivative upper bound for the smooth real-weight model**. Its sampling and disk inequalities are conditional on exact or soundly enclosed real activation coordinates. Report refusals as `insufficient_global_bound`, source/metadata mismatch, missing-new-fields, or geometric rejection; track each source-task hash and immutable audit SHA-256.
5. Match a positive real-model disk null-braid certificate against any extracted nontrivial Artin/Milnor result. Such a contradiction is an instrumentation alarm demanding causal and finite-precision investigation.

## Adversarial controls and registered interpretation

- A causal prefix whose observed tokens precede the injected token must have **exact zero angular derivative**, regardless of checkpoint norm. Tests enforce this structural control.
- Unsupported exact block structure, training mode, modified source files, forged hashes, and missing amplitude metadata are refusals.
- Differentiation tests use automatic differentiation solely to try to falsify the analytic upper bound on *synthetic small models*. Their observed derivatives are not substituted for a global proof.
- A nontrivial braid with a global sampling inequality and verified real sample enclosures would be a result for the smooth real-valued chart model. Moving that claim to actual floating-point inference requires a separate verified numerical error envelope; this study does not assume one.
- An enormous global bound that certifies zero later-visit examples counts as a negative practical outcome of this proof method. Follow-up tightening must be preregistered, e.g., interval LayerNorm variance lower bounds, state-radius propagation, and causal attention Jacobian bounds.

## Standalone runner

```sh
python research/seifert_activation/scm06_audit.py \
  --input /path/to/immutable_scm04_capture.jsonl \
  --out /path/to/new_scm06_audit.jsonl \
  --rmp-root /path/to/LoopedTransformers \
  --runs /path/to/rmp/runs \
  --artifacts /path/to/rmp/artifacts \
  --dictionary /path/to/registered_dshared.pt
```

This runner leaves all RMP sources, weights, gates, datasets, and previous receipts unchanged.
