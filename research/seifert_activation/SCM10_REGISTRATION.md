# SCM-10 pre-registered scalable clean-enclosure test

**Registered 2026-11-14, before any pretrained RMP capture.** Additive to SCM-01 through SCM-09; do not edit the existing G0 gate or sealed RMP results. The synthetic model generator, shapes and provenance are frozen here before attempting trained-model interpretation.

## Main questions

Can an exact-dyadic Gram norm bound and tokenwise causal balls keep a useful clean-state enclosure through all eight visits at the original RMP shape (sequence 64, hidden dimension 128, four heads)? Can an independent 80-digit small-model evaluator confirm the new backend's inclusion predictions?

## Synthetic fixed cases

- Fully tied, `beta=0.1`, 64×128, eight visits. Report per-visit maximum clean-center error and accept the *numerical scalability* gate only if visit-eight upper radius is below `0.01`.
- Fully tied, `beta=0.25` (registered RMP block coefficient), 64×128, eight visits. Report all radii, including failure to certify; no favorable-data selection.
- Fully untied with post-visit frozen input reinjection, `beta=0.25`, same shape. Accept the numerical-scalability gate only if visit-eight upper radius is below `0.005`.

Weights follow the already frozen `scm09_reference.DeterministicRMP` integer/dyadic recipe. Tokens are the source generator's 64-token vector. Positional frequencies use the registered dyadic table `[2^-k]` for `k=0..63`, explicitly separate from PyTorch's production frequency computation. These cases are synthetic and untrained.

## Soundness checks and controls

- Exact integer Gram/Gershgorin bound checked against independently computed float64 SVD for matrices with both tall and wide shapes.
- All elementary interval primitives compared with exact `Fraction` or 80-digit `mpmath`, including negative scores, the GELU transition near zero and the coarse out-of-domain fallback.
- Independent 80-digit reference for tied and untied miniature RMP decoders, checking every token and all pre/mid/post visit enclosures.
- A deliberately over-wide coordinatewise-only run and the tied `beta=0.25` model are retained as informative negative checks.
- Source-constrained architecture, non-EVAL model, incompatible source frequency table, nonfinite values, overflow and invalid matrix dimensions must refuse.
- Check 64-token/128-hidden audit path with original `monitor_calibration` selection-half, source/provenance hashes and immutable exclusive-create output semantics. Do not read reporting half or sealed tests for this pilot.
- Default audit scope is an **ideal-real source function under explicit CPU IEEE/BLAS verifier assumptions**; report `floating_execution_certified=false` and `observed_link_certified=false` in all pilot receipts.

## CI/reproducibility

Use Python 3.13, NumPy 2.3.5, PyTorch 2.10 CPU, SymPy 1.14 and mpmath. SCM-10 dedicated CI runs high-precision proof tests and full-shape synthetic cases. Older minimal CI environments without NumPy skip only SCM-10's optional tests; their existing 190-test suite remains intact. Across BLAS vendors, compare certified *thresholds*, architecture hashes and source provenance, allowing different harmless center rounding; do not require bitwise-identical float64 centers across machines.

## Next gate

Pretrained RMP checkpoint bytes, calibrated D-shared dictionary and monitor bundle, plus an independently attested CPU BLAS numerical contract, are needed before treating the new source-level bounds as a fully trusted real-model certificate. Physical PyTorch execution requires separate bounding or a bit-for-bit validated implementation.
