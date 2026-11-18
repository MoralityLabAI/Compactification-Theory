# CGC-01 source-only mathematical audit and future physical-block study

**18 November 2026.** This protocol accompanies [THEORY.md](THEORY.md). The initial code tests are source-only checks of explicitly specified finite matrices and exact rational witnesses. They are **not** a prospective neural-model experiment and have no newly scored scientific roots. Future neural tests require a new source/observer lock before model captures.

## Stage 0: executable mathematical checks (included)

From repository root, with `numpy` installed:

```bash
python -m unittest discover -s research/causal_gauge_compactification -p 'test_*.py' -v
python research/causal_gauge_compactification/finite_audit.py
```

The finite tests cover these mathematical claims under explicit contracts: a protected current/future readout Jacobian generates a physical orthonormal null basis; the task-conditioned truncated SVD minimizes worst-case and Frobenius first-order error; a high-magnitude physical weight singular direction can be task irrelevant; BPD component-gauge shears preserve the complete physical operator and its SVD intervention; Euclidean singular values depend on the physical state metric; a full-row-rank multioutput public Jacobian admits the advertised minimum-norm protected correction; a nonlinear tangent-safe edit can still change finite public outputs; the correction's exact quadratic residual fits a Hessian bound; the real-linear affine row-space quotient is closed under all supplied actions; the toy rational Krylov rank matches the independent floating quotient rank; full-action observability may require more state dimensions than a passive single-action probe; observability Gramian rank cannot decrease with horizon; nonnormal recurrences have early gain despite small eigenvalues; separated futures incur a quantized-code bit lower bound. All data are generated from short, fixed matrices contained in source.

**Exactness:** `Fraction`-based Gaussian elimination gives an exact rank certificate for the integer examples. NumPy SVD/projectors and floating normative tests are numerical checks against analytical formulae at small scale, **not** formal verification of arbitrary real matrices. The analytic theorems have handwritten proofs, not Lean certificates. Record Python/NumPy versions in CI output.

## Stage 1: prospective physical-model campaign (not executed)

**Hypothesis:** A development-selected rank-`r` *physical* BPD block built from the protected task Gramian transports a task-specific causal variable across previously unseen recurrent passes and source/program families, beyond equal-norm, matched-future-Jacobian and matched-task-sensitivity shams, while respecting a complete protected observable channel.

Choose one frozen model family and named recurrent state bus. Candidate first pair: a small trained TRM (source-generated Sudoku/task-variable interventions) and a compact looped transformer (source-generated Horn-graph/equation tasks); the second model should constitute independent replication, not a substitute for root replication within the first. Before any checkpoint outcome, freeze:

- The *full* readout definition, including normalization/head, all token sites needed for the behavioral target, exact protected output conditioning and input context. Preserve true full recurrent state, caches and forcing.
- Output metric/whitening, physical hidden metric, selected development visit/horizon, radius ladder, and the rank/soft-rank tolerance; the public protected Jacobian and future task Jacobian must come from the true frozen target model on development samples.
- The physical BPD reconstruction fit (including residual weights, active masks, dictionary training phases), independent dictionary initializations, GL factor gauge checks, physical output-SVD and task-optimal SVD, independent SAE decoder frames, and full native BPD replay. Avoid selecting a rank from heldout task scores.
- Admissible donor pairs, independent task-variable label oracle, current true normalized-logit match, protected future output tolerance, whether any finite public-level-set corrector uses privileged future gradients, source family hashes, and count of *attempted* donors. Exact LayerNorm shift gauges may be treated separately from merely Jacobian-null directions.
- Minimality and utility criteria: heldout native task correctness, effect on an independent target variable, recurrent depth extrapolation, finite-radius protection of all monitored outputs, total forward/fit/scheduling costs, and model-root-level uncertainty.

**Controls:** full native state transplant, no edit, full physical `VU` reconstruction, output-SVD of `VU`, task-conditioned Gramian block of the *teacher physical transition*, randomized protected dark subspaces at matched norm, equal signed target-gradient responses, task-sensitivity-matched negative block, best-rank `W` SVD baseline, fixed cross-root gauge alignment and phase-disjoint learned-dictionary training. A compact factor claim must beat relevant controls and not merely reproduce the whole physical matrix.

**Causal ranking:** let `R_B` be independent task-variable donor recovery after the selected block edit, and `R_0..R_63` the same eligible-pair recovery under 64 development-calibrated shams per independent teacher root. Report total task effect, first-order future-response prediction, residual nonlinear effect and the physical/gauge stability of the selected subspace. Use root as inferential unit, not token/pair/sham. A preregistered strong effect must clear an independent-root null tolerance rather than a conveniently chosen single random direction. A 59-root maximum of identically distributed sham-derived root statistics supplies a one-sided 95th-percentile bound with ~95% confidence under the stated iid root law, and must be fixed before a separate >=32-root confirmation cohort. Source-family-disjoint tasks and effective sample counts are required for any behavioral transfer claim.

**Falsifiers:** dominant task eigenvectors collapse to matched target-gradient controls; equally good BPD fits yield inconsistent physical subspaces without a prescribed alignment; top physical-weight SVD suffices; best causal block merely preserves a local tangent and breaks finite public protection; the observed low-dimensional map fails semiconjugacy under additional passes; a small decoder hides task labels in a schedule/external memory; global separated-basin counts defeat the local rank frontier; or eligible natural interventions are too rare to support a robust claim. A negative is recorded without redefining the metric or moving the rank threshold.

**Protected exclusions:** no existing NOR, SMC, SCM, LSV, BPD SF03/04/05 or other source-locked receipts are revised; do not access ARG-03-N's sealed 168-case set. Stage 0 passing is proof-oracle hygiene only. The exact finite results must never be reported as evidence that a trained TRM has learned this quotient.
