# IC-01: Intervention-preserving representation compression

Status: prospective protocol, 2026-10-28. No experiment has been run for IC-01. Numeric thresholds, model assets, and seeds remain unregistered; freeze them in a run manifest before collecting scored results.

Theory: [local research bridge](../../note/intervention-preserving-compactification.md).
Canonical code and proofs: [EvalMath at the publication commit](https://github.com/MoralityLabAI/EvalMath/tree/adeb45fbb6b611e0e1fc38989e28f0d4d621698b/research/interventional_compactification).

## Question and ownership boundary

At matched representation and predictor budgets, which candidate encodings preserve the effects of the specified interventions and the observed outputs? A secondary question asks whether a percolation selector adds held-out predictive value after the selection controls are matched.

This is a new downstream compression study. E10 remains with its existing owner and frozen protocol. E10's selection-stage controls and IC-01's representation/modeling controls are separate comparisons. No selector is retuned against E10 outcomes, and no E10 run is launched by this package.

## Required run manifest

Before running, record the checkpoint and tokenizer revisions with content hashes. Pin the code, layer boundary, cache schema, prompt-family split, and intervention grammar. Include the random seeds and a source hash for every trained compressor. Candidate assets such as an SAE dictionary or VPD transform need explicit paths and hashes; their availability is currently unverified.

Specify K and the norm, the observables, each allowed operation, and the horizon budget. Where experiments only enumerate states reachable up to T, make the remaining horizon part of the contract. Record how full-state context is reconstructed, including unchanged activations or caches required to make each operation deterministic.

Freeze k, coordinate precision, normalization, encoder size, transition/readout architecture, fitting budget, and the evaluation metric. Set numerical tolerances and a procedure for validating any Lipschitz upper bound. Spectral estimates and adversarial searches are estimates unless accompanied by a proof that they cover the required domain.

Register a minimum useful improvement and the statistical decision rule for the percolation-specific comparison. Specify the independent sampling unit, multiplicity policy, and how prompt-family dependence is handled. Until these values are fixed, outcomes cannot pass a preregistered benefit gate.

## Candidate comparisons

Compare PCA and a dimension-matched random projection with SAE features, VPD coordinates, and the percolation-selected representation. Match k and encoding precision while reporting encoder complexity and wall-clock cost separately. Hold the transition/readout modeling budget fixed. The first predictor can be affine; a fixed-budget nonlinear predictor is a separately declared comparison.

For percolation component selection, use matched alive-component quotas for degree, weighted degree, mean causal importance, attribution mass, firing rate, and frequency-matched random controls. Import E10 artifacts only after their owner supplies the pinned outputs and reuse terms. Their existence is not asserted here.

Split prompt families and intervention templates before fitting. Hold out at least one intervention family and test composition-order changes. Candidate fit and selection occur only on the declared training partitions.

## Measurements and falsifiers

Measure held-out one-step commutation error and readout error. Search for near-colliding encoded state pairs with diverging post-intervention states, and report the near-fiber obstruction under a declared L bound. Follow shared action words at horizons 1, 2, 4, and 8. State whether logit-margin conditions hold at each decision when reporting action agreement.

Include the hidden-coordinate counterexample from EvalMath, shuffled features, and an oversized encoder as diagnostic controls. A random-initialization alignment control requires the same fitting and complexity budget as the trained-model comparison. Preserve the original state/action cases and hashes used for every claimed failure.

## Decision rules

An exact behavioral quotient claim is falsified by a reproducible equal-code pair with different observable futures within scope. For a transition-factorization claim, equal codes with different encoded successors suffice. Floating-point equality needs an explicit numerical policy; near-fiber bounds provide a separate approximate test.

A uniform epsilon claim fails when a verified residual or verified near-fiber lower bound exceeds epsilon on the declared domain. A held-out sample mean estimates performance under that sampling procedure. It establishes no sup-norm certificate without the required coverage and regularity bounds.

A trajectory guarantee requires the one-step and stability hypotheses on all relevant states. When an upper bound is too loose to constrain the desired behavior, report it as inconclusive for useful mechanistic prediction.

The percolation-benefit hypothesis fails its registered gate when matched controls meet or outperform it under the frozen statistical rule. Evidence favoring percolation requires the predeclared improvement on untouched prompt/intervention families. Missing assets, unresolved scope, insufficient power, or unverified geometric assumptions receive an inconclusive status.

## Replay of the available reference only

In a checkout of the linked EvalMath commit:

```bash
python research/interventional_compactification/verify.py --check
python -O research/interventional_compactification/verify.py --check
```

These commands run five synthetic Python reference tests. They neither download a checkpoint nor invoke a training or inference workload. The source-hashed receipt records that narrow scope.
