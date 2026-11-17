# SCM-21 results: rank-four capacity on SCM-20's trained-model receipts

**Status:** executed sandbox CPU audit and separate constructed geometry control on 2026-11-17. No GPU, trained-model replay, or additional training was performed. SCM-20 is the original source of every trained-model outcome here.

## Primary retrospective rank-four comparison

Each row uses exactly the common-eligible recipient/donor pairs from SCM-20's fixed visit-two assay, on the **causal-trained** model. All numbers are numbers of correct donor future predictions after patching.

| Seed | Stratum | n | Full CLS (128d) | PCA4 | Diff-PCA4 | Native span4 | Native write | Guaranteed Diff-PCA-only vs native span |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 191 | ID | 128 | 115 | 43 | 39 | 14 | 14 | 25 |
| 191 | OOD | 120 | 105 | 32 | 38 | 15 | 12 | 23 |
| 1191 | ID | 128 | 110 | 37 | 38 | 11 | 8 | 27 |
| 1191 | OOD | 128 | 110 | 30 | 39 | 11 | 9 | 28 |
| 2191 | ID | 128 | 110 | 34 | 45 | 8 | 8 | 37 |
| 2191 | OOD | 116 | 89 | 23 | 18 | 4 | 4 | 14 |
| **Total** | | **748** | **639 (85.4%)** | **199 (26.6%)** | **217 (29.0%)** | **63 (8.4%)** | **55 (7.4%)** | **154** |

Other reference controls: `head_readable` 455/748, `head_fiber` 68/748, `sae_group4` 27/748, norm-matched *random* fiber4 6/748, sham 0/748. `head_readable` and `head_fiber` have dimensions beyond the four-dimensional comparator budget.

### Sharp witness from public marginals

Each of the six cells satisfies `Diff-PCA4 > Native span4`. Irrespective of unreported pairwise success overlap, at least `25+23+27+28+37+14 = 154` individual method-by-pair comparisons must be successes for Diff-PCA4 where native span4 fails. The total count difference is also 154. This statement follows from the sharp Fréchet overlap bound and requires no independent-trial assumption.

A rank-four subspace calibrated from task differences can mediate significantly more of the tested counterfactuals than the model's existing native write span, in the same trained causal model. This is evidence that **rank four as an abstract dimensional budget is not the sole bottleneck**. It leaves open whether the effect comes from subspace orientation, differences in patch amplitude, calibration bias, or downstream nonlinearities.

### Patch norms and clean accuracy

The six mean norm ratios `Diff-PCA4 / Native span4` are 1.0349, 1.0421, 1.0620, 1.0532, 1.0252, 1.0251. They are **not per-pair norm-matched**. The actual learned native-write delta has still smaller norm than the full-difference projection onto its own write span. A matched-norm prospective replay on the private model checkpoints is needed for a causal explanation.

SCM-20's causal model loses clean accuracy relative to the frozen SCM-19 warm start in five of six seed/split comparisons at the registered 2-percentage-point tolerance. That registered failure remains unchanged.

## Separate matched-norm CPU geometry control

A frozen **constructed** R128 rank-four linear readout was evaluated on 256 independently generated donor-recipient pairs. The Diff-PCA4 span was fitted on 80 different calibration pairs. An orthogonal rank-four native-fiber comparator received exactly the same patch norm on each test pair. The resulting scores were:

- Full donor state difference: 256/256.
- Calibration-derived Diff-PCA4: 256/256.
- Orthogonal native-fiber rank-four: 0/256.

The construction is intentionally separable and illustrates an elementary linear-algebra mechanism; it is not evidence that the trained SCM-20 system behaves like the synthetic control. It establishes that rank and norm matching by themselves cannot make two causal subspaces equivalent.

## Validation and reproducibility

- CPU Python 3.13, NumPy 2.x for the independent control. Nine unit tests passed.
- Public distilled count receipt, Git-blob source lock and strict aggregate consistency checks included. The full source-audit command will also reparse the SCM-20 720-cell summary in a complete Git checkout.
- Frozen CPU outputs: `results/scm21_cpu_audit.json`, `results/scm21_synthetic_control.json`.
- No official EvalMath hidden test queries, model weight updates, 3050 access or trained RMP access occurred.
