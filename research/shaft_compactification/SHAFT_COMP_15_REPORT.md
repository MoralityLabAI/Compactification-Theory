# SHAFT-COMP-15: Factorial test of readout access and directional observability

**Registered November 11, 2026**, before running the four new GRU seeds. See [protocol](SHAFT_COMP_15_PROTOCOL.md), preregistration commit 0240f2f3378647d1106111b000e0aa425cdaee27. The completed full CPU experiment was reproduced byte for byte.

**Primary finding:** Learned-readout directional sensitivity predicts delayed behavioral disagreement with ROC-AUC **0.710225**, versus **0.484385** for learned-readout rank deficit on the same held-out pairs. Registered AUC improvement **+0.225840**, 95% paired within-model bootstrap interval **[+0.158064,+0.292475]**, two-sided within-seed/depth permutation p **0.000999**. All four fresh held-out model seeds favor directional sensitivity.

Only synthetic four-bit lookup tasks and 3,265-parameter GRUs were evaluated. The learned task family and the full trained model substrates remain available throughout. The result supports a narrow scientific claim about interpreting recurrent hidden states; it gives no evidence about hidden frontier-model objectives, distributed-device persistence or recovery of erased weights.

## Design and mathematical definition

COMP-13 preregistered a learned-probe effective-rank statistic that performed at chance level. COMP-14 obtained a stronger result with a native-output task-weighted *directional* statistic, while changing the readout type. COMP-15 isolates the factors through a 2 × 2 experiment.

For naturally reached recurrent states hA and hB, define the initial contrast Δ=hA−hB. The two states receive the same independently sampled five-step diagnostic token sequence. Let J(s,t) be the GRU transition Jacobian at time t for state member s, and Φ(s,0)=I, Φ(s,t+1)=J(s,t) Φ(s,t).

Two continuous four-output readouts are compared:
- Learned C=L is W, the fixed 4 × 24 linear behavioral probe.
- Native C=N is the Jacobian of all four actual counterfactual query-step GRU outputs, including the original nonlinear answer head.

For either readout stack the six blocks C(s,t) Φ(s,t) for t=0,...,5, forming O(s,C). The **rank-deficit** feature is

\[
R_C=24-\frac12\left(\operatorname{rank}_{10^{-3}}O(A,C)+\operatorname{rank}_{10^{-3}}O(B,C)\right).
\]

The relative rank threshold 10⁻³ was frozen in the protocol. The **directional sensitivity** feature is

\[
D_C=\left[\frac12\sum_{s\in\{A,B\}}\|O(s,C)\Delta\|_2^2\right]^{1/2}.
\]

Larger values were registered as predicting greater chance of delayed divergence. Both statistics use the *same observation matrix* within a readout condition. The primary hypothesis compares AUC(D_L) and AUC(R_L), isolating the statistic at fixed learned-readout access. Native rank, native direction, and the four-cell interaction are prespecified secondary analyses.

The unweighted directional statistic was chosen before the experiment because COMP-14's boundary-weighting ablation had shown no additional predictive benefit. Norm-normalized directional scores D_C/max(||Δ||,10⁻⁹) and raw state distance were prespecified secondary controls.

## Data and checks

Calibration models: seeds 101 and 202. **New holdout models: seeds 707, 808, 909 and 1001.** The earlier COMP-13/14 holdouts 303,404,505,606 were excluded. All GRUs use the unchanged architecture, training schedule and initial task-goal embedding. The probe learns the native model's counterfactual behavioral outputs without task-label supervision.

At recurrent depths 12 and 40, each model yields 16 goals × 16 naturally reached state pairs per goal. Within each pair the members share the same synthetic task but have independently sampled input prefixes. Eligibility requires matching current native four-query binary signatures and matching learned four-bit probe codes.

The **outcome** is native four-query signature disagreement after an independently sampled, shared 32-step input continuation. No diagnostic feature uses outcome tokens or outcomes. Across the four new model seeds there were 2,048 generated state pairs; **1,534 were eligible and 118 diverged**. The calibration sample had 819 eligible pairs and 60 divergences.

The analytic native-output Jacobian was compared with PyTorch autograd for every model. Maximum absolute discrepancy was **2.86102294921875×10⁻⁶**, within the preregistered 10⁻⁵ tolerance.

## Results: four factorial cells

| Readout | Effective rank ROC-AUC | Directional ROC-AUC | Directional minus rank |
|---|---:|---:|---:|
| Learned four-output probe | 0.484385 | **0.710225** | **+0.225840** |
| Native task-output Jacobian | 0.440053 | **0.722051** | +0.281998 |

The preregistered learned-probe improvement is positive in all four fresh models.

| Model seed | Eligible pairs | Divergences | Learned rank AUC | Learned direction AUC | Improvement |
|---|---:|---:|---:|---:|---:|
| 707 | 362 | 44 | 0.489601 | 0.682461 | +0.192860 |
| 808 | 424 | 24 | 0.578594 | 0.785208 | +0.206614 |
| 909 | 362 | 24 | 0.407421 | 0.675419 | +0.267998 |
| 1001 | 386 | 26 | 0.480876 | 0.668803 | +0.187927 |

Removing any one held-out model leaves an improvement between +0.210279 and +0.237164. The primary preregistered support rule required a positive paired-bootstrap lower bound and improvements in at least three of four model seeds. Both criteria were met.

The 1,000-permutation test shuffled outcomes within each model/depth cell and evaluated the primary *paired AUC difference*, centered on the stratified permutation-null mean 0.019203. The two-sided p-value was 0.000999. The corresponding 1,000 within-model/depth bootstrap resamples gave the primary AUC-difference interval **[+0.158064,+0.292475]**. The learned directional AUC itself had a 95% interval [0.661270,0.759270].

These uncertainty estimates describe sampling variability among pairs from four fixed trained models. They are not a confidence interval across unknown model families.

### Readout effect and interaction

At a fixed directional statistic, native-minus-learned AUC was **+0.011826**. At fixed rank, native-minus-learned was **−0.044332**. The four-cell interaction on the AUC scale was +0.056158, with substantial model variability.

This factorial experiment supports the view that **directional sensitivity matters when readout access is fixed**. It gives less support for extra native readout access as the explanation of the COMP-14 gain. Those secondary contrasts should not be treated as separately confirmed hypotheses.

### Direction versus magnitude

| Held-out score | ROC-AUC |
|---|---:|
| Learned directional sensitivity | 0.710225 |
| Learned directional divided by pair distance | 0.673525 |
| Native directional sensitivity | 0.722051 |
| Native directional divided by pair distance | 0.693234 |
| Raw Euclidean state distance | 0.514459 |

The drop following norm normalization indicates that contrast magnitude contributes to prediction, while the normalized scores retain information consistent with how the contrast aligns with task-relevant output directions.

## Frozen receipts and exact AUC verification

The [executable GRU experiment](experiments/factorial_observability.py) reproduces the four readout/statistic cells from frozen seeds, performs paired stratified permutation and bootstrap analyses, and emits all state-pair rows. The [frozen compact result](results/comp15_summary.json) stores model/probe hashes, exact pair-count AUC receipts, seed breakdown, and uncertainty. The [pure-standard-library audit](factorial_auc_audit.py) reconstructs seven ROC-AUC values independently, while the [finite 2 × 2 example](factorial_reference.py) demonstrates two rank-one readouts with different directional energies. The [Lean proof](lean/Shaft/TaskFactorial.lean) formalizes this finite example.

AUC was independently verified using exact positive-negative pairs. On held-out data, learned directional scores won 118,670 of 167,088 positive-negative comparisons without ties. Learned rank scores had 62,941 strict wins and 35,988 ties. These produce the displayed AUC values exactly up to rounding.

The two full 3,072-pair JSON executions (1,024 calibration, 2,048 new held-out) were byte-identical under PyTorch 2.10.0+cpu and NumPy 2.3.5, deterministic torch settings and three CPU threads. Both outputs have SHA-256:

c9c4ec664780be6ffa631748edb6d64ab338011b4132212f93210150ce38c667

To reproduce, from repository root run Python unit tests, the finite reference, the AUC audit, and then the optional experiment using the command:

python -m research.shaft_compactification.experiments.factorial_observability --per-goal 16 --permutations 1000 --bootstraps 1000 --out /tmp/shaft_comp15_full.json

The standard-library tests and Lean 4.28 kernel build run in lightweight CI. Full CPU retraining is an opt-in task and stays out of rapid PR CI.

## Scope and next falsifier

The experiment uses a **known same-goal pair contrast** and an observable frozen recurrent state. The measured directional sensitivity may help design defensive checks on locally accessible model internals, but its performance is specific to synthetic truth-table goals and the two selected recurrent depths. More varied task families, partial measurement access, and new model architectures require fresh preregistration.

The original Q5 F1/F8 parameter-gradient fragility ratio remains mathematically distinct. Its twelve reference checkpoints must be verified and made available before direct cross-benchmark comparison.
