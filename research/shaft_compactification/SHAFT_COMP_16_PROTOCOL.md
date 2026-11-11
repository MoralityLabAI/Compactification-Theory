# SHAFT-COMP-16: Cross-task-family directional observability with fixed FPR

**Preregistered 2026-11-11, before running new task-family pilots.** Source thresholds use only the published COMP-15 calibration negatives, and every new family, seed, feature, horizon and decision rule is fixed here. This remains an isolated synthetic study with intact model weights.

## New task dynamics

The prior truth-table objective is a function of the current query alone. Both new families add one-step input history. For goal g in 0..15, input x_t in 0..3, previous input x_(t-1), with sentinel x_(-1)=0:

- lag_rotate: output is goal bit indexed by (x_t+x_(t-1)) mod 4.
- lag_match: output is goal bit indexed by x_t, XOR the indicator x_t=x_(t-1).

Each new model uses the unchanged 24-hidden-unit TaskMemoryGRU architecture and initial goal embedding. Train using Adam 1200 steps, batch 128, length 12, LR 0.01, gradient clip 2.0. Train the four-output linear predictive probe on frozen-model native counterfactual query responses at state depths 4,8,12, using 48 contexts per goal per depth and Adam 400 steps at LR 0.04. Report fresh-sequence 12-step task accuracy; include all seeds, including weakly trained models.

Model seed assignments: lag_rotate calibration 1101/1102, heldout 1103/1104; lag_match calibration 1201/1202, heldout 1203/1204. Earlier SHAFT test seeds are excluded.

## Identical monitoring and outcome

For every model and state depth (12,40), sample 16 same-goal pairs per each of 16 goals (256 pairs per depth/model); each pair uses independent prefix sequences. Eligibility requires matching current native four-query output bits AND matching four probe bits. Use a five-token diagnostic continuation shared within each pair, independent of state prefixes. Construct the learned-probe observability stack O_5 using native GRU transition Jacobians, for both pair members. The pair contrast is delta=hA-hB.

Primary predictor D_L=sqrt((||O_A delta||^2+||O_B delta||^2)/2), larger predicts divergence. Matched baseline rank deficit R_L=24-(rank_0.001(O_A)+rank_0.001(O_B))/2, larger predicts divergence. The relative numerical rank tolerance is 0.001. Also report norm-normalized D_L and raw state distance as secondary controls. The separate shared 32-token outcome continuation is independent of diagnostic inputs; Y=1 when the native four-query signatures disagree after those 32 steps. No outcome information enters features.

## Fixed-FPR transfer: preregistered source thresholds

Derived once from COMP-15 calibration negative state pairs (759 negatives; 90th percentile with NumPy quantile method higher, predict only for score strictly greater than threshold):

- D_L threshold = 6.854760490520719. Source calibration FPR = 75/759.
- R_L threshold = 2.5. Source calibration FPR = 68/759.

Apply thresholds without tuning to each new family heldout and aggregate. Report FPR and TPR and any calibration-domain shift. A *source-threshold transfer success* requires D_L heldout FPR <=0.15 separately for both new families. A family can fail this criterion even when discrimination AUC is good.

Secondary adapted threshold: 90th percentile (method higher, strictly greater) of negative scores on each family's **calibration models only**, separately for D_L and R_L; evaluate heldout. This is a distinct setting.

## Confirmatory evaluation and decision

Primary AUC difference = AUC(D_L)-AUC(R_L) on pooled new-family heldout eligible state pairs. Report family/model/depth AUCs and label prevalence, plus raw pair distance and normalized D_L. Use 1000 bootstraps resampling eligible pairs within family×seed×depth strata for a paired percentile 95% CI, and 1000 label permutations within the same strata for a two-sided p-value centered on its stratified null mean.

Transfer is supported in this synthetic benchmark iff the pooled AUC difference bootstrap 95% CI lies strictly above zero, BOTH family-pooled differences are positive, and at least three of four heldout model-seed differences are positive. Preserve negative results and do not reverse score orientations or retune thresholds. Report uncertainty as within the four fixed trained models.

Record independent model and probe hashes, actual training accuracy and full per-pair JSON. The experiment remains CPU PyTorch opt-in; standard-library and Lean tests run in lightweight CI. New weights remain intact. This cannot establish unknown hidden objectives, distributed persistence, or checkpoint reconstruction. The Q5 checkpoints remain unavailable for direct cross-benchmark measurement.
