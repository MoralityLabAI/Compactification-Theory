# ALG-DESCENT-07: cross-family risk and shared-factor intervention

DataGoblin PR #89; protocol 3da5bee751ef7e4d5e8b9d810418f9896ab37174, experiment source 9e10106e409a3edbb4fffb5a969abbbb879755da. See docs/alg_descent_v7_results.md in DataGoblin for full scope. Synthetic certificate routing only.

96 train, 24 calibration, 24 IID and 32 held-out-family development roots. 2,240 rollouts all close. OOD log loss bucket .3968, logistic .3546, MLP .3330, recurrent factor .3256. Factor vs MLP improves 2.2% (<5% gate), descriptive paired bootstrap crosses zero. The second-difference family favors bucket. Mean route work logistic 988.47, factor 989.52; routing gate fails. Neural inference increases warmed latency.

Calibrated rank-one deletion increases OOD log loss .00154 pooled, two-child loss decreases .00102, median matched-sham damage -.000085. Causal gate fails; ablation/enhancement exactly tie intact routing work. Only 6/96 intact OOD factor routes had a rejection. Initial h=0 makes recurrent-matrix edits unable to affect first proposal. Gauge W=UV=(UA)(A^-1 V) also prevents end-to-end factor labels from identifying behavioral components absent extra constraints.

Independent sampled audit: 10,907 eligible algebraic identities; 640 checkpoint metric fields agree within 5.96e-8; 32,721 zero-history comparisons exactly equal; all routing receipt budgets checked. No independent full routing replay. This was a shared-factor pilot, not native BPD. Next: initialize recurrent state from public features before first attempt, preregister minimum exposure, separate forced histories and natural routes, match reset and activation perturbation controls, use frozen-model behavioral-minimality decomposition. No MMS integration or DLP inference.
