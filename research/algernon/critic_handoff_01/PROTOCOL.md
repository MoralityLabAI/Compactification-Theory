# Proposed critic-transfer experiment

The initial bridge is to an existing finite storyworld or algebraic-machine state space with an independently computed value oracle. Keep learned features and reward labels separate from protected authorization metadata. Preflight must resolve the current integration files and checkpoint provenance before any learned run.

Use a 2x2 intervention: unchanged/changed actor policy crossed with unchanged/changed reward. The comparison includes four critic treatments: full reset plus calibration, unchanged inherited critic, inherited critic plus calibration, and inherited features with a reset readout plus calibration. The unchanged inherited critic is a diagnostic; the calibrated arms receive equal optimizer updates and identical fresh on-policy data. Report total acquisition cost, so inheritance never receives hidden extra rollouts. Add a component-wise/distributional critic as a preregistered extension after the base comparison.

Freeze the actor during calibration. Predeclare checkpoints at 0/1/2/4/8/16 calibration updates and choose the stopping rule using validation only. Four updates are an experimental coordinate, not a guarantee. Use five paired initialization seeds [11,23,37,53,71]. Report exact sup error on the finite oracle, deployment-distribution error, strict-success Brier score, advantage-sign disagreements and downstream actor regret. Evaluate calibration on newly generated data from the post-rewrite policy.

Cross reward-only reweighting against policy-only drift. The component-wise critic should gain under a pure reward readout change; that result alone does not justify carrying it through an actor rewrite. Include the adverse reward-flip control and report any inherited-worse-than-reset cells.

Compression screen: compare a public-feature state, a structured state with host-validated authority fields, and the existing quotient oracle. Search for pairs with the same compressed code and different authorized successors or continuation values. A witness rejects the compression contract even if average prediction error is small.

Promotion: beat the full-reset and reset-readout baselines on newly sealed families at the same data/update budget, with no degradation in protected strict completion. Parameterize a practical minimum effect before running; proposed screen is a >=20% reduction in post-calibration MSE and a positive paired interval. Both criteria must hold. The experiment remains bounded by CPU-first toy models; scaling to an existing local 5080 model requires a separate approved run.

Current status: exact reference complete; learned comparisons UNRUN. No archived checkpoint download, GPU job, CI edit or automatic merge.
