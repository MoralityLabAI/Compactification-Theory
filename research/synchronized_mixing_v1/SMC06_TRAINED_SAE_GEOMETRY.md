# SMC-06: Trained sparse-autoencoder features, geometry recovery and a sharp one-swap barrier

**November 15, 2026.** Continues [SMC-05](SMC05_ANONYMOUS_FEATURE_GEOMETRY.md). Pure mathematics and exhaustive combinatorial certificates: [MathCan OLM-06](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM06_APPROXIMATE_SAE_RIGIDITY.md).

## Status and research boundary

SMC-05 reconstructed six vertices from ideal binary three-hot observations. This stage uses **actual learned sparse-autoencoder activations** from a compact 40→15→40 ReLU SAE trained by reconstruction and activation sparsity penalties. The source of those activations is a **synthetic dense teacher**, not a pretrained language model. Only twelve of the fifteen matching classes appear as complete matching patterns in SAE training. The learner receives the dense mixtures, not the planted edge dictionary, matching IDs or group-frame labels.

A finite experiment has now demonstrated the essential distinction between achieving low reconstruction error and preserving a rigid, verifiable feature geometry. The model can reconstruct dense vectors accurately while representing some edge atoms with merged or split SAE channels, leading the exact geometry checker to abstain.

The external mathematical sources motivating the project remain OpenAI Mathematics [#131](https://github.com/openai/math/blob/main/lean/docs/131.md) on graph-switch mixing and [#238](https://github.com/openai/math/blob/main/lean/docs/238.md) on full-permutation mixing. This synthetic experiment uses its own random dictionaries and transport simulator. No source mixing bound is transferred to a trained model.

## 1. Neural and data protocol

Frozen synthetic teacher: 15 latent atom vectors in 40 continuous coordinates, randomly generated and normalized using NumPy seed **7301**. Each observed dense vector is a nonnegative sparse combination of planted atom vectors plus iid Gaussian noise with standard deviation 0.015 per dimension. A fixed 16,000-example training population uses 70% singleton-atom exposures, 24% independent pairs and 6% three-atom perfect-matching observations drawn **only from matching classes 0–11**. Atom identities and matching labels are not supplied to the SAE trainer. Singleton and pair exposures cover all 15 atoms, including an edge that is silent in the twelve matching-class observations. They supply essential unlabeled dictionary information.

SAE: `Linear(40,15) -> ReLU -> Linear(15,40, bias=False)`, **1,215 trainable parameters**. Train with Adam learning rate 0.003 for 9,000 steps per seed, minibatch 256, and a loss equal to squared L2 reconstruction error per observation plus 0.0015 times the sum of nonnegative SAE features. Decoder columns are normalized after each update. ReLU activations, not planted atoms, are used for all geometry and holdout inference.

Measurement: after freezing the network, for each of 15 matching-state fixtures average 64 independent noisy readings of the SAE code. **Only the 12 permitted training matching-state averages** enter geometry reconstruction. The three excluded averages enter evaluation only, after the geometry and hidden-frame calibration have been fixed. Calibration still uses 12 positive input-permutation / decoded-graph pairs sampled from the truth-conditioned 576-input train cohort, as documented in SMC-04; the earlier independent-query counting bound does not apply to this cohort.

Environment: Python 3.13.5; PyTorch 2.10.0+cpu; NumPy 2.3.5; two CPU threads. No hosted training or GPU is required. Run seeds 101–110 from [sae_train.py](sae_train.py), and retain the source-pinned [seed-103 checkpoint](sae_seed103_checkpoint.json) for direct inspection.

## 2. Unknown feature coordinates and bounded structural certification

[approx_geometry.py](approx_geometry.py) receives only the **15 anonymous SAE feature-score coordinates**. The perfect-matching model class is a stated prior. The program infers vertex count n=6 from the observed feature count C(n,2)=15 and three active features per matching; it reconstructs the stars via the global K6 incidence constraints and then enumerates 720 automorphisms of the recovered graph. The **specific** feature-to-edge map and induced vertex labeling remain unknown before inference. This is group identification *within the promised complete-graph matching class*, not unconstrained discovery of symmetry from general activations.

The first candidate mask for each training matching-state average consists of its top three SAE feature scores. The model attempts a unique six-star cover as in SMC-05. If this fails, it tries at most one **active-to-inactive channel replacement in one of the twelve rows**, enumerating all 432 possibilities. If exactly one candidate yields a valid six-star cover, it returns that correction. Multiple valid repairs, exhausted search budgets, or missing geometries cause explicit abstention. The worst-case finite search-node count is bounded; all 455 known legitimate twelve-state cohorts fit within the limit of 1,000 nodes, and the program never silently treats a truncated search as unique.

A structurally consistent twelve-row output certifies its *observed incidence data*. It is not automatically a certificate of the original planted dictionary. The exact OLM-06 alias audit establishes that 180 of 455 twelve-state holdouts permit a single feature substitution that produces an **alternative valid geometry**. All 180 aliases can preserve equivariant matching labels on the training cohort. Therefore even a mathematically unique six-star cover may correspond to a corrupted input. OLM-06 proves that thirteen distinct true matching states suffice to detect any one such substitution, and twelve do not provide a worst-case guarantee.

For the particular classes 12–14 withheld in this experiment, a one-swap error in seed 101 led to an **invalid** row collection. The bounded search found exactly one repair: row 8, learned channel 14 replaced by channel 8. A planted-geometry audit confirms that the repaired model is correct up to vertex relabeling. This audit is kept separate from the inference algorithm.

## 3. Ten-seed CPU training results

The complete machine-readable record is [smc06_full_10seed_results.json](smc06_full_10seed_results.json). The supported replay cases are [sae_activation_evidence.json](sae_activation_evidence.json): seed 101 (single repair), seed 103 (clean reconstruction) and seed 105 (abstention). All results use the same architecture, training generator, optimizer and 9,000-step budget.

| SAE seed | Mean train squared L2 reconstruction error | Geometry gate | Repairs | Heldout balanced accuracy after frame calibration |
|---:|---:|---|---:|---:|
| 101 | 0.02499 | Valid after unique repair | 1 | **13/18 = 72.22%** |
| 102 | 0.02486 | Valid | 0 | **5/6 = 83.33%** |
| 103 | 0.00607 | Valid | 0 | **5/6** |
| 104 | 0.00610 | Valid | 0 | **5/6** |
| 105 | 0.05576 | Abstain | N/A | Not scored |
| 106 | 0.02487 | Abstain | N/A | Not scored |
| 107 | 0.00607 | Valid | 0 | **5/6** |
| 108 | 0.02484 | Abstain | N/A | Not scored |
| 109 | 0.00611 | Valid | 0 | **5/6** |
| 110 | 0.02501 | Abstain | N/A | Not scored |

Six of ten seeds yield a **unique consistent geometry with ≤1 repair**, and all six are verified against the planted feature incidence up to vertex relabeling. Five of the six accepted SAEs decode all three unseen matching classes correctly and reach the Bayes limit; seed 101 decodes two of three and reaches 13/18. The accuracy **conditional on acceptance** is **22/27 ≈ 81.48%**, with **60% coverage**. The four abstained seeds remain in every aggregate. A second, exploratory 9,000-step training continuation recovered one additional formerly failing seed (108), while tested seeds 105, 106 and 110 still abstained. This continuation uses a reinitialized optimizer and is reported separately from the fixed 9,000-step sweep.

For each accepted feature model, the previously developed SMC-04 frame solver recovers exactly one group-frame/reference parameter pair using twelve calibration pairs for each of three fixed hidden-frame seeds (43, 77, 91). These eighteen calibration tasks share only six learned feature models; they are not eighteen independent SAE-training replications. The withheld positive/negative class marginals are matched by construction, including after a mistakenly decoded heldout state is represented by its wrong matching label.

## 4. Baselines and controls

Three seeds were tested with an untrained fifteen-unit ReLU encoder, and with 15 principal-component coordinates followed by absolute-value and top-three extraction. Both baselines abstained in all three cases. The PCA subspace had **lower dense reconstruction squared L2 error (0.00563)** than several trained SAE runs that recovered geometry. This demonstrates in this bounded setting why dense reconstruction error is insufficient evidence for mechanistically useful sparse coordinates.

An **anchor ablation** trained the same SAE on only the twelve allowed complete matching classes, removing singleton and pair exposures. Both tested initializations (101 and 102) abstained. OLM-06 gives a formal explanation for one necessary part of this behavior: a planted atom that is absent from *every* training latent vector leaves its decoder direction statistically unidentifiable, regardless of training-set size. The small two-seed ablation alone does not establish a universal failure rate.

The exact combinatorial reference tests check all 455 clean twelve-state holdouts, the one-swap adversarial example with a *different valid geometry*, all 105 thirteen-state holdouts, coordinate-gauge permutations, the frozen SAE cases and heldout calibration for three frame seeds. The new pure-math test suite additionally enumerates all **180** single-swap aliases and proves the 13-state detection theorem conditional on the established 455-case rigidity classification. Only these finite cases and assumptions are certified.

## 5. Reproduction

Standard-library replay (no Torch installation):

    python -m unittest discover -s research/synchronized_mixing_v1 -p 'test_approx_geometry.py' -v

CPU training (PyTorch and NumPy are required):

    python research/synchronized_mixing_v1/sae_train.py --seeds 101 102 103 104 105 106 107 108 109 110 --steps 9000 --out smc06-train.json
    python research/synchronized_mixing_v1/sae_train.py --seeds 101 102 --no-anchors --out smc06-noanchors.json
    python research/synchronized_mixing_v1/sae_controls.py

GitHub Actions reruns the exact frozen-activation tests and archives the ten-seed source-bound results and the immutable reference checkpoint. The remote CI **does not retrain the SAE** because of the larger PyTorch dependency; the training script and source seeds are provided to reproduce it locally. The per-run JSON includes the loss checkpoints, feature geometry gate, calibration outcomes and explicit reason for abstention.

## 6. Next scientific gate

The logical next step is to replay the **same estimator** on a source-pinned tiny looped-transformer or TRM checkpoint with a trained SAE from real hidden activations. That must specify the activation layer, checkpoint hash, SAE dictionary version, heldout prompt families, controls for arbitrary feature permutation, and causal interventions. The present planted sparse teacher shows that learned SAE representations can preserve enough matching incidence for a strict geometric certificate and hidden-frame recovery. It does not imply that a pretrained language model has a K6 matching geometry or that such a probe improves model capability.