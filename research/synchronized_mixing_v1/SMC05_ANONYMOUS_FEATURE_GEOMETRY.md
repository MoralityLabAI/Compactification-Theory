# SMC-05: Discovering anonymous sparse geometry before frame calibration

November 15, 2026. Companion pure mathematics: [MathCan OLM-05](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM05_SPARSE_GEOMETRY_RIGIDITY.md).

## New experiment

SMC-04 received named graph matchings and knew the vertex-permutation action; it inferred only the hidden frame and reference matching. **SMC-05 first reconstructs the six-vertex edge geometry from anonymously permuted sparse activation channels.** The estimator receives binary masks with three active channels per state, without the channel-to-edge dictionary, vertex labels, reference matching or frame.

The structural domain prior remains explicit: six latent vertices, fifteen edge-like binary features, and perfect matchings whose observations activate three disjoint features. Grouping repeated noisy measurements of the same state episode is supplied. The learner discovers the **specific geometry within that model class**, and from it derives the candidate vertex symmetry S6. This is synthetic structured learning, not discovery of arbitrary neural representations or a trained SAE.

The external research inspiration remains OpenAI Mathematics [#131](https://github.com/openai/math/blob/main/lean/docs/131.md) on fixed-degree graph switches and [#238](https://github.com/openai/math/blob/main/lean/docs/238.md) on permutation mixing. Our finite inference algorithm is distinct from the original papers' Markov kernels.

## Information and leakage contract

1. The host chooses hidden output frame σ, reference matching B and an independent random permutation of the 15 **feature channels**. An input permutation S yields graph G=σSB, emitted as a scrambled three-hot activation vector.
2. The geometry estimator receives **12 distinct anonymous training-state vectors**, grouped into episodes when sensor noise is present. None of the three withheld matching classes appears even as an unlabeled geometry-training example.
3. The estimator builds the *non-coactivation* graph on channels, enumerates five-cliques, and searches for a **global six-star cover**: six size-five cliques, each pair meeting in exactly one channel, and each channel belonging to exactly two cliques. It abstains if zero or multiple covers satisfy these requirements.
4. Each latent channel then becomes the abstract edge between its two recovered star/vertex labels. Anonymous matching masks can be decoded into graphs on six reconstructed vertices.
5. Only after geometry fitting does the solver construct the possible S6 group actions and use SMC-04's 10,800-candidate frame/reference model. It sees twelve sampled positive calibration pairs from the 576 train-cohort input permutations; the host secret is absent from the inference function.
6. The model is evaluated on all 144 permutations whose graphs belong to the **three completely withheld matching classes**, against the exact matched-marginal independent-graph null. A correct inferred relation achieves balanced accuracy **5/6**.

The host selects calibration-compatible permutations conditional on its hidden model. As in SMC-04, this is a truth-conditioned observation cohort; the independent-query information-counting lower bound does not apply to that cohort. A separate evaluation-only oracle compares the inferred parameters to the unique model satisfying *all* 576 training observations. That oracle never enters the twelve-example hypothesis selection.

## Complete finite geometry certification

The [MathCan exact enumerator](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/latent_stars.py) exhausts **all 455 three-matching holdouts**. A naive five-clique count is unique in only 80 cases. The other 375 contain 13, 24 or 47 candidate five-cliques, but the complete global star-cover constraints select exactly one in **all 455**.

With four heldout matching classes and eleven observed, **1,320/1,365** deletion choices retain a unique geometry, and the other **45** have exactly two valid geometries. The 45 ambiguous omissions consist of two disjoint pairs of omitted matchings, each sharing an edge, with no cross-pair shared edge. The estimator rejects these rather than arbitrarily committing to a topology.

The default end-to-end fixture intentionally omits classes **(12,13,14)** from the canonical list. They all contain edge (0,5), so the corresponding anonymous feature is **never active in training**. The coactivation graph therefore has 69 absent-pair/conflict edges and 47 candidate five-cliques; its unique global cover still locates the completely silent feature. This is a more demanding case than the six-clique direct reconstruction.

The recovered vertex labels are arbitrary: changing all feature-channel indices merely permutes the six reconstructed stars. Exact tests verify equivariance under several independently scrambled feature dictionaries.

## Executed local CPU experiments

Three hidden-frame seeds (43,77,91), three independent channel scramblings and twelve sampled positive calibration pairs produced a **singleton frame/reference candidate** in each run. Every recovered model achieved the **5/6** Bayes balanced-accuracy limit on all three previously excluded matching classes. Three additional successful end-to-end tests used 40 repeated sensor readings per training state at sensitivity 0.75, false-positive probability 0.25.

Geometry reconstruction was separately tested with 100 fixed seeds per noise condition. It was accepted only if the full six-star certificate passed; fixture-side comparison of star sets verified every accepted reconstruction against the true geometry up to vertex relabeling.

| True-positive p | False-positive q | Sensor repeats/state | Correct and certified / 100 | Rigorous population success lower bound |
|---:|---:|---:|---:|---:|
| 0.90 | 0.10 | 16 | **100/100** | 99.694% |
| 0.75 | 0.25 | 40 | **95/100** | 95.524% |
| 0.75 | 0.25 | 64 | **100/100** | 99.882% |

The analytic lower bound is the exact binomial probability that all 36 active bits and 144 inactive bits across the twelve hidden training states are correctly decoded by majority vote. It assumes independent Bernoulli sensor noise. The 100-seed frequencies are finite synthetic measurements, not confidence intervals or estimates for arbitrary feature encoders.

### Row- and column-preserving null

A negative control performs 4,000 random 2×2 switches on the 12×15 binary activation matrix, preserving every per-state row sum **and every per-feature column total**. It scrambles coactivation structure without changing trivial activation counts. **0/100** seeded switched matrices passed the global six-star geometry certificate. This checks a stronger null than independent feature shuffling, while remaining a bounded finite test.

## Reproduce

Standard-library-only, no pretrained checkpoints, torch or network calls:

    python -m unittest discover -s research/synchronized_mixing_v1 -p 'test_discover_geometry.py' -v
    python research/synchronized_mixing_v1/run_discovery.py --output smc05-smoke.json
    python research/synchronized_mixing_v1/run_discovery.py --full --output smc05-full.json

The [source](discover_geometry.py) isolates geometry inference from fixture labels. The [pipeline](run_discovery.py) runs training-only recovery, the twelve-pair calibration, heldout class scoring, sensor-noise sweeps and row/column-preserving null. The scoped GitHub Actions workflow executes the exact tests and smoke replay; the [committed full results](smc05_full_results.json) include all three clean frames, three denoised replays, 300 sensor trials and 100 null matrices.

## Interpretation and next experimental gate

The experiment discovers **which fifteen anonymous channels form the edges of which six abstract vertices**, then infers the remaining unknown frame/reference parameters with the SMC-04 finite-group algorithm. A real mechanistic-interpretability application must learn the relevant structural class from partially entangled SAE or transformer activations, tolerate different feature counts and intervention identities, and transfer causal tests across loop depths. Neither a pretrained SAE nor such a model-derived representation has been tested here.
