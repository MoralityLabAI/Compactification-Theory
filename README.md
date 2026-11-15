**NOR-12:** [Task-supervised basin-conditioned residual readout](research/nor12_task_readout/README.md) fits four 16×14 matched residual rankers over the frozen 1,000-parameter RRC-25 GRU using only exact degree-192 public demo verification labels. The first heldout run yielded 44/72 future-correct for the basin residual vs. 42/72 original, with both gains arising from a single aliased late-source feature tensor. Four independent seed-separated 72-case confirmations yielded 167/288 vs. 160/288, matched by the random-feature control, with unadjusted case-level paired p≈0.21. An ablation of the basin feature left the first split's 44/72 result unchanged. [NOR-12B hosted independent results](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43709095616) passed, and [NOR-12C frozen receipt audit](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43709504979) is green. The original GRU is unchanged, and basin-specific task gains remain unestablished.

**NOR-10:** [Constructive one-bit terminal metric for two source-certified GRU attractors](research/nor01_numerical_orbit_sketch/NOR10.md) uses a single shared rational inter-center distance and one runtime basin-label bit to represent all late 24-step states of the signed and squared source streams. Exact integer-square-root quantization and source-locked nonlinear contraction yield an all-pair late-tail error <=1.096e-11 after 64 retained blocks, while a zero-bit shared code must incur >5.0209 cross-basin error. This one-dimensional terminal representation requires the two previously certified source labels and is separate from unseen-input basin classification and native task performance. [NOR-10 CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43704307173) passed.

**NOR-09:** [Certified asynchronous two-source history compactification](research/nor01_numerical_orbit_sketch/NOR09.md) joins the two separately verified public-input GRU basin captures. The 24-step sampled signed and squared trajectories admit distinct rational-anchor late-state clipping with a certified all-pair error below 1.2e-11 after 64 additional blocks per source, conditional on the observed finite-prefix embedding error. A shared terminal code would incur cross-basin distance error exceeding 5.0209113; the paired [MathCan asynchronous stitching theorem](https://github.com/patrickdugan/MathCan/tree/research/nor09-asynchronous-two-basin-history-20261115/research/nor09) formalizes its discrete metric core. These results apply to the real-analytic zero-input trained GRU and leave native task outcomes for separate experiments.

**NOR-08:** [Enlarged second trained-GRU basin, source capture and basin-label controls](research/nor01_numerical_orbit_sketch/NOR08.md) certifies a radius-1e-6 ball around the second attracting fixed point, with the true 24-step block derivative bounded by 0.8303095. Squared-family source FP64 first crosses the region at step 651, and an independent 55-block **exact-rational capture certified entry at step 660** ([verified run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43700133484)). A 72-case held-out partition and intervention probe finds both recurrent modes across families, while the [MathCan NOR-08 theorem](https://github.com/patrickdugan/MathCan/pull/60) proves that m sufficiently separated attractors require at least m terminal-code labels. [NOR-08 executed radius sweep](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43699717963).

**NOR-07:** [Two distinct certified attracting fixed points in the original trained GRU](research/nor01_numerical_orbit_sketch/NOR07.md), both period one for the same zero-input map. [NOR-07 CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43698679035) passed 12 source-locked rational checks. The paired [MathCan multibasin obstruction](https://github.com/patrickdugan/MathCan/pull/56) and [noncommutative Volterra theorem](https://github.com/patrickdugan/MathCan/pull/58) are merged.

**NOR-06:** [Signed-family public-input basin entry certified after 1,308 zero-input steps](research/nor01_numerical_orbit_sketch/NOR06.md), using 109 source-locked exact-rational 12-step interval products. [Verified capture](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43695479529) and [MathCan finite-entry theorem](https://github.com/patrickdugan/MathCan/pull/54) establish the local all-depth nonlinear convergence consequence.

**NOR-05:** [Trained GRU period-one fixed point and full-depth nonlinear convergence](research/nor01_numerical_orbit_sketch/NOR05.md) closes the twelve-cycle ambiguity of NOR-04. Its exact-rational one-step containment bound is 2.82488314450335e-14 within the same radius-1e-11 ball; [hosted CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43693151218) passed 4/4 tests. The paired [MathCan theorem](https://github.com/patrickdugan/MathCan/blob/research/nor01-orbit-stitching-20261115/research/nor05/PERIOD_ONE_NONLINEAR_COMPACTIFICATION.md) extends nonlinear Lp history compactification to every single recurrent depth. The certified basin is local; natural input states have not been shown to enter it.

**NOR-04:** [Trained GRU full nonlinear invariant-ball certificate](research/nor01_numerical_orbit_sketch/NOR04.md) proves that the actual twelve-step zero-input recurrent map is a contraction on a rigorously enclosed radius-1e-11 ball: uniform derivative bound K<=0.928242792263, center residual <=1.186388155e-15 and positive exact invariant margin. [GitHub Actions run 43691939525](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43691939525) passed; the [MathCan theorem](https://github.com/patrickdugan/MathCan/blob/research/nor01-orbit-stitching-20261115/research/nor04/NONLINEAR_INVARIANT_BALL.md) derives all-depth nonlinear Lp history compactification locally. Basin entry for arbitrary RRC-25 inputs and task accuracy remain open.

**NOR-03:** [Source-locked trained GRU nonlinear block certification](research/nor01_numerical_orbit_sketch/NOR03.md) proves a rational-interval disk enclosure for a 12-step derivative of the real RRC-25 trained GRU, checks early amplification and later contraction, and bounds the full-block quadratic Taylor remainder. Six independent public test-family continuations and a bounded two-case deeper-warmup check are recorded with explicit successful/failed certificates. The [pure-math product and shadowing theorem](https://github.com/patrickdugan/MathCan/blob/research/nor01-orbit-stitching-20261115/research/nor03/TRAINED_BLOCK_JACOBIAN.md) records the analytic transfer limits.

**NOR-02:** [Noncommuting matrix-valued recurrence filters and curvature-aware Lp history sketches](research/nor01_numerical_orbit_sketch/README.md#nor-02-matrix-valued-curvature-filter-15-november-2026) extend NOR-01 to 128-to-24 states with exact rational disk geometry, a (t^{-r/2}) spectral-tail improvement, and the [MathCan filtered-orbit theorem](https://github.com/patrickdugan/MathCan/tree/research/nor01-orbit-stitching-20261115/research/nor02). Ten additional exact/empirical controls isolate the linear surrogate from trained-model claims.

**NOR-01:** [Numerical-range-gated recurrent Lp orbit sketches](research/nor01_numerical_orbit_sketch/README.md) combine OpenAI Math #094 and #325 with an exact rational nonnormal tail gate, a reproducible 64-to-24 Gaussian L4 sketch, seven deterministic checks, and the [MathCan NOR-01 stitching theorem](https://github.com/patrickdugan/MathCan/pull/45). The finite pilot is a linear surrogate; analytic transfer to trained loops remains open.

# Compactification Theory

**RefutationBench v0.1:** [evidence-grounded proof audits and compact certificates](research/refutation_bench/README.md), with a pinned Riemann-submission case study, paired valid/invalid controls, exact finite certificate checks, and train-only distillation export. This public seed runs offline; learned reasoning gains remain an experimental target.

**RRC-06:** [raw-coefficient Euler-trace benchmark](research/rrc_compactification/RRC06_REPORT.md): CPU-trained recurrent/feedforward students, exact first-fault certificates, held-out grammar tests and negative complete-program result.

**RRC-05 research:** [finite-depth Euler inversion and trained residual circuit](research/rrc_compactification/RRC05_REPORT.md), with exact symbolic controls, source-family holdouts and a CPU-replayed result.

**New research track:** [Algernon: a neural-net processor, a learning computer](research/algernon/README.md), joining compiled Qwenputer-style exact computation, recurrent CoT/LDT grafts, and sparse weight-space repair. [Experimental protocol](research/algernon/PROTOCOL.md) and [agent workorder](research/algernon/AGENT_WORKORDER.md).

The new [Pluto-Gym research package](research/pluto_gym/README.md) adds a runnable
interface-relative gym, a compact CPU student with bounded DRAM replay, and
curriculum experiments with a configurable designer webhook. Its pinned
TensorCraft adapter reuses the existing StarCraft surrogate, frozen-Pluto farm
and trace checks.

How many bits does a trained adapter store per parameter, and what limits it? This repository tests a theory note on that question against LoRA adapters on Qwen3-0.6B. Everything runs on CPU in RAM.

The note, [`note/bits-depth-lipschitz.html`](note/bits-depth-lipschitz.html), works out these claims for ReLU networks approximating Hölder functions:

- **Total bits bound capability.** W weights of b bits can store at most W·b bits, so error ε ≳ (W·b)^(−s/d) for any model. No Lipschitz assumption is needed.
- **Per parameter, capability is not bounded by itself.** It is bounded only through the effective bits per weight, log2(R·K/ε). Here K is the Lipschitz constant of the map from weights to function.
- **For bounded weights, that effective precision is a depth quantity.** The classical W^(−s/d) barrier holds exactly when effective precision is O(log W). The W^(−2s/d) ceiling for ReLU comes from VC dimension, not from any Lipschitz constant.

Open the HTML in a browser; its maths is typeset with MathJax from a CDN. Every claim is labelled as proved, cited (opened or recalled) or conjectured. Its §4 lists predictions for small-network experiments, and its §5 reports this repository's results.

## The experiment

**Task.** M random keys (`"482910 ->"`), each labelled A or B at random. The labels are M fair bits, the language-model version of the note's random-sign packing. An adapter gets an item right when the logit of its label beats the other one.

**Exact and in RAM.** The frozen model runs once per item to cache the hidden state the adapter needs, and is then freed. Training and every measurement recompute only the adapted part of the model, which is that part's exact computation.
- [`src/adapter_bits.py`](src/adapter_bits.py) adapts the last layer's MLP at the answer position. Its output matches the full model's margin to 2e-6.
- [`src/adapter_bits_deep.py`](src/adapter_bits_deep.py) puts last-token LoRA on `gate_proj` in any of layers 24 to 27. It caches the keys and values of earlier, unchanged positions in fp16, and matches the full model to 2.4e-4.

**Measured per run:**

- **Stored information** I = M(1 − H2(err)), a Fano estimate of the bits the adapter holds about the labels.
- **I_b**, the same after rounding every adapter weight to b bits, for b from 1 to 16. The counting bound I_b ≤ W·b + 32·(number of tensors) is checked on every run.
- **The knee**: the fewest bits per weight that keep the error within one point.
- **Two predictions of the knee from per-item gradients.** The worst case uses the ℓ∞ Lipschitz constant, as in the note. The typical case treats rounding errors as independent noise on the margins.
- **The rank of the items × parameters gradient matrix.** Cover's theorem caps a linearized adapter near twice that rank.
- **The top Hessian eigenvalue** of the training loss.

Adapters:
- `probe`: a linear readout on k whitened principal directions of the final hidden state, the baseline with known capacity.
- `lora:<matrix>`: low-rank factors on last-layer MLP matrices.
- `deep-gate`: last-token gate LoRA spread over several layers.

## Results

Single seed, Adam, 2,000 to 4,000 steps, CPU. Full tables are in `results/<run>/report.md` and raw records in `results/<run>/results.jsonl`.

### What sets single-layer capacity (`results/sweep-1`, 36 runs)

| adapter | W | best stored bits | bits / parameter | knee (bits/weight) | 2 × gradient rank at 1e-3 |
|---|---|---|---|---|---|
| whitened probe, k = 64 | 64 | 128 | 2.0 | 7 | 128 |
| whitened probe, k = 512 | 512 | 557 | 1.09 | 8 | 1,020 |
| gate LoRA, rank 1 | 4,096 | 256 | 0.063 | 7 | 252 |
| gate LoRA, rank 2 | 8,192 | 381 | 0.047 | 16 | 292 |
| gate LoRA, rank 4 | 16,384 | 324 | 0.020 | 10 | 390 |
| gate LoRA, rank 8 | 32,768 | 493 | 0.015 | 10 | 646 |

- **The counting bound held on all runs at every rounding level**, as it must.
- **The linear probe stores bits in proportion to its size, as Cover's theorem says.** It reaches 2 bits per parameter.
- **Last-layer LoRA does not.** Eight times the parameters gave about twice the bits, a log-log slope of 0.26.
  - Its stored bits stay within a factor of 1.3 of twice its gradient-matrix rank at 1e-3.
  - The limit is how varied the frozen model's features are at the adapter site. The answer-position residual has an effective rank of about 12 across keys.
- **Precision goes to conditioning, not information.** LoRA needs 7 to 16 bits per weight to keep at most 0.06 bits per parameter.
- **Overloading costs information the adapter could have kept.** With 2,048 to 4,096 labels a LoRA adapter ends with 40 to 240 bits, less than it keeps with 256 to 512.
- **The knee follows first-order sensitivity.** The typical-noise model predicts the probe's knee within 2 bits on 12 of 12 runs, and exactly on 7. The worst-case ℓ∞ prediction overshoots the probe's full-fit runs by 3 to 4 bits, which is ½·log2 W, the gap the note predicted.

### Placement and spread at fixed W = 16,384 (`results/deep-2`, `results/deep-3`)

All rows use last-token gate LoRA.

| placement | stored bits at M = 512 | at M = 2,048 | at M = 8,192 | knee | top Hessian eigenvalue | gradient rank at 1e-2, init → trained |
|---|---|---|---|---|---|---|
| rank 4 in layer 27 | 220 | 106 | — | 8 | 1,297 | 22 → 16 |
| rank 2 in layers 26, 27 | 512 (all) | 782 | — | 10 | 141 | 28 → 207 |
| rank 4 in layer 24 | — | 2,048 (all) | 848 | 8 | 5 to 20 | 37 → 509 |
| rank 1 in each of layers 24–27 | 512 (all) | 2,048 (all) | **3,284** | 7 to 8 | 8 to 51 | 35 → 463 |

`deep-2` ran 2,000 steps. `deep-3` (the rows with M = 8,192, and layer 24 alone) ran up to 4,000. Neither M = 8,192 run converged before that cap, so the comparison there is at equal compute.

- **Placement matters most.** The same adapter moved from layer 27 to layer 24 goes from 106 bits to all 2,048 labels. Layer 27's features at the answer position are too uniform across keys for an adapter there to separate them.
- **Spreading adds more on top.** At M = 8,192, rank 1 in each of four layers stored 3,284 bits (0.20 per parameter), 3.9 times the 848 bits of the same parameters in layer 24 alone.
- **This is not the theory's route through depth.** Bit extraction gains capacity through more bits per weight and sharp, high-K solutions. Here every high-capacity solution needed 7 to 8 bits per weight and was 25 to 250 times flatter than the layer-27 one.
- **What grew was the gradient matrix's rank**, from 16 in layer 27 to 385 to 509 whenever layer 24 is adapted.
  - In `deep-2`, stored bits track twice that rank, as in `sweep-1`.
  - In `deep-3` the rank at 1e-3 reached the 512 items sampled, so this check cannot tell those runs apart. A larger gradient sample is needed.
- **The note conjectured that trained adapters would not use depth.** That is wrong as stated, since spreading helped. It holds only in the narrower sense that they do not spend depth on precision.

### Follow-ups in progress

[`docs/followups.md`](docs/followups.md) records predictions written and pushed before running them:
- whether plain gradient descent's stability limit (sharpness ≤ 2/η) caps what an adapter can store;
- seeds 1 and 2 of the main comparisons;
- where the four-layer adapter saturates.

### Pilots

- **`results/pilot-2`** used the template `"Item {key} belongs to group"` and a probe on raw random directions.
  - Its answer-position features have effective rank 4.7. Plain Adam could not fit even M = k with that probe, though Cover's theorem guarantees the labels are separable.
  - That led to whitening the probe and to switching to the template `"{key} ->"`. The template comparison is `src/adapter_bits_templates.py`.
- **`results/pilot-3`** is the first run on the current setup.

## Running it

Python 3.12 with the packages in `requirements.txt`. The model is `Qwen/Qwen3-0.6B` at revision `c1899de2`, loaded in fp32 on CPU, about 5 GB of RAM while caching.

```bash
pip install -r requirements.txt
python src/adapter_bits.py --pool 4096 --steps 3000 --lora-lr 0.03 --name sweep-1
python src/adapter_bits_deep.py --first 24 --pool 2048 --items 512,2048 --specs "27:4;24:1,25:1,26:1,27:1;26:2,27:2" --lr 0.03 --name deep-2
python src/adapter_bits_deep.py --first 24 --pool 8192 --items 2048,8192 --specs "24:4;24:1,25:1,26:1,27:1" --lr 0.03 --steps 4000 --name deep-3
python src/adapter_bits_report.py runs/sweep-1
```

- Runs go to `runs/<name>/results.jsonl`; set `ADAPTER_BITS_OUT` to change that. Set `ADAPTER_BITS_HF_CACHE` to use an existing Hugging Face cache, and `HF_HUB_OFFLINE=1` to stay offline.
- `--plan` takes entries like `probe:256:512,lora:gate:1:2048` (adapter, size, items).
- `--cache-file` keeps the residual cache on disk between invocations.
- `--wait-commit-gb` makes a Windows job wait for memory headroom before loading the model. This is for shared machines.
- Expect about 3 hours for `sweep-1`, 2 hours for `deep-2` and 3 hours for `deep-3` on an 8-thread laptop CPU.

## Layout

```
note/bits-depth-lipschitz.html   the theory note (open in a browser)
src/adapter_bits.py              single-layer adapters, all measurements (measure())
src/adapter_bits_deep.py         last-token LoRA over several layers, reuses measure()
src/adapter_bits_report.py       results.jsonl -> report.md
src/adapter_bits_templates.py    prompt-template comparison by feature spread
src/config.py, src/qwen_suffix.py model pin, paths, model loading, layer runner
results/<run>/                   raw records and reports of the runs described above
```

**LSV-04 learned recurrent repair:** [frozen-weight mathematical certificate, held-out CPU experiment, and delayed-state controls](research/levin_sieve_v1/NEURAL_REPAIR.md).


## LSV-15: certified LayerNorm gauge quotients

The [LSV-15 math and checkpoint protocol](research/levin_sieve_v1/pretrained_trm/GAUGE_QUOTIENT.md) formalizes the 81 per-token shared-shift directions erased by the final LayerNorm of a frozen 44K-parameter TRM-inspired Sudoku model. A complete source-locked Arb and exact-rational evaluation certifies all 64 selected top-one predictions for transverse radius 0.01, plus arbitrary shared shifts at the final pre-normalization bus. A closed-form sufficient inequality certifies every selected top-one for all radii below 0.07546051515573275. See [LSV-15 results and permanent reproducibility](research/levin_sieve_v1/pretrained_trm/LSV15_RESULTS.md). This intervention contract is distinct from the earlier 3888-dimensional initial-state ball in LSV-14.


## RRC-04: raw-coefficient symbolic program discovery

[Measured RRC-04 CPU pilot](research/rrc_compactification/RRC04_REPORT.md) tests complete product-program proposals from unprocessed partition counts, with family-disjoint finite certificates, a source-hashed exact synthesizer, matched neural controls, and honest abstention/collision measurements. The trained recurrent student failed the registered direct-program test; the exact solver supplies the meaningful upper control.


## RRC-07: iterative Euler first-fault correction

[RRC-07's executed CPU experiment](research/rrc_compactification/RRC07_REPORT.md) trains a raw-only recurrent corrector above the frozen RRC-06 checkpoint and separately accounts for exact algebraic feedback. The recurrent model moves the first wrong exponent later but recovers zero complete held-out programs; a matched feedforward network achieves more correct coordinates. The source-bound replay, proofs, models, tests and eight-budget feedback controls live in `research/rrc_compactification/`.


## RRC-08: supervised operator induction

[The RRC-08 report](research/rrc_compactification/RRC08_REPORT.md) measures a learned eight-weight exact Euler recurrence on raw coefficient series from held-out mathematical families, against a similarly sized local feedforward model, symbolic inversion, parallel depth and six indistinguishable-prefix collision pairs. Teacher-forced algebraic features and decoder costs are documented.

## SMC-01: Synchronized mixing (November 15, 2026)

[Finite exact-rational graph-switch and permutation-sweep gym](research/synchronized_mixing_v1/README.md) combines the ideas of OpenAI Mathematics #131 and #238 into a coupled-state test. A common-randomness channel leaves each marginal mixing rapidly while measurable joint dependence persists. It includes executable [gym code](research/synchronized_mixing_v1/gym.py), [tests](research/synchronized_mixing_v1/test_gym.py), and scoped CI; see the separate [MathCan OLM-01 proofs](https://github.com/patrickdugan/MathCan/tree/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock).

## SMC-02: Markov lift, relational entropy and intervention controls (November 15, 2026)

[SMC-02 report](research/synchronized_mixing_v1/TRANSPORT_LIFT_RESULTS.md) and [exact Markov lift](research/synchronized_mixing_v1/transport_lift.py) close SMC-01's snapshot/trajectory gap using a permutation random walk projecting exactly onto the lazy matching-switch chain. Its joint graph/permutation relation persists at every sweep despite both marginals mixing. [Tests](research/synchronized_mixing_v1/test_transport_lift.py) certify four- and six-vertex lumpability, the 24-state spectrum, optimal relational detection and gauge interventions. [MathCan OLM-02](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM02_COLLISION_SPECTRUM_LIFT.md) contains the pure proofs.

## SMC-03: CPU-learned relational probes (November 15, 2026)

[SMC-03 research report](research/synchronized_mixing_v1/SMC03_LEARNED_PROBES.md), [trainable raw/structured probes](research/synchronized_mixing_v1/relational_probe.py), and [three-seed results](research/synchronized_mixing_v1/smc03_cpu_3seed_results.json) extend SMC-02 with a 51-input CPU MLP trained on orbit-stratified synthetic matching/permutation pairs. The raw MLP reaches mean 95.10% held-out balanced accuracy across three fixed seeds versus a 96.67% exact relational ceiling, with gauge-relabel controls and exact matched-marginal negatives. The pure-math [OLM-03](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM03_FINITE_MEMORY_QUOTIENT.md) proves a sharp finite-code dependence frontier and a one-bit finite-horizon overshoot. A stricter whole-matching-class holdout dropped the raw MLP to 52.66% average, while a 34-parameter learned shared-edge probe with explicitly supplied gauge transport reached the 83.33% Bayes ceiling in all three seeds. [Family and equivariant results](research/synchronized_mixing_v1/SMC03_LEARNED_PROBES.md) expose the difference between discovering alignment and learning a shared edge rule. This is synthetic training, with no pretrained-model SAE evaluation claimed.

## SMC-04: Unknown frame calibration and noisy transport recovery (November 15, 2026)

[Experiment and proof boundary](research/synchronized_mixing_v1/SMC04_UNKNOWN_ALIGNMENT.md) with [inference code](research/synchronized_mixing_v1/identify_transport.py), [replay script](research/synchronized_mixing_v1/run_smc04.py), [180-trial data](research/synchronized_mixing_v1/smc04_calibration_180trial.json) and [exact regression tests](research/synchronized_mixing_v1/test_identify_transport.py). Both the vertex frame and reference matching are hidden, with 10,800 hypotheses, three withheld matching classes and up to 25% independent label corruption. A fixed three-secret, 180-trajectory replay recovered all hidden models by ten clean or 24 quarter-corrupted calibration pairs; the independent-pair negative control stayed near chance. The structural group-action family remains supplied. [MathCan OLM-04](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM04_TRANSPORT_IDENTIFICATION.md) proves the exact sample-ambiguity and noisy consensus bounds.

## SMC-05: Anonymous sparse geometry and induced frame family (November 15, 2026)

[Research report](research/synchronized_mixing_v1/SMC05_ANONYMOUS_FEATURE_GEOMETRY.md), [channel-geometry learner](research/synchronized_mixing_v1/discover_geometry.py), [heldout replay](research/synchronized_mixing_v1/run_discovery.py), [full results](research/synchronized_mixing_v1/smc05_full_results.json), and [exact tests](research/synchronized_mixing_v1/test_discover_geometry.py). The learner receives only twelve anonymous three-hot matching-state masks, reconstructs six latent vertices by a unique global six-star cover, and then infers an unknown S6 frame/reference matching with twelve calibration examples. It handles the difficult holdout (12,13,14), where one feature never activates during training, and reaches the 5/6 heldout Bayes accuracy on three frame seeds. Sensor recovery is 100/100 at (p,q,R)=(.9,.1,16), 95/100 at (.75,.25,40), and 100/100 at (.75,.25,64); a row/column-count-preserving switching null yields zero accepted geometries among 100 seeds. [MathCan OLM-05](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM05_SPARSE_GEOMETRY_RIGIDITY.md) proves and exhaustively certifies the 455/1365 deletion-boundary results. This is synthetic feature-geometry discovery within a specified K6 matching model class, not a pretrained-model SAE experiment.

## SMC-06: Trained sparse autoencoder and feature geometry (November 15, 2026)

[Experiment and controls](research/synchronized_mixing_v1/SMC06_TRAINED_SAE_GEOMETRY.md) train a **real 1,215-parameter 40→15→40 ReLU SAE** on unlabeled dense superpositions of synthetic matching-edge atoms. The [bounded decoder](research/synchronized_mixing_v1/approx_geometry.py) infers K6 feature incidence and its 720 automorphisms from anonymous approximate features, allowing at most one active-to-inactive feature correction and explicit abstention. Across [ten fixed CPU seeds](research/synchronized_mixing_v1/smc06_full_10seed_results.json), six models yield a consistent geometry, five reach the 5/6 Bayes limit on completely heldout matching classes and one reaches 13/18; four abstain. [Frozen activation evidence](research/synchronized_mixing_v1/sae_activation_evidence.json) and a [seed-103 checkpoint](research/synchronized_mixing_v1/sae_seed103_checkpoint.json) support independent exact-test replay, while [training code](research/synchronized_mixing_v1/sae_train.py) permits retraining with PyTorch. The new [heldout fidelity gate](research/synchronized_mixing_v1/fidelity_gate.py) separates training structural consistency from independent evaluation: frozen seed 101 fits after repair yet all 3 heldout top-three supports violate the geometry, while seed 103 passes 3/3; five exact tests and a CI artifact audit score penalties. Three PCA and untrained-ReLU controls and two no-anchor ablations each fail the geometry gate. [MathCan OLM-06](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM06_APPROXIMATE_SAE_RIGIDITY.md) proves the one-swap alias/13-state detection boundary. The teacher is synthetic and the complete-graph matching class is a structural prior; pretrained transformer activations remain untested.

## SMC-07: Source-locked pretrained recurrence SAE and interventions

[New research protocol](research/real_recurrence_sae_v1/README.md), [top-k SAE and causal replay](research/real_recurrence_sae_v1/smc07_real_sae.py), and [isolated GitHub Actions pilot/full jobs](.github/workflows/smc07-pretrained-sae.yml) extend SMC-06's synthetic feature tests onto an independently pretrained, checkpoint-hashed 44,001-parameter Sudoku recurrence. Puzzle-disjoint depth-one SAE fitting is compared with PCA4 and random4, with frozen depth transfer and equal-norm goal/remote-token interventions. FP64 common-channel LayerNorm shifts provide an exact structural negative control. Hosted source-locked receipts are required before empirical claims about this actual recurrent model.

## SMC-08: Source-locked recurrent causal curvature (November 15, 2026)

[Pre-registered audit](research/real_recurrence_sae_v1/SMC08_PROTOCOL.md), [source-locked Jacobian and finite-radius replay](research/real_recurrence_sae_v1/smc08_curvature.py), and [exact-real global Hessian certificate](research/real_recurrence_sae_v1/validated_causal_tube.py) deepen SMC-07's active-SAE direction sensitivity on the frozen 44K Sudoku recurrence. The new experiment computes an FP64 10×48 goal-token Jacobian on each heldout puzzle, compares active/inactive/isotropic directions against the Jacobian-optimal singular direction, measures first-order Taylor errors at fixed radii and carries a rigorous rational outer curvature ceiling from exactly represented checkpoint weights. The analytic certificate may be too conservative for finite-radius inference, which is explicitly reported. Companion pure mathematics: [MathCan OLM-08](https://github.com/patrickdugan/MathCan/blob/research/olm08-validated-tube-20261115/research/orbit_lock/OLM08_VALIDATED_CAUSAL_TUBE.md).

## SMC-09: Verified finite endpoint effects of recurrent SAE directions (November 15, 2026)

[New source-locked protocol](research/real_recurrence_sae_v1/SMC09_PROTOCOL.md) and [validated interval endpoint evaluator](research/real_recurrence_sae_v1/smc09_endpoint_arb.py) extend SMC-08's causal Jacobian audit. For the same 44K-parameter external pretrained recurrent Sudoku checkpoint and fixed puzzle split, selected active/inactive SAE, isotropic and top-Jacobian directions are interpreted as exact dyadic vectors normalized to unit length in exact real arithmetic. Arb intervals then enclose all ten output logits at the clean and radius-shifted endpoints and certify strict equal-radius output-norm ordering whenever squared-effect intervals separate. This compares the *actual fixed endpoints* and avoids SMC-08's vacuous \(10^{53}\) global Hessian bound, without certifying an entire intervention tube. [MathCan OLM-09](https://github.com/patrickdugan/MathCan/blob/research/olm09-endpoint-dominance-20261115/research/orbit_lock/OLM09_ENDPOINT_CAUSAL_DOMINANCE.md) proves the endpoint separation contract. Results are explanatory analyses of the already-used heldout puzzle IDs.

## SMC-10: Robust neighborhoods of recurrent SAE interventions (November 15, 2026)

[Registered source-locked protocol](research/real_recurrence_sae_v1/SMC10_PROTOCOL.md) and two validated implementations continue SMC-09's 432 finite-endpoint certificates to uncertain stage-one goal-token states and selected exact unit decoder directions. [Direct Arb boxes](research/real_recurrence_sae_v1/smc10_robust_endpoint_arb.py) propagate the outer set through the full recurrent model, conservatively abstaining when interval dependencies widen. [Verified local-Lipschitz transfer](research/real_recurrence_sae_v1/smc10_local_lipschitz.py) reuses source-locked Arb-verified weight spectra and per-layer LayerNorm radius transport to bound output changes around each nominal endpoint. Exact L2 uncertainty budgets are converted to sound coordinate boxes or local Euclidean endpoint balls, with the 80 other tokens and original model weights fixed. The two methods are checked against the SMC-09 zero-budget source-pinned baseline. Pure mathematics: [MathCan OLM-10](https://github.com/patrickdugan/MathCan/pull/78).
