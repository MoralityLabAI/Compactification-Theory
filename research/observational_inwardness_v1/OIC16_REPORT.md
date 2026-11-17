# OIC-16: Learning unknown phase dynamics from public histories

**Completed November 10, 2026.** Evidence grade: **executed synthetic CPU experiment** with independent, torch-free reconstruction of the archived predictions, exact reference checks and bounded-fault controls. The primary preregistered transfer set is \(n=7,8\). Later phase-count scaling is explicitly post-hoc.

## Experiment

Each episode uses the OIC hidden-cycle simulator, with fixed or rotating hidden phase and visible output \(x_{t+1}=x_t\oplus1\oplus u_t[z_t=0]\). The observer receives the known phase count n, the actions and the raw sensor sequence. It is not told the initial hidden phase or the true family at inference. Supervised family labels are used in training.

The **learned belief operator** contains two row-stochastic circulant transition kernels over five allowed shifts \(-2,-1,0,1,2\), a three-coefficient shared Fourier emission, and one learned raw-sensor mismatch parameter. Its fourteen trainable scalars are reused for all n. It carries \(2n\times2\) live belief-state coordinates, which must be included in resource accounting.

Training matched the OIC-15 samples, phase splits and 500-step/128-episode batch budget. Seeds 0–4, full-phase/heldout-phase splits, and clean versus independent 4% substitution plus 4% marked erasure augmentation yield **20 fitted observers** (1.28 million simulated training episodes). Training input phases were n=4,5,6. Two initial phases, z0=1 and z0=n−2, were withheld under the heldout condition. OOD n=7,8 was reserved for evaluation. Total measured training/evaluation CPU wall time across the six sequential batches was approximately **118.6 seconds**, using one PyTorch CPU thread; these are hardware-specific timings. Six evaluation schedule families were retained from OIC-15, including passive and n-active controls.

### Clean phase-count transfer

Across all five seeds in each training condition, every model correctly identifies all n initial phases and both dynamical families after n active interventions for n=4 through 8. The balanced passive-control accuracy is 50%, with posterior deviations from 0.5 below approximately 1.5×10⁻⁶. Under consistent cyclic marker relabeling, the sorted class probability distributions match exactly in the archived evaluations.

The learned fixed-family kernel assigns approximately 99.7% probability to shift zero. All twenty rotating kernels choose either +1 or −1 as their maximal shift; 13 choose −1 and seven choose +1. The orientation is formally unidentifiable under a uniform unknown-phase prior and this sensor interface (OIC-T23). The exact strength of the learned rotating mass varies: one clean/heldout seed puts only 54.7% on its selected one-step generator while still classifying the tested clean episodes correctly. Clean classification alone is therefore a weak certificate of exact hidden-state recovery.

### Raw sensor corruption

Each corruption panel contains every initial phase in both families and every single substituted or erased raw-sample position at the sharp even-phase horizon T=n+2 (n active interventions followed by two passive transitions). The source of corruption is the **raw state sensor**, so a single substitution can disturb two adjacent derived transition audits.

| Training | Phase split | Clean n=8 | n=4 single flip | n=6 single flip | n=8 single flip | n=8 single erasure |
|---|---|---:|---:|---:|---:|---:|
| Clean | full | 100% | 87.50% | 80.56% | 58.30% | 100% |
| Clean | heldout | 100% | 86.43% | 82.41% | 66.48% | 100% |
| Corruption augmented | full | 100% | 99.64% | 100% | 84.89% | 100% |
| Corruption augmented | heldout | 100% | 99.29% | 100% | 79.32% | 100% |

The table pools five seeds; each included single-fault position is tested against every initial phase and both families. The augmented models improve substantially at n=4,6, but leave failures at the unseen n=8. These panel values are exact for the archived deterministic threshold decisions, rather than confidence intervals for a population of real sensor failures.

### Post-hoc deterministic projection

After inspecting the raw fault failures, a **separate, explicitly post-hoc** projection was implemented. It takes the maximal-probability offset from each learned kernel, selects the highest-scoring phase marker, and then constructs exact deterministic cyclic codebooks for raw minimum-distance decoding. This replaces the smooth learned emission with a discrete one-hot marker and replaces the stochastic transition by a deterministic generator.

In all 20 fitted models the projected fixed generator is zero, and the rotating generator is ±1. For n=4 through 12 the selected marker is uniquely phase zero, and the projected raw-state codebooks match the exact reference families, up to reflection of hidden phase labels. In an independent enumeration, all models decode every single substituted or erased sample correctly for **even n=4,6,8,10,12** on the n-active plus two-passive schedule. By OIC-T24, the corresponding distance guarantee extends to all n and schedules for which the projection's structural hypotheses hold.

The projection inherits a **strong deterministic cyclic prior** and the exact raw sensor scoring rule. It is useful as a structural-control experiment, and is not evidence that the stochastic learned posterior has become worst-case robust.

### Post-hoc phase-count scaling: a failure at large n

The learned continuous filter was also evaluated for n=9,10,12,16,20,24,32, all outside the registered OOD n=7,8 holdout. The average clean full-active accuracy across five seeds shows an important capacity limit:

| Training | Split | n=9 | n=12 | n=16 | n=20 | n=32 |
|---|---|---:|---:|---:|---:|---:|
| Clean | full | 100% | 100% | 56.3% | 54.0% | 50.0% |
| Clean | heldout | 100% | 100% | 65.0% | 64.0% | 55.9% |
| Fault augmented | full | 100% | 95.8% | 93.8% | 53.5% | 50.3% |
| Fault augmented | heldout | 100% | 95.8% | 93.8% | 52.5% | 50.0% |

The smooth Fourier emission acts as a phase detector with a fixed angular width. As n increases, neighboring discrete phases become geometrically closer. OIC-T25 shows the adjacent-phase logit difference is O(n⁻²) for fixed Fourier coefficients. In seed 0 of the clean/full model, the probability of a pulse at the *neighboring* phase rises from 0.131 at n=8 to 0.976 at n=16 and 0.995 at n=32. The learned filter's emission becomes too broad relative to a single-phase pulse simulator; the resulting classification accuracy approaches chance. The phase-count scaling is a diagnostic chosen after seeing the n=7,8 results and requires a separately registered replication before drawing general conclusions about architecture scaling.

## Evidence and reproducibility

- `oic16_train.py`: matching OIC-15 train samples, 20 trained fits, phase/family priors, heldout splits and fault augmentation. Requires CPU PyTorch and NumPy.
- `oic16_model.py`: two learned circulant family kernels, shared Fourier emission, and raw-state belief updates.
- `results/oic16_full.json.gz.b64`: deterministic gzip/base64 receipt of **20 fitted parameter sets, all predicted posterior panels and original training metadata**.
- `results/oic16_summary.json`: SHA-256 source and receipt pins and aggregate metrics.
- `oic16_verify.py`: independent pure-Python reimplementation that reconstructs all saved predictions from the weights, including passive leakage, family accuracy, phase-relabel gauge controls and raw-sensor corruption cases. No PyTorch required for replay.
- `oic16_projection.py`: deterministic group projection, exact observable codebook reconstruction and minimum-distance raw decoding.
- `oic16_scale.py` with `results/oic16_scaling.json`: independent post-hoc phase-count stress replay.
- `test_oic16_projection.py`, `test_oic16_verify.py`: 14 unit tests, including mutation controls, exact gauge equivalence and the Fourier resolution inequality.

The full receipt verifier checks **600 clean panels, 100 marker-relabel controls and 1,080 fault-position panels**. Normal and Python-optimized test modes pass. The evidence archive verifies predictions from trained weights, while full optimizer reproduction requires rerunning `oic16_train.py` under the pinned versions and seeds. No Lean formalization or pretrained transformer test is included.

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic16*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic16*.py' -v
python research/observational_inwardness_v1/oic16_verify.py --check
python research/observational_inwardness_v1/oic16_scale.py --check
# Requires torch / NumPy, and reruns the optimizer rather than replaying receipts:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/observational_inwardness_v1/oic16_train.py --out /tmp/oic16_retrain.json
~~~

## Falsification boundary

The exact results concern finite synthetic cyclic systems under known intervention rules. The trained operator knows the two-family hypothesis structure, the allowed modular offsets and a Fourier phase sensor, and it is supervised with model-family labels. Posterior calibration at n=7,8 does not establish state recovery for arbitrary n. The projected decoder relies on stronger constraints than the learned stochastic observer. This paper contains no claim that the same mechanism exists in looped transformer activations, cosmological processes, or subjective experience.
