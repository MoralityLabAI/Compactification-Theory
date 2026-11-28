# OIC-15: Hidden-family inference without privileged phase initialization

**Completed CPU pilot, November 10, 2026.** Thirty fitted observers across three architectures, two phase splits, five seeds each. Matched training examples and optimizer-step budgets. A separate exact finite control and independent receipt verifier accompany the neural experiments.

## Research question

Does a recurrent monitor infer the hidden dynamical family from **only interventions and visible state histories**, while respecting the complete passive non-identifiability guaranteed by OIC-T01? How does structural cyclic knowledge interact with unseen phase counts and corrupted raw readings?

The model family is fixed per episode and is either a static hidden phase or a rotating hidden phase. The monitor is told the known phase-count `n`. The true starting phase `z0` is never passed to any observer. A simulated family label is available only as the **supervised training target**.

## Experiment contract

- **Finite generator:** \(x_{t+1}=x_t\oplus1\oplus(u_t\,[z_t=0])\), \(z_{t+1}=z_t+a\pmod n\), with `a∈{0,1}` and known `x0=0`.
- **Training phase counts:** `n=4,5,6`; held-out counts `7,8` have no training examples.
- **Initial-phase split:** full includes every phase; held-out excludes `z0=1` and `z0=n−2` at each training phase count. Both models are evaluated on *all* initial phases.
- **Training:** 500 Adam steps per fit, 128 examples per step, `train n` round-robin, matching family-blind random action words across architectures at each seed. Loss is mean binary cross-entropy at every observation prefix. Five seeds `0..4`, two splits, three architectures, for 30 fits and 1.92 million supervised simulator episodes across all fits. An episode's time steps are repeated supervision targets, so this is an episode count rather than an independent-target count.
- **Generic GRU:** hidden width 32, 3,873 learned parameters, six input features per transition including visible readings, masks and `1/n`.
- **Event-based cyclic filter:** one learned mismatch penalty, `2n` candidate phase/family slots, the two transition rules and event readout supplied explicitly. Audit symbols are derived from consecutive visible states.
- **Raw-likelihood cyclic filter:** same candidate hypothesis structure with one learned raw-state mismatch penalty. It consumes the state samples directly and masks erased readings.
- **Evaluation:** six schedules per `n` including full active, passive, latch, alternating and random-budget protocols, all phases, both model families; 900 fitted-condition/schedule/phase-count panels. Evaluation is independent of training trajectories.
- **Compute:** CPU PyTorch 2.10.0, NumPy 2.3.5, Python 3.13.5; deterministic single-thread setting. Roughly 30.3 seconds wall time for the 30-fit training sweep in the reported environment. CPU execution was used because no GPU was available to this runtime.

Architecture comparisons match the sampled training examples and gradient-step count. The cyclic filters have strong supplied transition knowledge and variable-size `2n` belief states; parameter counts or gradient steps alone do not establish matched total compute or a fair fully generic learning comparison.

## Clean evaluation: model-family identification after n active probes

Pooled balanced accuracy over all family/phase episodes and five seeds, averaging phase counts within each group:

| Training | Observer | Seen n=4–6 | Unseen n=7–8 |
|---|---|---:|---:|
| Full phase | GRU | 100.00% | 95.18% |
| Full phase | Event cyclic | 100.00% | 100.00% |
| Full phase | Raw likelihood | 100.00% | 100.00% |
| Two phases withheld | GRU | 98.67% | 95.80% |
| Two phases withheld | Event cyclic | 100.00% | 100.00% |
| Two phases withheld | Raw likelihood | 100.00% | 100.00% |

**Passive control:** all models attain exactly 50% accuracy on the strictly passive balanced evaluation panels; both cyclic filters return exactly `0.5` probability for each family. GRU probabilities show small nonzero deviations from 0.5, giving passive Brier score approximately 0.252 in held-out runs, versus the exact Bayes optimum 0.250. These are calibration errors, not above-chance passive classification.

On random-budget schedules, all models score below 100% because some histories are mathematically ambiguous; the exact posterior and OIC-T19 distinguish that uncertainty from model error.

## Adversarial raw-state sensor checks

For each even `n` in `{4,6,8}`, evaluate one substitution of **every possible raw-state sample**, using `n` active probes followed by two passive steps (`T=n+2`). Also enumerate each possible single marked erasure with `n` active probes and one passive step (`T=n+1`). Tests cover all initial phases of both families and five learned seeds.

**One raw-sample substitution, initial phases withheld during training:**

| n | GRU | Event filter | Raw filter | Exact codebook decoder |
|---|---:|---:|---:|---:|
| 4 | 73.6% | 53.6% | **100%** | **100%** |
| 6 | 76.1% | 65.7% | **100%** | **100%** |
| 8 | 74.4% | 63.1% | **100%** | **100%** |

The raw likelihood observer also attained **100%** over every tested marked-erasure position for `n=4,6,8` across all five seeds. In each case, the minimum across initial phases and fault locations was correct. The exact codebook decoder's guarantee follows independently from OIC-T08.

The event filter's mismatch channel assumes effectively independent transition-audit errors. A single raw sample fault distorts two adjacent audit measurements, which helps explain its failure. The raw filter evaluates raw samples directly, and its fitted penalties satisfy the sufficient condition derived in OIC-T21: at the sharp fault-correcting horizon, \(\beta>\log n\).

The raw-filter model's success is tied to **provided exact candidate dynamics**, a binary raw-sample fault model, and the declared `n` values. The generic GRU gets less transition knowledge, and the aggregate CPU cost of these architectures has not been matched.

## Mathematical new results

[OIC15_THEORY.md](OIC15_THEORY.md) develops the following narrow, fully specified statements:

- **T18:** Identify the hidden transition family after `n` active probes while retaining \(\log_2(n-1)\) bits of irreducible phase uncertainty in the nonzero fixed-family fibre.
- **T19:** The exact Bayes optimum for the unknown-phase task; in particular a passive episode yields no model-family information.
- **T20:** A finite audit mismatch penalty has an eventual phase-count likelihood-collapse threshold. Learned event-filter penalties predict first collapse between `n=578` and `n=611`; this threshold is analytic, not an executed 600-phase network test.
- **T21:** A raw-likelihood filter is guaranteed correct under at most `e` raw substitutions and `r` marked erasures when \(\beta(d_{min}-2e-r)>\log n\). The sharp even-phase sensor schedule reduces this to \(\beta>\log n\).
- **T22:** Sensor-level faults induce correlated audit errors via \(\Delta b_t=f_t\oplus f_{t+1}\).

These results extend classical finite-state identification, likelihood weighting, and error-correcting codes to the declared OIC toy. No novelty priority is asserted for the general methods.

## Evidence, integrity, and reproducibility

The experiment directory includes an executable Python reference, all 30 parameter snapshots in a deterministic compressed archive, a separately archived compact clean-evaluation receipt, the 900-panel aggregate analysis, and an SHA-256 manifest with source-code hashes.

- `oic15_pilot.py` contains simulator, architectures, training and clean evaluation.
- `oic15_analyze.py` reconstitutes saved weights and **independently reruns model inference**, checks every clean-evaluation probability array, and enumerates all one-fault positions.
- `oic15_math.py` supplies finite proofs' exact arithmetic and codebook references.
- `oic15_verify.py` and `test_oic15_verify.py` use **only the standard library** for artifact integrity, independent oracle, output metric and tampering checks. They cannot prove that the recorded weights were produced by the stated optimizer; the optional PyTorch replay is the corresponding implementation check.
- `oic15_package.py` produces both gzip/base64 archives with deterministic gzip timestamp and SHA-256 manifest.

Run from the repository root:

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic15*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic15*.py' -v
python research/observational_inwardness_v1/oic15_verify.py
# Optional torch/NumPy replay, no retraining:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/observational_inwardness_v1/oic15_analyze.py \
  --input research/observational_inwardness_v1/results/oic15_full.json \
  --out /tmp/oic15_analysis_check.json
# Optional CPU retrain:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/observational_inwardness_v1/oic15_pilot.py \
  --out /tmp/oic15_retrain.json --steps 500
~~~

The optional PyTorch analysis requires `oic15_full.json`, packaged separately for use outside lightweight CI. The repository stores the compact per-run predictions and the fitted weights in compressed receipts. The standard-library verifier works entirely from those receipts.

## Interpretation and next experiment

The experiment provides an operational example of *observationally inaccessible hidden phase* coupled with a model-family property that can be determined after intervention. Generic recurrent learning partly recovers the same distinction without phase inputs. The stronger cyclic observers use explicit dynamical hypotheses and recover the finite oracle's robustness after the raw sensor model is specified correctly.

The next falsifiable step is to infer **the candidate transition laws themselves** rather than supplying both alternatives to the observer. That would test whether an amortized learned recurrent belief state discovers the structure, while the current experiment demonstrates the value and limits of knowing it in advance.

These are finite synthetic models, with exact group-transition equations and controlled faults. They provide no empirical evidence for physical cosmological inwardness, subjective awareness, or hidden recurrence in a deployed frontier model.
