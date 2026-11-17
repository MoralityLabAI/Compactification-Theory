# OIC-18 | Phase localization without exact cyclic projection

**Executed CPU experiment.** Date: November 10, 2026. Evidence: 60 newly trained learned-belief observers, the 20 *frozen* OIC-16 baseline fits, 1,610,400 enumerated single-raw-sensor-fault cases, deterministic archived weights and predictions, independent pure-Python replay, eight new unit tests in standard and optimized modes. This is a finite synthetic dynamical-system study. It does not test a pretrained transformer, physical cosmology or phenomenal consciousness.

## Registered objective and evidence labels

[OIC18_PROTOCOL.md](OIC18_PROTOCOL.md) preregistered n-dependent phase-localizing emissions after the OIC-16 Fourier resolution failure. [OIC18_LOCK.md](OIC18_LOCK.md) froze precise model equations, initialization, seed and evaluation choices before OIC-18 outcomes were inspected. Training used n=4,5,6; evaluation n=7,8,12,16,24,32,64, with clean n=24 transfer requiring at least a 10-point gain over the frozen OIC-16 filter. Both family labels and all n initial phases were balanced at evaluation. Phase indices 1 and n−2 were withheld in the separate heldout training condition. Model-family labels were available as supervised targets during training. The inference-time monitor received public raw-state histories, intervention histories, and known n; it received neither the model identity nor the hidden initial phase.

## Architecture comparison

Every architecture uses the same two learned five-offset stochastic circulant transition kernels over shifts -2,-1,0,+1,+2, a learned raw-sensor noise parameter, and a 2-family × n-phase × 2-visible-state posterior. The emission differs:

| Name | Emission logit | Trainable scalars | Architectural prior |
|---|---|---:|---|
| OIC-16 Fourier | \(c_0+c_1\cos\theta+c_2\cos2\theta\) | 14 | Fixed-bandwidth Fourier |
| Temperature | \(a+[1+\rho((n/5)^2-1)]\,[c_1(\cos\theta-1)+c_2(\cos2\theta-1)]\) | 15 | Learned 0–1 n² multiplier \(\rho\) |
| Von Mises | \(a-\operatorname{softplus}(k)(n/5)^2(1-\cos\theta)\) | 13 | Explicit n² concentration scale |
| Higher harmonic | \(c_0+c_1\cos\theta+c_2\cos3\theta\) | 14 | Same parameter count as original, fixed bandwidth |

All new fits train for exactly 500 Adam steps with 128 supervised episodes per step, learning rate 0.035 and the OIC-16 0.008 kernel-concentration regularizer. Original OIC-16 baseline fits had the same 500-step × 128-episode budget and same family-blind seeded examples. Thirty new fits use the clean training sensor and thirty independently inject 4% raw-bit substitutions plus 4% marked erasures. Each configuration uses five seeds and the full/heldout phase splits. **Total newly trained fits: 60**, equivalent to 3,840,000 simulated training episodes. Sum of reported per-fit CPU training time was 721.99 seconds, with four one-thread workers run concurrently, and the recorded wall time of each worker was 140.43–231.79 seconds. Exact hardware-dependent timings are captured in receipts.

All architectures retain a 4n-float mutable belief state (128n nominal FP32 bits). Convolution uses 2 family kernels × 5 shifts on n phase slots at every forward step, plus raw likelihood updates. Frozen Fourier uses 14 trainable FP32 scalars, temperature 15, von Mises 13, and higher harmonic 14, before architectural code, intermediate buffers, readout history, or decoding costs. Their training episode counts are matched, while their emitted math and operation counts differ. The GRU comparator from OIC-15 has 3,873 parameters and different computational costs; it is retained as an input-matched architectural reference rather than an equal-compute control.

## Registered clean model-family identification

Balanced accuracy after exactly n active probes, averaged across five independent seeds and all model-family/hidden-start-phase pairs. All n beyond 8 were prospectively designated OOD for this run.

### Clean training, two initial phases withheld

| Architecture | n=7 | n=8 | n=12 | n=16 | n=24 | n=32 | n=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen Fourier | 100% | 100% | 100% | 65.0% | 59.6% | 55.9% | 52.2% |
| Temperature | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| Von Mises | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| Higher harmonic | 100% | 100% | 100% | 62.5% | 60.0% | 60.0% | 58.4% |

### Clean training, all initial phases

| Architecture | n=7 | n=8 | n=12 | n=16 | n=24 | n=32 | n=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen Fourier | 100% | 100% | 100% | 56.3% | 51.3% | 50.0% | 50.0% |
| Temperature | 100% | 95.0% | 100% | 100% | 100% | 100% | 100% |
| Von Mises | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| Higher harmonic | 100% | 100% | 100% | 86.3% | 79.2% | 78.1% | 67.5% |

The **von Mises clean-training architecture** passes the declared 10-point transfer gate at n=24, while retaining n=7 and 8 accuracy across both training splits. Its result at n=64 represents 128 balanced deterministic initial phase/family cases per seed, five seeds. The temperature architecture passes large-n transfer under heldout-phase training, with one failed in-distribution n=8 seed in the full-phase condition; report that seed. The equal-capacity higher-harmonic control retains the scaling failure under heldout-phase training.

**Calibration and worst-case phase:** for clean/heldout at n=64, mean Brier scores are 0.45299 (frozen Fourier), 0.01503 (temperature), 0.08585 (von Mises), and 0.43342 (higher harmonic). The minimum per-phase accuracy across five seeds is 100% for both scale-aware emissions under clean/heldout training. All fitted observers retain 50% passive balanced classification, with maximum posterior deviation from the 0.5 family prior below 1e-4 in their passive panels. These figures were checked against all archived probabilities.

### Fault-augmented training

The perturbation-augmented models preserve almost all clean identification on unseen n, but exact worst-case single-phase accuracy can fall below one. For heldout-phase training, mean clean n=24/n=64 accuracies are 99.58%/99.22% (temperature) and 97.92%/99.22% (von Mises). The corresponding original OIC-16 fault-trained Fourier filter scores 51.67%/50.00%.

## Registered single raw-sensor fault sweep

At every even n in {4,6,8,12,16,24,32,64}, evaluate every phase of both families, every single raw state **substitution** position and every single **marked erasure** position on the schedule of n consecutive active probes followed by two passive transitions (T=n+2). The independent verifier checks all 60×8×2 fault panels, including 1,610,400 individual episodes, with exact finite reference and selected reconstructed posteriors. At this sharp horizon, the exact deterministic cyclic codebook has Hamming distance three and its minimum-distance decoder corrects any one substitution. The learned stochastic filters have no such finite guarantee.

Five-seed means, **held-out phase split**, substitution at any one raw state sample:

| Architecture | Sensor training | n=8 | n=24 | n=64 |
|---|---|---:|---:|---:|
| Temperature | Clean | 75.23% | 58.58% | 71.90% |
| Temperature | Fault augmented | 99.66% | 99.95% | 99.97% |
| Von Mises | Clean | 72.16% | 59.18% | 87.07% |
| Von Mises | Fault augmented | 100.00% | 99.77% | 99.97% |
| Higher harmonic | Clean | 69.55% | 57.99% | 58.17% |
| Higher harmonic | Fault augmented | 99.32% | 76.23% | 69.99% |

Every single erasure was classified correctly by **temperature and von Mises models** in both clean and augmented conditions for the displayed even phase counts. Other fault-panel scores, especially the higher-harmonic architecture at n=24 and n=64, appear in the full machine-readable receipt.

Because a small number of substitution cases fail under fault-augmented training, the OIC-18 result satisfies the declared **clean transfer objective** while the separate **robustness ceiling remains unattained** by these continuous stochastic models. The exact projection reference is structurally stronger and achieves 100% under the same single-fault condition.

## An additional identifiability failure: classification without phase tracking

A model can identify the fixed/rotating family from public output histories while assigning highly diffuse probability to its actual hidden rotation. This happens in the von Mises architecture. In clean/heldout training, its rotating kernel assigns only **24.7% mean probability** to the highest-probability correct ±1 step and has mean shift entropy **2.08 bits**; its mean rotating-phase MAP accuracy at n=64 (under the implementation's reflection gauge) is **2.81%**. Yet the observed family classification is 100%. In the same split, the temperature architecture places **88.4%** mean probability on its selected correct ±1 step, with mean rotating-phase MAP accuracy at n=64 of **80.0%**. These figures come from the saved model weights and full-phase/posterior diagnostics; phase MAP is a gauge-relative diagnostic, not a global state-identification certificate.

The training objective optimizes **family-label likelihood**, so a diffuse moving-phase process can produce sufficient evidence of movement without recovering the exact underlying one-step generator. This result limits any inference about hidden-state reconstruction from family classification alone. The original fixed nonzero-phase fibre remains irreducibly observationally indistinguishable under the registered interface, as in OIC-T18.

## Theoretical contribution

[OIC18_THEORY.md](OIC18_THEORY.md) proves an n-independent lower bound on the von Mises marker-to-neighbor logit gap:

\[
\frac{8\kappa_0}{25}\le\ell_n(0)-\ell_n(1)\le\frac{2\pi^2\kappa_0}{25}.
\]

It also proves a uniform Gaussian bound on off-marker emission probability mass, with limiting phase width measured in discrete phase slots. A related conditional result holds for n²-temperature Fourier emission when the learned localization coefficients are nonnegative. These bounds concern an explicitly n-dependent architectural prior. They are independent of, and insufficient for, learned transition-kernel fidelity and adversarial fault tolerance.

## Reproducibility and verification

The [manifest](results/oic18_summary.json) includes per-fit scores, per-phase worst-case performance, Brier calibration, and complete sensor-fault aggregates. Four deterministic gzip/base64 receipts preserve all 60 trained model weights and 85,440 saved clean/posterior probabilities. Four fault archives preserve every tested fault position and per-family/phase correctness count. The frozen OIC-16 baseline is replayed from its archived weights at all registered OIC-18 phase counts. All receipt hashes, source hashes, parameter and training budgets are checked by a **torch-free** verifier, which independently reconstructs selected posterior probabilities from learned weights and re-simulates selected corruption cases. The verifier passed on the executing machine. Eight math/control tests pass under ordinary Python and `python -O`.

Reproduction from repository root:

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic18*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic18*.py' -v
python research/observational_inwardness_v1/oic18_verify.py --check
# Optional training replay with torch and numpy, high CPU cost:
cd research/observational_inwardness_v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python oic18_train.py --out results/oic18_retrain.json
~~~

**Scope:** model class assumes a known cyclic phase count n, two known candidate family labels and a five-shift circulant transition architecture. Supervision uses simulated family labels and a family-blind intervention distribution. Numerical family identification, phase reconstruction, model gauge, and robust corrupted-sensor identification are separate claims. Training with exact modular dynamics, a one-hot projection, or hidden phase supervision would introduce additional architectural or informational priors.
