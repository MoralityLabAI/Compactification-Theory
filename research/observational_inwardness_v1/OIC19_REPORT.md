# OIC-19: Predictive raw-observation likelihood recovers a cyclic generator class

**Executed synthetic CPU experiment, November 10, 2026.** Status: 30 new 500-step trained fits, ten frozen OIC-18 von Mises baseline fits, six hyperparameter-validation fits of 150 steps, ten complete predictive-model raw-fault sweeps, and independent scalar verification. The numerical and theoretical claims are restricted to the declared finite hidden-cycle problem.

## Question and preregistered comparison

OIC-18 identified rotating versus fixed phases across unseen cycle sizes but generally learned diffuse rotating kernels. [OIC19_PROTOCOL.md](OIC19_PROTOCOL.md) registered a test of kernel recovery and gauge-relative phase reconstruction through stronger kernel entropy, public-history predictive likelihood, and an intentionally privileged simulator-phase ceiling. The baseline is frozen OIC-18 von Mises family BCE with the original Gini-style kernel penalty. The new models inherit the exact 13-parameter architecture and the known phase count.

The observed system has \(z_{t+1}=z_t+a\pmod n\) and \(x_{t+1}=x_t\oplus1\oplus u_t[z_t=0]\). The monitor knows \(n\), the action history, and the visible sensor readings. Initial phase \(z_0\) and the model family \(a\) are withheld at inference. Model-family labels are used during supervised training. The privileged ceiling also uses hidden-phase labels during training, separately marked.

All new fits use 500 Adam steps, batch 128, learning rate 0.035 and random seed schedule `151100+seed*19`, matched to OIC-18. Train \(n\in\{4,5,6\}\); evaluate the previously registered \(n\in\{7,8,12,16,24,32,64\}\), with seeds 0–4. The held-out initial-phase split excludes \(z_0=1,n-2\) for every training phase count. There are five fits per objective and split, hence 30 new fits plus ten pinned baseline fits.

### Entropy coefficient search

The entropy treatment uses \(\lambda\sum_a H(K_a)\). Its coefficient was chosen using only training-phase-count validation: candidates \(\{0.02,0.06,0.15\}\), seeds 0 and 1, full-phase split, 150 training steps per candidate/seed. The criterion was mean family log loss on independent full-active validation histories restricted to \(n=4,5,6\). The scores were 0.25155, 0.25415, 0.25343, respectively; the selected \(\lambda=0.02\). These six extra 150-step fits and their costs are recorded separately, outside the matched final-training budget. The validation did not inspect \(n\ge7\).

## Results: generator mass, family accuracy and phase reconstruction

The preregistered clean joint gate demands rotating-kernel mass on \(\pm1\) at least 0.90, rotating-phase MAP accuracy at least 0.90 for both n=24/64, family accuracy at least 0.95, and passive posteriors close to the balanced prior. Values below are means across five seeds.

| Objective | Split | Mean ±1 shift mass | Rotating-phase MAP n=64 | Family accuracy n=64 | Joint pass (of five) |
|---|---|---:|---:|---:|---:|
| Frozen OIC-18 family BCE | Full | 32.55% | 1.56% | 100% | 0 |
| Frozen OIC-18 family BCE | Heldout | 24.74% | 2.81% | 100% | 0 |
| Stronger entropy \(\lambda=0.02\) | Full | 0.013% | 1.56% | 100% | 0 |
| Stronger entropy \(\lambda=0.02\) | Heldout | 0.016% | 1.56% | 100% | 0 |
| Observable predictive likelihood | Full | **99.83%** | **100%** | **100%** | **5** |
| Observable predictive likelihood | Heldout | **99.75%** | **100%** | **100%** | **5** |
| Privileged phase supervision | Full | 99.89% | 100% | 100% | 5 |
| Privileged phase supervision | Heldout | 82.94% | 75.31% | 100% | 3 |

Mean n=64 Brier score with held-out phase training is 0.08585 for baseline, 0.00593 for entropy, **0.000356** for predictive likelihood, and 0.00603 for privileged phase supervision. All conditions remain at exactly balanced 50% family accuracy on passive histories, and their family posteriors remain near the 0.5 prior.

The entropy-regularized kernels select an almost deterministic shift of two phase slots, sufficient for classifying the family in this architecture but wrong relative to the fixed simulator's registered one-step transition. The supervised ceiling sometimes falls into the wrong transition solution on unseen initial phases. The predictive likelihood objective recovers the registered one-step representative in all ten runs, without the hidden simulator phase labels used by the privileged ceiling.

**Identifiability boundary:** OIC-T26 establishes that constant shifts with the same \(\gcd(n,d)\) induce the same public trace law under uniform hidden-phase prior. OIC-19's \(\pm1\) generator recovery is therefore conditioned on its five-offset hypothesis class and mixed training phase counts, and can only be interpreted up to observable phase-gauge symmetry. Perfect family identification alone does not establish unique numerical generator recovery.

## Raw-state fault audit

On even phase counts \(n\in\{8,24,64\}\), use \(n\) active probes and two passive steps. For each model, every initial phase of both families and **every raw-state sample position** are evaluated with one substituted bit or one marked erasure. The 10 predictive fits produce 200,960 faulted episodes in the frozen audit, including every registered position and phase. The original baseline uses independently archived OIC-18 single-fault receipts on the same interface.

| One substituted raw state sample | OIC-18 baseline full | Predictive full | OIC-18 baseline heldout | Predictive heldout |
|---|---:|---:|---:|---:|
| n=8 | 72.16% | 77.84% | 72.16% | 73.07% |
| n=24 | 59.18% | 99.69% | 59.18% | 99.69% |
| n=64 | 58.72% | 99.95% | 87.07% | 99.95% |

All predictive-model single-erasure tests were correct, across both splits, all five seeds, all three phase counts, all initial phases and all erase positions.

**A guaranteed fault-correction gate is still unmet.** In the predictive models, four misclassified substituted-sensor episodes remain per fit at n=24 and n=64. They occur at raw observation positions 1, 2, n−2 and n−1. At n=8, additional middle-of-probe faults lower mean accuracy substantially. The exact deterministic cyclic-codebook minimum-distance decoder is guaranteed to correct one raw substitution at this horizon (OIC-T08), whereas the learned stochastic filter is a distributional approximation.

Only the predictive treatment received the *new* exhaustive fault rerun in OIC-19. The baseline uses preserved OIC-18 fault receipts. The stronger-entropy and privileged treatment results are confined to clean, passive and phase-posterior tests in this run, with their fault robustness unmeasured here.

## Verification, storage and scope

`oic19_pack.py` combines 30 new fits and ten pinned baseline fits into a deterministic gzip/base64 archive with SHA-256 pins for all source files. `oic19_verify.py` checks every saved family classification, Brier score, passive-prior gate, kernel mass and training budget. It also checks every one of the **200,960** archived predictive-fault episodes through aggregate arithmetic, independently replays 96 clean posterior probabilities and 512 adversarial-posterior probabilities through a separate standard-library scalar filter, and checks the exact gcd-class theorem for representative phase counts. Twelve new unit tests pass in normal and `python -O` execution.

The verifier reconstructs model predictions from archived weights and checks derived metrics. It does not rerun the stochastic optimizer, and numerical training reproducibility requires the recorded CPU versions and seeded protocol. Recorded resource use includes 13 learned FP32 parameters, \(4n\) persistent belief-state FP32 coordinates, five kernel shifts for each family per time step, and the additional conditional likelihood computed during training. The parameter count omits the external code implementing cyclic indexing, action scheduling and exact benchmark generators, which are separately supplied structural information.

There is no empirical measurement of a pretrained looped transformer, physical cosmological system, or sentient internal state.

### Reproduce

~~~sh
python -m unittest test_oic19_identifiability.py test_oic19_verify.py -v
python -O -m unittest test_oic19_identifiability.py test_oic19_verify.py -v
python oic19_verify.py
# Full CPU optimizer replay (torch and numpy required):
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python oic19_train.py --variant entropy predictive phase_supervised --split full heldout --seed 0 1 2 3 4 --entropy-lambda 0.02 --out /tmp/oic19_optimizer_replay.json
~~~

Supporting records: `OIC19_THEORY.md`, `OIC19_PROTOCOL.md`, `OIC20_PROTOCOL.md`, `results/oic19_lambda_selection.json`, `results/oic19_full.json.gz.b64`, `results/oic19_summary.json`, and `results/oic19_faults_predictive.json`.
