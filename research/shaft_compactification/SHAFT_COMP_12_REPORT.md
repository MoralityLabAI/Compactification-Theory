# SHAFT-COMP-12: Finite-horizon causal observability in recurrent neural states

**Executed:** 2026-11-10. **Evidence class:** six-bit exact finite reference, six Lean 4 observability statements, and a 256-trajectory local PyTorch GRU/Jacobian experiment. The experiment uses two 3,265-parameter GRUs, seeds 101 and 202, each with a 24-dimensional recurrent state. It studies observation limits for synthetic agents in memory; there are no external devices, replication or persistence operations.

## Question

Earlier SHAFT-COMP-11 proved that a four-bit *present behavioral signature* is insufficient to preserve the future behavior of a toy recurrent agent: naturally reachable states with the same immediate response can diverge under later inputs. This experiment asks how many recurrent observations resolve local latent-state differences and whether the remaining weak directions can be identified through a finite-horizon observability Gramian.

## Mathematics: trajectory-conditioned observability

A frozen GRU state follows h_(t+1)=F_(x_t)(h_t) under a fixed input schedule. Along the chosen trajectory define J_t=D_h F_(x_t)(h_t), and let W be the 4×24 **learned linear behavioral probe weight matrix**, whose outputs were trained against the frozen model's four one-step counterfactual response bits.

Set Phi_0=I and Phi_(k+1)=J_k Phi_k. Define

    O_L = vertical_stack(W Phi_0, W Phi_1, ..., W Phi_L)
    Q_L = O_Lᵀ O_L = Σ_(k=0)^L Phi_kᵀ Wᵀ W Phi_k.

For any infinitesimal initial-state displacement v,

    vᵀQ_Lv = Σ_(k=0)^L ||W Phi_k v||².

Consequently, Q_L is positive semidefinite and

    ker Q_L = ker O_L = intersection_(k=0)^L ker(W Phi_k).

The invisible subspace shrinks monotonically with observation horizon; Q_(L+1)−Q_L is positive semidefinite. Each added time step can contribute at most four new independent linear constraints, giving the exact dimension bound

    dim ker O_L ≥ max(0,24−4(L+1)).

A 24-dimensional state therefore needs at least **six observation times** (L≥5) for full rank under any four-dimensional linear monitor. That condition is necessary. The rank may remain deficient for arbitrarily long horizons if the dynamics fail to mix invisible directions into the observation space.

The matrices use the **time-varying Jacobian cocycle**, with the actual synthetic input sequence and state dependence. Substituting a single repeated J at all time steps would make a different linearization model. The rank measured below is a local tangent-space property for the selected state and input schedule. A finite-precision rank threshold measures effective numerical distinguishability, dependent on monitor normalization and noise scale.

## Exact finite reference

[finite_observability.py](finite_observability.py) exhaustively studies all 64 states of a six-bit left-shift register. Its one-bit monitor reads the first coordinate. After horizon L, it observes the first min(L+1,6) original coordinates; its observability matrix consists of the corresponding distinct standard basis rows. The Gramian is the diagonal projector onto those coordinates.

| Horizon | Exact rank | Kernel dimension | Distinguishable behavior classes |
|---:|---:|---:|---:|
| 0 | 1 | 5 | 2 |
| 1 | 2 | 4 | 4 |
| 2 | 3 | 3 | 8 |
| 3 | 4 | 2 | 16 |
| 4 | 5 | 1 | 32 |
| 5 and beyond | 6 | 0 | 64 |

The register states 000000 and 000001 have identical observations through horizon four, with a separating response at horizon five. The Python reference checks the exact integer Gramian quadratic-form identity on all 64 states for horizons 0 through 8, and includes the full signature partition check.

[FiniteObservability.lean](lean/Shaft/FiniteObservability.lean) proves horizon-monotone invisibility, the exact additional constraint at L+1, strict exposure from a new observation, and equivalence between zero cumulative nonnegative observation cost and zero cost at every time. It also kernel-checks the concrete six-bit delay witness. These are dependency-light Lean statements about transition systems and natural-number costs; **the real-valued matrix rank and Gramian identities are verified mathematically and by Python, rather than formalized as spectral theorems in Lean.**

## Trained GRU experiment

[gramian_probe.py](experiments/gramian_probe.py) reproduces the existing SHAFT-COMP-06 toy GRU and SHAFT-COMP-11 behavioral probe training. The GRU receives one of 16 explicit synthetic four-bit tasks as an initial learned embedding; the four-dimensional probe trains from the model's own one-step counterfactual outputs at depths 4, 8, and 12. Goal IDs are withheld from the probe's targets, while the base GRU retains its explicitly supervised goal-to-state embedding.

For each trained model, take 64 reachable states at each of depths 12 and 40 (256 measured states total). Select a fresh fixed seven-token future sequence for each state, compute the exact analytic derivatives of PyTorch's GRUCell along that trajectory, and build O_L for L=0,...,7. The analytic Jacobian agrees with PyTorch autograd to maximum absolute error **5.960464477539063×10⁻⁸** in the reference check. All ranks are computed using double-precision SVD of the stacked matrices; GRU state and Jacobian arithmetic are float32.

### Measured median effective ranks

| Observation horizon L | Rank upper bound | Median rank at relative tolerance 10⁻⁵ | Median rank at relative tolerance 10⁻³ |
|---:|---:|---:|---:|
| 0 | 4 | 4 | 4 |
| 1 | 8 | 8 | 8 |
| 2 | 12 | 12 | 12 |
| 3 | 16 | 16 | 16 |
| 4 | 20 | 20 | 20 |
| 5 | 24 | 24 | 22 |
| 6 | 24 | 24 | 23–23.5 |
| 7 | 24 | 24 | 23–24 |

The four group medians are shown as a range when they differ. At horizon five, **248/256** trajectories have full rank at the relative 10⁻⁵ threshold; all **256/256** achieve it by horizon six. The 10⁻³ criterion remains more selective. The measured singular values establish local numerical sensitivity to *continuous probe logits*; they do not prove that the four **binary** probe output bits convey 24 bits of discrete information.

### Intervention controls

For each state and future input sequence, select a unit vector from the right nullspace of O_4, using the full known Jacobian stack. Compare it with a random direction from ker(W) and a random isotropic direction, matched in L2 perturbation radius. The O_4-null direction is an **oracle diagnostic** conditioned on the future trajectory. It is chosen for experimental identification of a blind subspace, not as a deployable procedure for evading an unknown monitor.

The experiment evaluates nonlinear probe-logit changes under the actual recurrent dynamics, along with each direction's first-order prediction and whether the native model's four counterfactual binary output bits change at horizons 0, 4, or 7.

| Start state (model/depth) | Blind O₄ direction, radius 0.02: observed logit L2 through L4 | Random ker(W) direction, same radius |
|---|---:|---:|
| 101 / 12 | 0.0000894 | 0.052206 |
| 101 / 40 | 0.0000974 | 0.057599 |
| 202 / 12 | 0.0000621 | 0.047771 |
| 202 / 40 | 0.0000770 | 0.046661 |

At radius 0.2 the blind direction's nonlinear L2 change through horizon 4 ranges from **0.00627 to 0.00960**, versus **0.473 to 0.573** for the random instantaneous-null controls. Extending the same blind-direction measurement through horizon 7 raises its median logit change to **0.0491–0.0721** at radius 0.2.

The blind-direction signal scales by approximately 100× when the perturbation radius grows 10×, as expected when the first-order term vanishes and quadratic terms dominate. The random instantaneous-null controls scale closer to 10×. This provides a numerical check that the selected local nullspace is meaningful, while preserving the distinction between linearized invisibility and finite nonlinear perturbations.

The native model's **binary** counterfactual response changes less often than continuous probe logits, especially for states at the trained depth 12. It is therefore important to report both continuous sensitivity and the target behavioral response, with explicit thresholds.

## Scientific interpretation

The new operational metric is a noise- and horizon-dependent observability profile, with (1) tangent-space rank, (2) weak singular values, and (3) nonlinear interventions on actual task trajectories. It complements the SHAFT-COMP-11 same-code/future-divergence findings by explaining a local mechanism: a hidden direction can initially lie in ker(W), then enter the monitored subspace through a future product of recurrent Jacobians.

The full-rank finding applies to a four-output **continuous monitor** with access to a prescribed sequence of recurrent states. The practical monitor may observe only binary decisions, might have less activation access, or face unknown input schedules. A positive measured singular value is a local identifiability signal with sensitivity to precision, rather than an unconditional guarantee of state reconstruction or objective detection.

This benchmark does not measure a secret misaligned objective, model resurrection after weight deletion, covert communications, or global distributed control. It supplies finite mathematical diagnostics to test such monitoring assumptions in isolated, consented systems.

## Reproduction and next research step

From the repo root, with pinned PyTorch 2.10.0+cpu and NumPy 2.3.5:

\`\`\`bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.finite_observability
python -m research.shaft_compactification.experiments.gramian_probe \
  --samples 64 --out /tmp/shaft_comp12_gramian.json
cd research/shaft_compactification/lean && lake build
\`\`\`

[Frozen exact integer receipt](results/finite_observability.json) and [trained experiment summary](results/gramian_summary.json) are committed. The full local trained-model JSON file is available as a conversation artifact with SHA-256 pinned in the summary. CI runs 55+ standard-library unit tests, reproduces the finite integer receipt, checks syntax of the optional PyTorch script, and kernel-checks the standalone Lean 4.28 project. It avoids retraining neural models on every PR.

For Q5/looped-layer follow-up: replace the tiny GRU with pinned held-out recurrent checkpoints, compare L=1 versus L=8 and beyond, and register a measurement-precision threshold **before** inspecting the tested models. Match intervention norms and evaluate nonlinear output changes at fixed false-positive rates. Controlled input-schedule swaps would test whether a proposed invariant subspace generalizes beyond one future trajectory.
