# OIC-16: Unknown dynamics with a learned phase-belief operator

**Prospective, unexecuted.** OIC-15 supplies the two candidate phase-transition laws in both cyclic observers. The next task is to learn those candidate dynamics from input/output histories without `z0` or privileged model-family information at inference.

## Target

Replace the exact candidate shifts with a learned, row-stochastic circulant transition kernel over a phase-belief vector. Learn both the state kernel and a shared phase-sensitive emission mapping from action/output histories. Include a generic GRU baseline and a fixed exact-Bayes upper reference with equal access to sensor readings and experiment schedules.

## Registered safeguards

Preserve the OIC-15 balanced passive-control and prior-level posterior gate; check worst-case class-identification at n full probes; train over known n=4,5,6 and hold out n=7,8, and initial phases 1 and n−2. Add cyclic relabeling controls in which the phase marker is shifted consistently. Keep sensor corruption **on raw samples** with matched erasure masks. Compare latent state posterior quality, model-family posterior, adversarial robustness and wall-clock computation separately.

For learnability, report the transition kernel's spectrum and its distance to the exact permutation generators after minimizing over label permutations. Match stored bits, effective phase slots, gradient steps and total inference operations where feasible. Treat the exact transition-family baseline as an oracle with a documented hypothesis-class advantage. If a model succeeds on passive histories above prior chance, fail the data-leak gate. Avoid selecting architecture or schedules using held-out outcomes.

## Acceptance

A reported transfer improvement requires a preregistered accuracy margin over matched compute baselines on both phase-count OOD tests, robust operation at raw-sensor fault levels registered before training, and calibrated model-family probabilities. State-reconstruction scores must explicitly account for the permanently unidentifiable fixed-family nonzero phase fibre (OIC-T18).
