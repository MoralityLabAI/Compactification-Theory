# OIC-15: transition inference without privileged phase initialization

**Status: proposed, unexecuted.** OIC-14 supplies `z0` to the learner. The next pilot removes that oracle to test phase inference through an active observation interface.

## Interface

Provide the learner only raw visible states `x_0,...,x_t` and the registered actions `u_0,...,u_(t-1)`. A bounded-memory recurrent encoder produces an internal phase estimate. The ground-truth phase and target model family remain hidden during inference. Include any recurrent cache, time and action history needed to make the update deterministic. Train on independent action schedules with known visible-state targets. Separate supervised auxiliary phase labels, if tested, into a distinct ablation with their cost recorded.

## Required identifiability tests

1. Passive no-probe control: both families have identical visible histories for every hidden phase, so a model-family classifier should remain at prior-level chance under a balanced dataset. Detect hidden phase and family leakage in the training sampler or timing metadata.
2. Active identifiability: test the OIC-T02 `n`-probe worst-case guarantee and the OIC-T08 raw-state error/erasure bounds in a separate exact finite oracle.
3. Learned control: compare an equivariant phase encoder, a generic GRU, and an explicit finite Bayesian posterior update with matched input history, state bits, model description and compute.
4. Dynamic phase count: evaluate on unseen n and randomized cyclic relabellings, with permitted observation maps held fixed and logged. A phase-label change that also shifts the distinguished readout marker requires consistent re-indexing.
5. Long-horizon and adversarial sensor faults: measure event trace fidelity separately from interfamily code distance, with a bound on raw-state substitutions and marked erasures.

## Falsification and evidence grades

Treat any classifier that distinguishes the families from passive outputs as a likely data leak until independently explained. A learned monitor passes the transfer gate only after held-out phase, phase-count, and raw-sensor perturbation tests, with the uncertainty across seeds and known oracle comparisons published. Separate empirical mean error from the sharp worst-case mathematical bounds. The exact finite results establish an observable detection task; they make no claim about synthetic self-awareness or the intrinsic nature of physical systems.

## Execution closeout (2026-11-10)

The protocol has now been executed in [OIC15_REPORT.md](OIC15_REPORT.md). All 30 fitted observers, five registered training seeds, three architectures, two phase splits, exact passive controls, new phase counts n=7,8, and exhaustively placed one-fault raw sensor checks are recorded. The cyclic filters were supplied the two candidate transitions as known structural models; the raw-likelihood architecture is a separately tested sensor-model repair. Training used 500 optimizer steps and matched batches, with no privileged phase input. Mathematical guarantees and measured learned-model outcomes are distinguished in [OIC15_THEORY.md](OIC15_THEORY.md).
