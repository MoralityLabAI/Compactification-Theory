# OIC-22: Fault-aware adaptive sensing and communication-constrained monitoring

**Prospective, unexecuted (10 November 2026).** OIC-21's clean teacher-distilled policy shows large adaptive query gains but misses its robustness gate under individual adversarial sensor errors. The next experiment optimizes a policy tree against those errors directly while preserving total control-channel bit costs.

## Registered experiment

For n in {4,5,6}, build a finite exact adversarial decision game over candidates (d,z0), query actions m_t, a bounded corruption budget e=1 and r=1 marked erasure per experiment (analyze the one-flip and one-erasure cases separately). The adversary may alter a report and thereby influence the controller's future sensor choices. The oracle value at an information set is the maximum guaranteed shift-identification success after an allowed query, minimized over legal adversarial choices and hidden initial states. For tractability, first evaluate n<=4 fully and n=5,6 through certified pruning with an explicit state-count bound.

Train five independent seeds of a fault-aware policy on n=4–6. Match OIC-21's 145-parameter sensor scorer and 500-step/96-context budget, with the same uniform shift/phase population and phase-heldout conditions. Compare against OIC-21 clean-imitation policy, entropy-gain, fixed precommitted markers, full simultaneous OIC-20 sets, and a robust nonadaptive error-correcting code reference with the **same sensor output plus address cost ledger**.

## Observation contract

The active monitor knows n and the allowed label positions, but receives only one bit per query and must transmit or preprogram each chosen address. Every comparator must report Q response bits, Q ceil(log₂ n) online address bits or an amortized preconfiguration budget, exact posterior slots and arithmetic operations, query latency and CPU wall time. Distinguish bit substitutions from marked erasures. Report whether the monitor uses exact cyclic transition hypotheses, a learned latent model, or a discrete projection and bill its stored description.

## Acceptance and falsification

A fault-aware learning gain requires a preregistered improvement of at least ten percentage points in *all-position* adversarial one-flip accuracy over OIC-21 at n=6 and 8, with no more than a 5-point loss in clean accuracy, all phases and candidate shifts enumerated for n<=8, and passive posterior equal to the uniform prior. To establish a worst-case certificate, produce a solver-checked adversarial policy tree with disjoint receipt sets across distinct d values, or explicitly report the residual ambiguity. Transfer to n=12,16,24,32 is a separate OOD evaluation using a fixed policy and sensor budget chosen before inspecting results.

Add independent uniform random label permutation and hidden phase-translation controls. Do not infer that perfect family classification constitutes full latent-state reconstruction. The exact policy remains a synthetic formal reference and makes no claim about neural consciousness.
