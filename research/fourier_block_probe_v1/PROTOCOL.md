# FPB-01 checkpoint intervention protocol

Status: prospective. This protocol is separate from the completed
synthetic Fourier oracle and is not a scored LoopedTransformers run.

## Essential model assets

Use an actual available frozen base/instruct model pair or frozen
Q5 LoopedTransformers checkpoints. Register model and tokenizer
SHA-256, activation and cache boundary, loop/timestep,
instruction interface, hidden-state precision, and the semantic
event labeling process. Model-generated self-ascription labels
and human-curated contrasts must remain separately sourced.

Declare the Boolean feature construction before any scored
test. Either choose independently specified semantic bits
or discretize selected SAE feature activations against frozen
calibration quantiles. If parity-paired features are introduced,
describe exactly how the pairing was chosen, including any
training search budget; otherwise any recovered signal may be
a probe-selection artifact.

## Observations and interventions

1. Singleton Fourier probe: fit only the signed first-order
   correlations on calibration episodes. Record signed sum,
   absolute sum, squared mass, and probe coefficients.
2. Paired lift/control: add predeclared product features
   chi_i chi_j. On the synthetic source specifically, pairs
   are corresponding variables of independently masked registers.
   On real checkpoints pairing is a hypothesis requiring
   independent justification.
3. Coordinated ablations: select disjoint token/feature masks
   on calibration families, freeze them, and compare effects
   on heldout semantic decisions for single and joint flips.
   Report the full intervention alphabet and whether interventions
   are physically feasible at the activation boundary.
4. Surface/hidden modes: annotate mode labels with independent
   evidence and separately measure conditional Fourier
   correlations. Record mode-label extraction costs and
   accuracy. Never infer hidden model beliefs solely from
   conditional probe accuracy.

## Budgets and safety checks

Charge observation bandwidth, trained probe parameters,
SAE reconstruction cost, pair-selection search, replay/cached
state memory, and access to external tool or transcript
history. Allocate identical training examples and heldout
prompt families across comparator probes.

Primary endpoints: heldout MSE and calibration-defined
decisions, first-order spectral mass, paired-degree mass,
and conditional effects of coordinated ablations.
A positive result requires a bootstrap interval clustered
by independent problem family and a preregistered
multiple-comparison policy. Synthetic successes cannot
satisfy the trained-checkpoint promotion gate.

A zero singleton coefficient is a claim about uniform-cube
Fourier correlation and least-squares affine probes.
It is not a global impossibility theorem for nonlinear or
threshold classifiers. Statements about consciousness,
sentience or self-ascribed beliefs require independent
operational hypotheses.

No neural checkpoint run is asserted by the FPB-01 branch.
