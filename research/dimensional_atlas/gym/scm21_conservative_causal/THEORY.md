# SCM-21: what stronger anchoring and bounded steps can establish

The study changes optimization of the existing 1552-parameter rank-four adapter.
The coefficient representation, tied recurrent core, linear head, SAE and
head-kernel projector remain fixed at their SCM-19 values. The causal objective
supervises a donor native-write interchange after the second visit; it does not
introduce an extra inference-time verifier, program search or architecture.

## Distillation is an anchor, not a guarantee

The clean objective is set-valued program negative log likelihood plus ten
times KL(warm prediction || current clean prediction). Causal setNLL targets
the donor's case-verified correct programs, only on training pairs the fixed
warm start solves on both sides. A mean native-write energy penalty has weight
.001. The causal coefficient is .25 or .5, compared with SCM-20's selected 1.
The zero-coefficient control executes the same three forward/backward graphs.

A stronger KL coefficient makes changes to the warm softmax distribution more
expensive, but cannot guarantee unchanged argmax or exact mathematical accuracy.
Small changes near a decision boundary can flip the top program. The one-point
development buffer and independent two-point final guard therefore remain
empirical tests, not consequences of the loss.

## Actual parameter increments

AdamW rescales gradients through its moments, so reducing a loss coefficient
alone does not imply a proportionally smaller parameter step. SCM-21 also lowers
the learning rate and clips each proposed parameter increment to an L2 ball
of radius .01 around the pre-step adapter vector. A small FP32 safety margin is
used when capped. The optimizer moments retain their unscaled proposal history;
this is projected-step optimization rather than ordinary AdamW.

The bound limits each increment, not total path length, displacement after many
steps, native-write norm, output sensitivity or clean accuracy. Every actual
increment and cap decision is logged. Base weights, dictionary and projector
are protected by exact state hashes throughout.

## Native-write interchange and fiber transport

The adapter writes P_perp U diag(g_v(a_v)) V^T h, where P_perp projects into the
kernel of the frozen first readout matrix W1. Replacing recipient CLS write
with donor write adds their difference, which lies in a global span of dimension
at most four. Donor gradients remain live during causal training. Later updates
may transport an earlier fiber write into head-readable directions.

At the final visit, W1 P_perp=0 in real arithmetic, so fiber writes are invisible
to the linear readout. The experiment checks FP32 residuals and terminal null
controls. Full CLS at the last visit reproduces the donor readout and serves as
a positive control. Norm-matched random rank-four fiber patches control patch
size separately for each pair/model/visit. Four coordinates act globally on
CLS; no four-per-token intervention budget is hidden in that claim.

## Freshness, factorial coverage and conditional conclusions

Public signatures contain query coefficients, while the model encodes only
demonstrations and topology. An additional exact float32 encoder-input registry
rejects observations already present in known SCM-19/20 roles, all current
pre-final roles, or earlier fresh cases. Query changes alone cannot make a
repeated model input enter a new role. Signature/encoding identity rejection
does not remove all algebraically equivalent or structurally related tasks.

Accepted cases are balanced by source family and topology, then shuffled with a
fixed role seed. OOD now covers all 16 family/topology combinations. This fixes
a previous coverage limitation but changes the data distribution; raw score
changes versus earlier studies do not isolate the optimization changes.
Warm, matched-control and causal models share every new confirmation case.

The primary assay conditions on both recipient and donor being exactly solved
under all three models and uses identical first-128 common pair IDs. This
controls eligibility differences but restricts the population. The first
operator changes while inputs/topology/other operators stay fixed; improved
interchange fidelity, if observed, establishes the tested finite-DSL function
on that subset, not a unique semantic code or broad mathematical reasoning.

The registered robust positive still requires >=90% fidelity, >=15-point
advantages over random/warm/matched controls, sufficient common eligibility
and fresh clean preservation in every seed and stratum. Smaller gains and null
results are reported directly. Calibration geometry and finite equivalence
probes provide no continuum, topology or algebraic-equivalence certificate.
