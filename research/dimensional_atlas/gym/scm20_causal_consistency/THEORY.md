# SCM-20: causal-consistency training and what it can establish

The experiment changes an objective on the existing 1,552-parameter rank-four
adapter. It holds the public representation, recurrent core, linear head,
SAE dictionary and head-kernel projector fixed. It therefore asks about learning
a useful intervention within that particular frozen reasoning system. It does
not compare reasoning architectures or learn a new SAE.

## Native-write interchange

Let r and d denote recipient and donor cases, with only the first generative
operator changed. At recurrent visit v, the adapter writes

\[
w_v(h)=P_\perp U\,\mathrm{diag}(g_v(a_v(h)))V^\top h.
\]

The patch adds w_v(d)-w_v(r) to recipient CLS. Because the native write was
already added to that state, the direct recipient write cancels and is replaced
by the donor write. Earlier recipient states and all other token states remain
recipient states. Both writes belong to one global subspace of dimension at
most four, so their difference also lies in that subspace.

During training, the donor write remains in the autograd graph. The causal
set-valued negative log likelihood penalizes patched-recipient predictions that
do not describe the donor's correct mathematical future. Supervision uses exact
case-specific labels, accepting verified equivalent programs. The clean loss
and distillation anchor retain both original case predictions. A fixed
warm-start clean-eligibility mask limits the causal training objective to pairs
the pretrained model can already solve; it never changes during optimization.
The model still receives clean supervision on all sampled training pairs.

This is functional interchange supervision, not a proof that one coordinate
uniquely encodes a named symbolic variable. The adapter could learn another
representation that achieves the same finite set of causal outcomes.

## Accuracy preservation and the zero-coefficient control

The registered loss is clean setNLL + 2*clean KL + lambda*causal setNLL +
0.001*write energy. Lambda is selected only from the preregistered values and
checkpoints. The lambda-zero control executes the same three forward graphs and
backward graph, but contributes zero causal gradient. Shared initial weights,
sampler and optimizer permit a training-objective comparison. Development
evaluation for selection is additional measured work and is not a matched
end-to-end FLOP budget.

Distillation is an optimization preference, not an accuracy guarantee. The
development gate rejects checkpoints that lose more than two percentage points
on either clean split, and an independent final gate repeats that requirement.
Passing development does not logically imply passing the final gate. If no
development candidate is admissible, the preregistered procedure promotes none
and retains the original warm starts; that fallback is not improved mediation.

## Fiber invisibility and recurrent transport

The frozen readout factors as W2 W1 h + b. Q spans the rows of full-row-rank W1,
so P_perp = I-QQ^T satisfies W1 P_perp = 0 in real arithmetic. A final-visit
fiber patch is exactly invisible to that head. FP32 residuals and terminal
controls are checked numerically. Earlier fiber writes may be transported into
head-readable directions by the remaining recurrent updates. A local derivative
contains the product of those remaining update Jacobians followed by W2 W1;
that product need not annihilate the initial fiber.

The permission for transport does not demonstrate its existence or semantic
usefulness. Actual native-write success, norm-matched random fiber controls,
warm/matched-model native baselines, clean preservation and terminal controls
are needed to assess the empirical claim. Secondary PCA/SAE directions can be
more effective than native writes without validating the primary hypothesis.

## Eligibility and generalization

Final pair generation uses only mathematical/generative rejection rules. The
primary assay then conditions on recipient and donor being exact-clean-correct
under all three compared models. This common intersection holds case identity
fixed across the comparison and prevents a model from receiving a more favorable
assay merely by changing its own eligible subset. It further restricts the
population; results cannot be generalized to all attempted pairs.

The larger OOD pool increases available eligible observations for the inherited
four OOD family/topology combinations. That generator couples rational/fork,
squared/cascade, lacunary/additive and twisted_product/hadamard, so a larger
sample is not a full family-by-topology cross-product test. No outcome-dependent
extension of final pools is permitted. All seeds share final public cases;
their outcomes are not independent replications of the case sample.

The registered all-seed, both-split threshold is deliberately stronger than any
positive-looking individual diagnostic. Improved fidelity below it is reported
as limited evidence. Calibration geometry, program equivalence probes and these
finite interventions establish no continuum, braid, curvature or topology
certificate, nor general natural-language mathematical reasoning.
