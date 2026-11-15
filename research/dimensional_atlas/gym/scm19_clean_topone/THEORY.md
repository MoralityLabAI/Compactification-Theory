# SCM-19: what the clean and causal measurements establish

## Correct programs form a set

A source program is a generator label, not necessarily a unique mathematical
answer. Distinct programs in this 64-program formal-series DSL can define the
same mapping. Cross-entropy against the source identifier penalizes some correct
solutions. SCM-19 instead minimizes

\[
L(z,A)=\log\sum_j e^{z_j}-\log\sum_{j\in A}e^{z_j},
\]

where A contains verified correct alternatives. Four fixed, training-only
integer probes propose equivalence groups; every proposed label must also match
the particular training case's full demonstrations and full query future.
Finite probes are not an algebraic certificate, and failure to propose a valid
alternative can make A incomplete. False positive training labels are rejected
by exact case-specific verification. Final grading accepts any candidate that
passes the independent oracle, regardless of source-program identity.

## Public mathematical features

The coefficient encoding exposes signed magnitudes, modular residues, degree,
demo identity and topology. It does not execute any candidate program. The
polynomial-Gram arm additionally builds normalized input monomials indexed by
integer partitions through degree eight, jointly with observed output values,
then forms their outer products. These are generic polynomial relationships
computed solely from the public demonstrations. The arm is explicitly stronger
preprocessing and has more parameters; its accuracy cannot establish that a
transformer independently discovered these polynomial features from raw text.

The query need not be encoded because the predicted program is shared across
the two demonstrations and query. A clean prediction is one neural argmax.
The oracle grades afterward and never participates in candidate selection.
The trained initial-CLS readout diagnostic measures accuracy with no recurrent
updates; its computation is recorded separately from the primary prediction.

## Exact head fiber and later transport

The head is z = W2 W1 h + b, with W1 of full row rank 64 in hidden dimension
128. Let Q be an orthonormal basis for the row space of W1. Then

\[
P_\perp=I-QQ^\top,\qquad W_1P_\perp=0.
\]

The adapter writes

\[
\Delta h_v=P_\perp U\,\mathrm{diag}(g_v(a_v))\,V^\top h_v.
\]

Its global write span has dimension at most four. The gate depends on the
shared SAE code and visit. The SAE is fitted only after the base freezes, and
the dictionary freezes before adapter fitting. A base policy sets codes to
zero. The base and adapter share the same frozen reasoning weights.

At the final visit, adding a head-fiber vector to CLS cannot change exact
real-arithmetic logits. FP32 residuals are measured as a numerical check.
Earlier writes can become head-readable through later recurrent updates:
locally their effect includes W2 W1 times the Jacobians of subsequent updates.
This permits transport but does not imply that useful semantic information was
written, retained, or read. The intervention test measures that additional claim.

## A named semantic interchange

Recipient and donor share input series, topology and operators two and three.
Only the first operator is cycled I→N→S→R→I. Generation rejects identical
futures, identical public/encoded observations and previously seen signatures;
it never consults model correctness. A pair becomes eligible for an assay only
when both independent clean neural predictions produce the exact right future.

The native intervention adds the donor-minus-recipient adapter write to the
recipient CLS after a fixed visit. Because both writes lie in the same global
rank-four fiber span, their difference does too. A fixed random rank-four fiber
control is matched to that difference's norm separately for every pair. This
controls intervention size; equal nominal rank alone would not do so.

Every patch acts on CLS alone. Four coordinates at each of 27 tokens would
provide up to 108 global intervention coordinates, so that different experiment
cannot support the rank-four claim here. Full CLS at the final visit directly
copies the donor readout and is a positive control. Sham and terminal fiber
patches are null controls. Earlier full-CLS copying need not reproduce the donor
because public demonstration token states remain those of the recipient.

## Limits of causal interpretation

Fidelity is exact correct counterfactual outcomes divided by eligible pairs
actually assayed. If no pairs are eligible, fidelity is undefined. Eligibility
conditioning restricts the population to cases that the model can already solve;
it cannot be generalized to all generated cases without further evidence.

The registered endpoint requires sufficient eligible ID and OOD cases, at least
90% native-write fidelity, at least 15 percentage points over the norm-matched
random control, and clean accuracy preservation in all three seeds. Other
prespecified methods and visits are secondary diagnostics. Their success cannot
replace a failed primary endpoint.

Calibration PCA, SAE feature spans and principal angles describe finite sampled
representations. They do not certify unique semantic coordinates, global causal
sufficiency, curvature, braid structure, topology, or mathematical compactness
outside this finite DSL. The combined acquisition intervention changes data,
training, features, model size and labels; a successful acquisition result does
not identify which individual change caused the improvement.
