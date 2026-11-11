# MC-01: source binding for a compiled recurrent quotient

2026-11-11. Additive bridge from RC-01 and Algernon-02 to the implemented
RMP-MC-01 experiment. These arguments are handwritten. No Lean verification or
neural-inference proof is claimed.

## Existing foundation

[RC-01](THEORY.md) owns the minimal intervention-preserving affine quotient. For
z=Bh and allowed operations h'=A_u h+b_u, its closure construction gives

\[
BA_u=D_uB,\qquad z'=D_uz+Bb_u.
\]

[Algernon-02](../algernon/circuit_mirror/README.md) already implements the exact
pointer relation and its Circom source mirror. MC-01 uses that relation as a
baseline. Its additions concern the provenance of a decoded trace, causal
interventions and the existing SEM-ECHO capture interface.

The implementation and longer proofs are in
[LoopedTransformers PR #5](https://github.com/MoralityLabAI/LoopedTransformers/pull/5),
initial commit `b1b05fba324243a6e2962451fd9901e53989a0fc`, under
`papers/rmp_looped_mechinterp/mechanistic_certificates/` and
`research_gym/analysis/mechanistic_certificate.py`.

## Conditional transfer theorem

Let a complete concrete state evolve by h_(t+1)=F_(x,u_t)(h_t), with fixed input x
and declared intervention alphabet U. Suppose a forward-invariant domain D admits
alpha and a finite machine delta satisfying

\[
\alpha(F_{x,u}(h))=\delta_{x,u}(\alpha(h)),\qquad
G(h)=g(\alpha(h))
\]

for every h in D and u in U. Induction then transports every concrete trajectory
in D to the symbolic machine, including trajectories reached by the declared
interventions. A sound circuit checking those symbolic transitions can certify
the abstract trace. To transfer its accepted witness back to a particular neural
execution, one additionally needs an independently established equality
q_t=alpha(h_t) at every visit, plus the initial input and execution identities.

**Proof.** Apply the commuting equality at the initial state. Closure keeps the
next state in D, so the same equality applies again. Induction identifies every
symbolic state; substitution into the readout equation identifies its output.
The equality to the circuit witness transfers its checked transition constraints
to this particular concrete trajectory. QED.

RC-01 supplies this premise exactly for its affine class. A trained transformer
requires separate evidence for its candidate alpha, numerical execution, and
intervention domain.

## A wrapper cannot recover missing source information

Suppose an abstract verifier sees only (x,q), and two concrete histories supply
identical (x,q) while differing on a proposed property of the actual neural run.
Every deterministic wrapper of that verifier has the same output on both
histories. For a randomized wrapper, condition on its random seed and then
average; its output distributions are also equal. Consequently no such wrapper
can distinguish that source property on both histories.

This applies conditionally to semantics-preserving verifier compilation,
recursive checking, and garbled evaluation. Each cryptographic layer still needs
its own correctness and soundness assumptions. A correct oracle trace can be
submitted for a different neural trajectory unless the source equality is bound.
A model hash without an inference relation leaves this obligation open.

MC-01 executes the counterexample: the graph (2,2,1) sends start 0 through
(0,2,1,2). The different captured labels (0,1,1,2) share its endpoint. An oracle
witness passes the abstract equations while the trusted-capture comparison fails.

## Coding changes preserve the semantic obligation

For bijections E_t on complete states, set

\[
F'_t=E_{t+1}F_tE_t^{-1},\qquad \alpha'_t=\alpha_tE_t^{-1}.
\]

Substitution cancels E_(t+1)^(-1)E_(t+1), so both branches of the commuting square
have exactly their former abstract values. Its defect is invariant under this
change of representation. Interventions and readouts must be transformed too.
A visit-dependent code can therefore require a visit-dependent decoder, even
when the same logical operation persists across loops. Charge that decoder and
its clock in RC-01's description/state ledger.

The proposed practical certificate budget records the decoder description,
symbolic transition description, intervention-lift description, live abstract
state, and executed decoding work. A decoder that reconstructs the answer from
the public program must pay for that work. The initial MC-01 importer instead
uses the captured native head at every post-loop state.

## Evidence and outstanding gates

The implementation's local suite passed 38 tests normally and under `python -O`.
It checked 712 valid trace/field cases, all 2,187 three-step candidate traces on
three-node deterministic graphs, and 108 single-wire mutations across BN254 and
M31. A shadow-register control separates natural-trace agreement from a causal
claim under intervention.

Its input is the pinned `full_trace.npz` and `metrics.json` produced by the
existing SEM-ECHO runner. The importer trusts that capture's execution; hashes
establish identity only. The Q4/Q5 checkpoint weights remain external, so this
step reports no new trained-model result. A separate CI gate compiles the abstract
Circom checker and checks compiler-produced witnesses against exported R1CS.
STWO proving, garbled verification and a cryptographic neural-source relation
remain outside this implementation.

Primary context: Geiger et al., [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997),
and the [Circom constraint specification](https://docs.circom.io/circom-language/constraint-generation/).
