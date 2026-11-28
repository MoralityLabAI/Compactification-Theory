# MOLT-13: exact quotient at a typed carry boundary

Development mathematical addendum to [observable quotient](OBSERVABLE_QUOTIENT.md)
and the [typed clock experiment](https://github.com/MoralityLabAI/LoopedTransformers/pull/411).
The original architecture record remains A=(S,Phi,w,rho,g,sigma,pi), with pi
the carry rule and kappa_g the visit-gradient alignment.

## Proposition (finite-word carry factorization)

Fix the physical parameters, the public future operand word, the epoch,
and a cut after primitive visit r. The typed clock's post-visit carry is
p in the 17-state simplex, obtained from the legal state softmax of the
actual normalized readout. Every future stage reads only that carry and
its own public operand. Its update and readout are deterministic.

If two cut states induce equal p, all their subsequent carries and
unrestricted vocabulary readouts agree. Proof is induction: at the first
suffix visit, both transition inputs agree; its deterministic outputs agree,
and therefore so do the inputs at the next visit. Repeat along the word.

Consequently the full future observer F factors as G composed with p. Let
E have orthonormal columns spanning the zero-sum tangent space in R^17.
In any differentiable chart of the physical intervention d,

    D F(d) = D G(p(d)) E E^T D p(d).

The equality follows from the chain rule and 1^T Dp=0, since the carry
sums identically to one. Thus rank DF <=16. This is a mathematical upper
bound independent of the numerical singular-value cutoff. Restricting the
intervention to a native BPD span Q further bounds rank by min(16,rank Q).
The current cut's unrestricted readout may distinguish states with
identical carry; appending it changes the observer and can increase rank.

This exact sufficiency result concerns a typed architecture with explicit
carry. Generic TRM or MoLT must demonstrate the relevant boundary property
separately; a low local Jacobian rank alone is insufficient.

## Representation and decision

The first 16 probability coordinates determine the last via the simplex
constraint. In real arithmetic the suffix can be replayed from these
coordinates, fixed operands, and frozen teacher. Floating-point replay
requires its own checked tolerance; a rank bound does not certify a
finite-displacement patch or preservation of its current output.

At fp64 the carry payload is 128 B. A width-64 normalized-state difference
archive costs 512 B per subsequent visit, before bindings, metadata and
shared teacher storage. The exact carry trades archive size for suffix
recomputation; all operands, teacher residency and resource costs must
be included. If response distillation requires the difference between
clean and patched trajectories, both carries or an independently reusable
clean baseline are required. The current cut response requires a separate
representation. This is not a claim that 128 B stores all teacher targets.

MOLT-12 reported full primitive and compositional competence on the addressed
fixture for both seeds, followed by a native BPD engineering fit on the
predeclared seed. MOLT-13 checks the factorization using independent
forward-mode derivatives and direct suffix replay on fresh development
worlds. The note does not claim a Lean proof, native causal specificity,
statistical confirmation or a measured resource improvement.
