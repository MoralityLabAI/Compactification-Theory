# Algebraic program semantics, sheaf obstructions, and sparse circuit repair

**Scope:** standard mathematical constructions instantiated on Algernon's pointer-machine interface. Formal Lean proofs of the composition/period claims live in `lean/Algernon/Semiconjugacy.lean`; finite-field sheaf examples and real-valued least-squares repairs are independently tested in Python. The Lean proofs are identified as kernel verified only after a successful Lean build in CI.

## A. Free-monoid actions and the reasoning quotient

Let `Σ` be the instruction alphabet, `Σ*` its word monoid, `T_a:S→S` an exact instruction, and `F_a:H→H` a neural instruction. A decoder `π:H→S` satisfies `π(F_a(h)) = T_a(π(h))` for every instruction on an invariant reachable region. Induction gives, for all words `w=a₁...aₙ`,

`π(F_(aₙ) ... F_(a₁) h) = T_(aₙ) ... T_(a₁) π(h)`.

The decoder's equality relation `h~h' ↔ πh=πh'` is a congruence on the reachable hidden system, because `π(F_a h)=T_a(πh)=T_a(πh')=π(F_a h')`. When π is surjective, S is a factor of the hidden algebra. If π is bijective, the systems are conjugate on that domain.

A cycle provides a necessary condition for exact simulation: if `F^m h=h`, then `T^m πh=πh`. If the semantic state belongs to a cycle of least period r, then r divides m. The converse generally fails because π can identify several latent states with the same semantic state.

**Practical caution:** proving composition requires the primitive correspondence on *every relevant reachable state*, including states encountered after previously unseen program prefixes. Training on steps 1–8, even with 100% accuracy, establishes a sample claim only. The deterministic example `machine.py` has a constructed projection, rather than a trained neural network.

## B. Computational graph as a twisted local system

On an m-edge directed cycle, assign `V_i=Fₚ^d` to vertices and a linear transport `A_i:V_i→V_(i+1)` to each edge. The cochain differential is

`δ: ⊕_i V_i → ⊕_i V_i`, `(δs)_i=s_(i+1)-A_i s_i`.

For this one-dimensional cellular cochain complex, `H⁰=ker δ` and `H¹=coker δ`. With `M=A_(m−1)...A_0`, solve the homogeneous edge relations in order: `s_i=A_(i−1)...A_0 s_0`. Closing the cycle requires `(I−M)s_0=0`, proving `H⁰ ≅ ker(I−M)`.

For the inhomogeneous problem `s_(i+1)=A_i s_i+b_i`, write `c=b_(m−1)+A_(m−1)b_(m−2)+...+A_(m−1)...A_1b_0`. The closing equation is `(I−M)s_0=c`. This gives an isomorphism from the obstruction quotient `coker δ` to `coker(I−M)` by eliminating the m−1 intermediate states. Consequently, `H¹ ≅ coker(I−M)`.

There are two useful distinctions:

- The ordinary constant-coefficient graph cohomology measures the cycle, while this **twisted** cohomology incorporates the transition matrices. For example, on a one-dimensional 3-cycle over F₅, edge maps (1,1,2) give `M=2`, so H⁰ and H¹ vanish despite the topological loop.
- For affine constraints, the **class of b in H¹** is the obstruction to extending local states to a globally consistent computation. With all maps equal to 1, b=(1,1,1) has nonzero obstruction 3 mod 5, whereas b=(1,1,3) closes consistently.

The analogous Circom/R1CS system uses quadratic constraints and Boolean/range checks over BN254 Fr. The linear sheaf is a separate exact model over Fₚ, or a *linearization* of nonlinear constraints when used for neural diagnosis; its Hⁱ alone does not imply an accepted R1CS witness.

## C. Gauge changes and coordinate independence

Change local bases by invertible `G_i`; the edge map becomes `A'_i=G_(i+1) A_i G_i⁻¹`. Around a cycle, with `G_m=G_0`, the monodromy transforms as `M' = G_0 M G_0⁻¹`. Thus ranks of `I−M` and `I−M'` coincide, and so do dimensions of H⁰ and H¹.

This gives an interpretable question for latent representations: does a putative computational feature survive a change of basis in the hidden representation? The toy tests apply a fixed invertible 2×2 conjugation over F₇. Neural basis changes require attention to a learned decoder and possible nonlinearities.

## D. The sparse intertwining repair theorem

Let the local learned transition be `M∈R^{h×h}`, the target `A∈R^{s×s}`, decoder `P∈R^{s×h}`. The intertwining defect is `D=PM−AP`. A restricted edit `ΔM=UB`, `U∈R^{h×r}`, changes it to `D'=D+(PU)B`.

Let `K=PU`. The projection theorem for least squares gives

`min_B ||D+KB||_F = ||(I−KK†)D||_F`,

where `K†` is the Moore–Penrose pseudoinverse. Exact repair is possible precisely when every column of D lies in `im K`, equivalently `rank[K | D]=rank K`. The least-squares choice `B=−K†D` achieves the minimum. This statement holds even if K is rank deficient; it makes *support choice* mathematically consequential.

A projected residual can vanish while a nonlinear network still fails, since the theorem concerns one linearization and allows changes to the reachable state distribution. All edits must be tested on the complete neural model, with matching norms, support counts and held-out tasks.

## E. Error amplification under repeated reasoning

For a column-stochastic state operator A, `||Av||₁≤||v||₁`. Let `qₖ` be decoded probability distributions, and define `eₖ=q_(k+1)−Aqₖ`. Unrolling the recurrence yields

`q_n−Aⁿq_0 = Σ_(k=0)^(n−1) A^(n−1−k)eₖ`,

hence `||q_n−Aⁿq_0||₁≤Σ||eₖ||₁`. Relative to a true initial one-hot state, add `||q_0−e_s||₁` to the bound. When this total is below 1, the correct target receives more than half the probability mass.

This theorem is compatible with a high apparent local score and poor generalization: small persistent errors can accumulate with depth. It also exposes a *vacuous solution*: for permutation A, a constant uniform q has zero step defect but carries no useful initial-state information. Initial-state binding and counterfactual tasks are therefore necessary.

## Next experimental measurements

LOTUS hidden-state captures will supply empirical decoders `πψ` and step maps. Register the task input and decoder range, estimate defects by depth and graph family, test whether hidden cycles preserve the semantic period spectrum, and fit local Jacobians for candidate edits. These measurements require genuine pretrained-model activations and are pending the GPU run.
