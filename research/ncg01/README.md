# NCG-01: exact boundary constraints and gluing

NCG-01 supplies an exact oracle for a learning gym about shared boundary charges. Its mathematical object is a finite abelian flow model. The null-cone paper motivated the choice of constraints and gluing; the results here concern this toy model.

The model is small enough to enumerate every state. An independent implementation accumulates charge directly at each edge endpoint, while the solver builds an incidence matrix and uses modular Gauss-Jordan elimination. Both implementations agree on the fixtures in `receipt.json`.

## Run

Python 3.10 or later is sufficient. The code uses the standard library.

```bash
python3 verify_ncg01.py
python3 verify_ncg01.py --check receipt.json --output replay.json
```

The first command writes the deterministic receipt and a separate file containing elapsed time. The second reproduces every check and requires exact equality to the saved receipt, including source hashes. Timing remains outside that comparison.

| Verification | Cases |
|---|---:|
| Graph and prime-field fixtures | 808 |
| Charge sectors | 15,228 |
| Compatible charge sectors | 6,073 |
| Incompatible charge sectors | 9,155 |
| Basis states enumerated independently | 23,922 |
| Lossless encoding and decoding round trips | 23,922 |
| Ordered edge bipartitions | 200 |
| Shared-boundary contractions | 8,956 |
| Gluing associativity cases | 1,404 |
| States checked under orientation reversal and edge permutation | 3,368 |
| Invalid inputs rejected | 13 |

The graph family includes every absent/forward/reverse choice for edges on up to four vertices over F_2, the corresponding family on up to three vertices over F_3, and selected cases over F_5. Additional fixtures contain loops, parallel edges and disconnected components. Gluing uses both positive counting weights and signed integer weights.

## The exact physical space

Let an oriented graph have n vertices, m edges and c connected components, including isolated vertices. Choose a prime p. Its signed incidence matrix B has one column per edge, with -1 at the tail and +1 at the head. A loop produces a zero column.

A flow x in F_p^m obeys the requested vertex charges b when

\[
Bx=b\pmod p.
\]

The ambient Hilbert space has one orthonormal basis vector for every flow label. The physical space for b is

\[
\mathcal H_b=\operatorname{span}\{|x\rangle:Bx=b\},
\qquad \dim\mathcal H_{\mathrm{ambient}}=p^m.
\]

The constraint uses addition modulo p. A group projector expresses it as

\[
P_b|x\rangle=
\prod_{v=1}^{n}\left[\frac1p\sum_{t=0}^{p-1}
e^{2\pi i t((Bx)_v-b_v)/p}\right]|x\rangle.
\]

Every bracket equals one or zero. The following formula therefore counts the rank of P_b:

\[
\dim\mathcal H_b=
\begin{cases}
p^{m-n+c},&\displaystyle\sum_{v\in C}b_v=0\text{ for every component }C,\\
0,&\text{otherwise}.
\end{cases}
\]

### Proof

Within each connected component, the sum of the incidence rows is zero. This gives rank(B) at most n-c. A spanning forest supplies n-c independent columns: after removing one vertex row per component, its incidence matrix has determinant +1 or -1, as follows by removing leaves successively. That determinant remains nonzero over every prime field, so rank(B)=n-c.

Every possible charge vector has zero sum on each component. Those linear conditions define a subspace of dimension n-c, equal to the rank of B; hence they characterize its image. If b belongs to that image, each solution is x_0+z with z in ker(B). Rank-nullity gives m-n+c free field coordinates and p^(m-n+c) different solutions. Distinct flow labels index orthogonal basis vectors, which establishes the Hilbert-space dimension above.

### Exact compression

The solver chooses free columns during elimination. If z contains their values, decoding computes every pivot coordinate from the reduced equations. Encoding reads those free coordinates from a valid flow. This gives the isometry

\[
U_b|z\rangle=|x_0+Nz\rangle,
\qquad U_b^\dagger U_b=I_{p^{m-n+c}},
\qquad U_bU_b^\dagger=P_b.
\]

Here N is the nullspace basis specified implicitly by the reduced equations. The proof follows from the bijection that the executable checks exhaustively.

A basis label needs m field symbols in the ambient representation and m-n+c in the physical representation. An arbitrary quantum state in the physical space still has p^(m-n+c) complex amplitudes. The label count and the amplitude count describe different resources.

| Example over F_3 | Ambient dimension | Physical dimension | Field symbols per physical basis label |
|---|---:|---:|---:|
| Triangle, zero charges | 27 | 3 | 1 |
| Square with a diagonal, zero charges | 243 | 9 | 2 |
| Tree with charges (1,1,1,0) | 27 | 1 | 0 |
| Two disjoint edges with charges (1,0,2,0) | 9 | 0 | Undefined |

The last example has total charge zero over the full graph. Each component must satisfy its own charge condition, which rules out that sector.

## Gluing regions

For integer edge weights w_e, define the unnormalized boundary amplitude

\[
F_G(b)=\sum_{x:Bx=b}\prod_{e\in E}w_e(x_e).
\]

Split the edges between regions A and B. Their shared vertices form the boundary S, while charges on private vertices remain fixed. The shared charges q from A and b_S-q from B must sum to the requested charge on the combined graph. Expanding the finite sums gives

\[
F_{A\cup B}(b)=
\sum_{q\in\mathbb F_p^{S}}
F_A(b_{A\setminus S},q)
F_B(b_{B\setminus S},b_S-q).
\]

Each global assignment restricts uniquely to the two edge sets. Its contribution appears once in the term indexed by its charge on S, which proves the identity. Applying the same bijection to three regions establishes associativity. The executable checks both groupings against direct enumeration of the whole graph.

Charge-sector multiplicities enter the physical dimension. With d_A=(2,1,3) and d_B=(1,4,2) over F_3,

\[
\dim\mathcal H_{AB}^{\mathrm{phys}}
=\sum_{q=0}^{2}d_A(q)d_B(-q\bmod3)=16.
\]

The ambient product has dimension 6 times 7, or 42.

### Preserve the normalization factors

For positive boundary weights A=(1,3) and B=(2,1), the unnormalized contraction is 5. Normalizing A and B first and discarding their factors changes that value to 5/12. The conditioned boundary distribution can be recovered by normalizing the final contraction. Its weights are (2/5,3/5).

Normalizing each charge sector separately also changes relative weights. Setting each nonzero sector weight to one gives (1/2,1/2). A learning benchmark must preserve the factors appropriate to the quantity it predicts. These numbers concern a positive measure; interpreting a signed or complex amplitude as a probability requires an explicit probability rule.

## What the cutoff test establishes

Take two copies of l2(Z), with integer charge operators Q_A|n>=n|n> and Q_B|m>=m|m>. The charge-matched physical basis is

\[
\{|n,-n\rangle:n\in\mathbb Z\}.
\]

A cutoff |n|,|m|<=K leaves 2K+1 physical basis states. The increasing sequence of spaces has countably infinite dimension. This is an explicit example of charge matching coexisting with infinitely many orthogonal states.

The diagonal operator with weight 2^(-|n|) on |n,-n> has trace

\[
Z=\sum_{n\in\mathbb Z}2^{-|n|}=3.
\]

Its normalized density operator has infinite rank and finite entropy

\[
S=\log 3+\frac43\log2.
\]

The receipt verifies exact rational partial sums and tails for cutoffs through K=128. A finite Gibbs trace alone supplies a bound on a weighted sum of states. To bound the number of states by this method, one also needs a uniform lower weight on the allowed space. If T>=epsilon I and Tr(T)<infinity on that whole space, then

\[
\dim\mathcal H\leq\frac{\operatorname{Tr}T}{\epsilon}.
\]

For a positive area-like operator A and T=exp(-beta A), a spectral restriction A<=A_max supplies epsilon=exp(-beta A_max). Establishing that restriction and the trace on the intended physical Hilbert space carries the substantive burden in an application to gravity.

As an explicit toy budget, A|n,m>=(|n|+|m|)|n,m> gives

\[
\operatorname{rank}\bigl(1_{A\leq A_{\max}}P_{Q_A+Q_B=0}\bigr)
=2\lfloor A_{\max}/2\rfloor+1.
\]

This finite rank concerns the budget projector. The full charge-matched space remains countably infinite.

## Use as a learning gym

The solver can generate exact targets for any compatible charge sector. A model receives a graph, charges, and a candidate flow or free-coordinate label. Useful tasks include predicting admissibility, repairing an invalid flow while preserving supplied observables, and composing local amplitude tables.

Each score should retain the mathematical quantity it tests. Constraint residual, exact-flow accuracy, and amplitude error are separate outputs. Report achieved utility on nontrivial decisions if most sampled labels make a task vacuous. For steering experiments, choose intervention directions using development data, freeze them before evaluation, and include a matched random-direction control.

The current artifact provides the oracle and exhaustive checks. Neural training, gravitational state counting, and a physical area law belong to subsequent experiments.

## Files

- `ncg01.py`: exact solver, affine codec and gluing functions.
- `verify_ncg01.py`: independent enumeration, exhaustive checks and receipt generator.
- `receipt.json`: deterministic numerical evidence and source hashes.
- `receipt.runtime.json`: generated by the runner; elapsed runtime is excluded from exact replay.

## Source

Venkatesa Chandrasekaran, *An algebraic quantization of gravity from the null cone*, arXiv:2611.34924v1, 25 November 2026. [Paper](https://arxiv.org/abs/2611.34924), [HTML](https://arxiv.org/html/2611.34924v1). The graph construction and counterexamples above are supplied as independent mathematical exercises.
