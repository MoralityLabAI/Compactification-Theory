# GAL-02: compositional galacticity
Status: restricted exact oracle theorem, constructive paper proof and finite
exact campaign. Native machine timing and learned-model transfer remain open.

## 1. Additive composition no-go
For m repeated uses, suppose components independently preserve semantics,
pay construction costs C_i >= 0, and save d_i > 0 per use. If costs add and
the common horizon is unchanged, the first nonnegative net savings horizon
is ceil(sum C_i / sum d_i). The ratio is a weighted average of C_i/d_i;
its ceiling lies between the minimum and maximum component ceilings.
If the workload requires m>=1, clamp all three ceilings to at least 1.
For q-fold speedup replace C_i by q C_i and d_i by u_i-q v_i, provided
each adjusted saving is positive. If savings overlap or a stage changes
the downstream reuse count, the hypotheses fail. This explains why
simple serial accounting cannot yield a new large scale barrier.

## 2. Exact rank barrier
Let F be a nonempty family of nonzero vectors spanning V=GF(2)^r.
A hidden world consists of b independent arbitrary vectors x_t in V,
for t=1..b. A request a in F must return XOR_t <a,x_t>.
A probe is any pair (t,v) and returns <v,x_t>, where v in V may be
chosen adaptively. Arbitrary probe vectors within one slice are allowed,
even outside F; probes across slices are excluded. Each probe and each
emitted bit costs one; arithmetic and sufficient memory are free.
All instance-dependent setup probes count. There is no free data-dependent
advice. This is a deterministic online exact oracle model.

For a fixed request sequence a_1,...,a_m, the cost of every correct
algorithm on every input is at least
    b rank(a_1,...,a_m) + m.
For each t, let S_t be the span of probe vectors in slice t along the
actual transcript up through the final answer. If any a_i is not in
S_t, nondegeneracy of the finite-field dot product supplies z in S_t^perp
with <a_i,z>=1. Replace x_t by x_t+z, keeping other slices fixed.
By induction on the deterministic path, every observed probe reply
and previous output stays identical (also for an adaptive external
requester with fixed randomness and the same initial state).
The exact i-th answer changes, a contradiction. Hence each S_t
contains the request span. Its dimension is at most the number of
probes in that slice. Summing gives the bound.
The same argument at an intermediate answer gives the prefix bound.

Online construction: maintain a GF(2) echelon basis of earlier requests,
with the parity answer stored for each basis vector. Reduce the current
request, accumulating the associated parity. On a zero remainder return
the accumulated parity without probes. Otherwise probe a in each slice
(b probes), XOR the replies and insert the remainder with the corrected
parity. Induction on basis updates proves correctness. It achieves the
bound simultaneously for every request sequence; no horizon or future
request access is needed. Storage O(r^2+r) bits for dynamic basis and
answers, plus model parameters; arithmetic is not metered here.

The worst-case cost over m requests is b min(r,m)+m, since F has r
independent vectors and a request sequence can enumerate them first.
The naive slice-by-slice scan costs m(b+1), and GAL-01 yields the exact
q-fold threshold ceil(q b r/(b+1-q)) for 1<q<b+1. No finite threshold for
q>=b+1, and no q>1 hit at m<=r. This is a non-perturbative rank invariant:
it holds for every hidden input and adaptive implementation in the model.

## 3. Product versus separable composition
Let each factor observation family F_i span V_i of dimension r_i.
The elementary tensor family {a_1 tensor ... tensor a_d: a_i in F_i}
spans the tensor product, dimension R=product r_i. Proof: choose a basis
from each F_i; the corresponding elementary tensors form a basis. Every
other tensor expands in this basis by multilinearity. Use the oracle above
on b arbitrary independent hidden tensors. Locally each factor has
rank r_i; globally coupled hidden tensors require R independent directions.
For b and q fixed the threshold is linear in R, exponential in depth when
r_i>=2 uniformly. This concerns product observation spaces with unrestricted
joint latent state. It is not an assertion about arbitrary serial function
composition, nor a lower bound relative to total explicit input length.

For the separable control, use one-hot factors e_{i,j} and observation
vectors concatenated in the direct sum, with one active coordinate per factor.
The span has rank 1+sum_i(r_i-1). Proof: choose one anchor tuple a0.
The differences of tuples varying exactly one factor give r_i-1 independent
vectors per factor. They span the direct sum of even-parity subspaces within
each factor; a0 has odd parity in every factor and is outside that subspace.
Every tuple is a0 plus such differences. More generally for arbitrary F_i,
let D_i be the span of a+a0_i for a in F_i. The concatenated family span
is span(a0)+direct_sum D_i; its rank is sum dim(D_i) plus 1 precisely
when a0 is outside the direct sum. Thus input coupling, not the number of
component labels alone, governs the rank scale.

The controlled d=8, r_i=8 example has R=16,777,216 versus rank 57.
With b=8 and q=2, the crossover is 38,347,923 versus 131 queries.
Each isolated factor crosses at 19. These models have the same query
label set; the hidden latent semantics differ. The separable arm is a
structural promise, not an algorithm that solves the unrestricted joint task.

## 4. Preprocessing, warm advice and randomized relaxation
First-response guarantee: if every first query can be answered with
at most t<b additional probes after preprocessing, each possible query
must already belong to the spans of at least b-t slices on every setup
path. Count memberships of r independent vectors from F. Each slice
contains at most its rank many of them. Total setup probes >=r(b-t).
Tight: precompute r basis evaluations in each of b-t slices. This gives
the exact setup/first-response tradeoff within the same model.

If cross-slice linear probes are permitted, measure (a,...,a) directly.
The optimal cost becomes rank(requests)+m by the same transcript argument
in the joint space. The matched direct baseline then costs 2m; a strict
2x gain is unattainable at any finite m because of output cost. Free warm
advice storing r target basis parities makes all query answers available
with no further probes. These are distinct resource contracts.

Randomized extension: for fixed request sequence and independent uniform
hidden vectors, condition on the random seed and actual probe transcript
through response i. The consistent worlds form an affine subspace. If
a_i is missing from any slice span, pair worlds by the orthogonal witness;
the target bit is uniform conditional on that transcript, so success at
most 1/2. Otherwise success at most 1. Per-response error <= epsilon for
each input implies probability of full span membership at answer time
>=1-2epsilon under uniform worlds and random coins. Choose s=min(r,m)
independent requests first. On every path, each slice's final rank is at
least the count of these s requests whose vectors were in all slice spans
at their answer times. Taking expectations gives worst-case expected cost
>= b(1-2epsilon)s + m for 0<=epsilon<=1/2. Attained for the per-response
error contract by a global coin: with probability 1-2epsilon execute the
exact basis algorithm; otherwise emit unbiased random bits without probes.
For any fixed request sequence the per-response error is epsilon and
worst-case expected cost equals the bound. This does not establish a tight
bound for joint transcript success or adversaries that choose requests
conditional on private random seeds. This section is a paper derivation;
the finite exact checks certify orthogonality, not a full randomized game.

## 5. Evidence
Run node Galacticity/gal02.mjs and diff output against gal02_receipt.json.
The campaign verifies 90 subspaces through dimension four and 852
outside-span cases with orthogonal witnesses, including exhaustive input
flips; 32,768 online cache replays (two families, all pairs of 4-bit
hidden vectors, every 3-request sequence); 16 paired ranks; 44,100
additive cost controls and exact integer crossover boundaries. Total
136,786 checks. No timings or native gym executions in this campaign.
The companion MathCan module adds core-Lean additive inequalities and
finite adaptive probe-tree transcript separation; full finite-field rank,
tensor dimension and probabilistic arguments are paper proofs.

## 6. Transfer protocol
- BPD: evaluate observation rank after source/reader interventions across
  context combinations; compare joint rank with a separable control, exact
  semantics or registered error tolerance, and charge basis construction.
- DISTILL: measure continuation feature rank across policy, credit class,
  and task-family joins, with source-pinned legal transition and utility/risk
  equivalence. Policy-relative quotient validity is essential.
- Looped Transformers: recurrent states can acquire interacting context
  dimensions; use held-out joint intervention combinations and matched
  recurrent work. Do not infer oracle lower bounds from numerical rank.
- Tensor algorithms: finite tensor powers are the direct algebraic analogy.
  Rank of these linear observation features is not tensor rank of a
  multiplication tensor or its exponent.
- DataGoblin: expose public query families and resource contracts, train
  a router to recognize direct-sum/tensor structure, require independent
  rank/threshold certificates and abstain when the coupling is unknown.
  No actual DataGoblin update or native cross-gym training in GAL-02.

Next: extend the interface to priced linear arithmetic, memory and
certificate size. This is where an algebraic rank advantage may itself
become galactic.
