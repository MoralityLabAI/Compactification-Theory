# Quantized multivariate jets: proved components and a conditional network theorem

Audit date: 2026-11-15. Status: the component results below have complete paper
arguments; six integer recurrence lemmas also pass Lean 4.35.0 and leanchecker.
The full bounded-weight uniform approximation theorem remains conditional on
the spatial circuit interface in Section 7. No finite test closes that obligation.

## 1. Class, parameters, and conventions

Fix d >= 1, s > 0, k = ceil(s)-1, alpha = s-k in (0,1]. Use the classical
C^{k,alpha} ball: every partial derivative through order k has sup norm <= 1,
and every order-k partial derivative has alpha-Hölder seminorm <= 1 for the
infinity norm on the cube. At integer s this means C^{s-1,1}. The usual C^s ball
with bounded order-s partials embeds in this class after a constant depending
on d. This convention does not mean the integer Zygmund class.

The exact multi-index set is I = { beta in N^d : |beta| <= k }. Thus

    J = |I| = binom(d+k,d) = binom(d+ceil(s)-1,d).

Let h=2^{-q}, q>=1, and tau=2^{-ceil(sq)}. Then h^s/2 < tau <= h^s.
Use N coarse blocks and B=K^d fine sites per block, with dyadic grid sizes and
M=NB comparable to h^{-d}. Logarithms in resource estimates are base two;
write ell=1+ceil(log2 M).

P counts every scalar affine coefficient and bias in the chosen architecture,
including fixed arithmetic constants. Excluding fixed gates would invalidate
the comparison with parameter/VC bounds. For a dense variable-width network
with widths n_0,...,n_L, count sum_l (n_{l-1}+1)n_l, including its zero slots.
For sparse circuits, count all designated edges and biases and specify that
sparsity pattern independently of f. Neither count is automatically w^2 L.

The quantization alphabet is [-R,R] intersect 2^{-b_frac} Z. Its actual storage
cost is b_frac+O(log(1+R)) bits. Below R depends on d,s only, so the distinction
from total bits b is additive O_{d,s}(1). Exact ReLU realization and evaluation
with rounded intermediate activations are different questions.

## 2. An integer Taylor transport removes matrix rounding

For a grid point y_i define factorial-normalized coefficients

    a_{i,beta} = h^{|beta|} D^beta f(y_i) / beta!,
    u_{i,beta} = a_{i,beta}/tau.

If y_{i+1}=y_i+h v_i, with v_i one of the signed coordinate vectors, then

    u_{i+1}=T(v_i)u_i+r_i,
    T(v)_{beta,gamma} = binom(gamma,beta) v^{gamma-beta}   (gamma>=beta),

and the other entries are zero. Taylor expansion of D^beta f along the one
coordinate gives ||r_i||_infinity <= 2. Indeed, after multiplication by
h^{|beta|}/beta!, the remainder is at most h^s/(beta!(k-|beta|)!), with the
factorial interpreted as 1 when its argument is zero. Dividing by tau costs
less than 2. The integral Taylor remainder proves this bound also at alpha=1.

Every T(v_i) is an INTEGER matrix. Its maximum row sum is at most

    A = 2^{k+1},

by sum_{l=0}^{k-|beta|} binom(beta_j+l,beta_j)
=binom(beta_j+k-|beta|+1,beta_j+1) <= 2^{k+1}.
It changes with the snake direction. A single fixed T cannot describe all
turns of a multidimensional snake. No approximation of T is needed with this
normalization. The original derivative normalization obscured this useful fact.

Choose tilde u_0 componentwise nearest to u_0 and, recursively, let

    e_i=tilde u_i-u_i,
    c_i=round(r_i-T(v_i)e_i),
    tilde u_{i+1}=T(v_i)tilde u_i+c_i.

Then e_{i+1}=round(z_i)-z_i, z_i=r_i-T(v_i)e_i. Therefore, for EVERY path
length, ||e_i||_infinity <= 1/2 and

    ||c_i||_infinity <= 2+A/2+1/2 <= C,    C=3+2^k.

The states tilde u_i are integers. The physically decoded state
v_i=tau tilde u_i has ||v_i||_infinity <= 3/2 and obeys

    v_{i+1}=T(direction_i)v_i+tau c_i.

Here the argument of T is the signed spatial direction. If an extension of f has
norm H instead of 1, replace the bounds by ||r||<=2H, ||v||<=H+1/2, and
C=ceil(2H+2^k+1/2).

For ||x-y_i||_infinity<=h, the polynomial

    tilde p_i(x)=sum_{beta in I} v_{i,beta} ((x-y_i)/h)^beta

satisfies

    |f(x)-tilde p_i(x)| <= (C_T+J/2)h^s,
    C_T=max(1,d^k/k!).

To prove the Taylor term, apply one-variable Taylor expansion to
t -> f(y_i+t(x-y_i)). The order-k directional derivative has Hölder
constant at most d^k h^s, and its integral remainder is at most d^k h^s/k!.
The quantization term is a sum of J terms each at most tau/2.
For k=0 use the Hölder inequality directly. All segments must lie in the
domain where the stated derivative bounds hold.

The wrong formula -round(Te+r) fails even with T=1,u=tilde u=0,r=1:
the next error is -2; the corrected formula gives zero.

## 3. Packed digits, margins, and precision

Encode each correction coordinate c in [-C,C] as a=ceil(log2(2C+1)) bits
of c+C. A block's correction stream has

    K_code = J a (B-1)

bits. Initial jet coefficients are separate dyadic weights with ceil(sq)
fractional bits; their number is J per block, not B J per block.

For bits beta_1,...,beta_m, use a trailing sentinel:

    z = sum_{j=1}^m (1+beta_j)4^{-j} + 4^{-(m+1)}.

This uses exactly 2(m+1) fractional binary bits. The reader

    D(y)=4[ReLU(4(y-7/16))-ReLU(4(y-1/2))],
    y_next=4y-1-D(y)

has maximum weight magnitude 4 and dyadic constants. Before each of the m
reads, a zero-leading state is in [1/4,5/12] and a one-leading state is in
[9/16,2/3]. Consequently D returns the exact bit and leaves the next encoded
suffix. The sentinel is retained, not read.

Both margins are at least 1/48. If only the initial code has error delta,
then all m bits remain exact when 4^{m-1}|delta| <= 1/96. Induction gives
suffix error 4^j delta while the digits agree. If each completed update adds
error at most eta, the bound after j updates is

    4^j |delta| + (4^j-1)eta/3.

The decoder's internal affine/ReLU evaluation errors must also stay within
the margin; a per-operation bound is converted to the above eta by a fixed
constant because one digit reader has constant size. Thus O(m) precision is
needed for ACTIVATIONS in an approximate evaluator, as well as stored codes.
This is not stable evaluation in fixed IEEE precision as m tends to infinity.

Without a sentinel the final one-leading state can be exactly 1/2. A negative
perturbation of size 2^{-20} changes D(1/2) from 1 to 65535/65536. Therefore a
uniform two-sided margin claim for that finite encoding is false. Exact digit
correctness without a sentinel remains valid. This counterexample does not
contradict ideal-arithmetic approximation by exactly quantized networks.

## 4. A fully counted bounded-weight lookup lemma

There is a simple alternative to relying on the unavailable dyadic-tree
source. For n=2^r>=2 and values a_0,...,a_{n-1} in [0,1], let F interpolate
the values at j/n and be constant to the left/right. Define

    c_0=a_1-a_0,
    c_j=a_{j+1}-2a_j+a_{j-1}  (1<=j<=n-2),
    c_{n-1}=-(a_{n-1}-a_{n-2}).
    F(x)=a_0+n sum_{j=0}^{n-1} c_j ReLU(x-j/n).

Differentiating on each interval shows slope n(a_{j+1}-a_j); continuity
and the left value determine F. The right slope is zero because the c_j
sum to zero. Thus this identity holds for every real x, not only grid nodes.

Implement the hinge layer of width n, then the signed pair ReLU(G),ReLU(-G)
for G=sum c_j hinge_j. Double both channels r times and take their difference
plus a_0 as the affine output. The widths are

    1 -> n -> 2 -> [2 repeated r times] -> 1.

Including ALL dense scalar slots, this is exactly

    P_lookup=4n+6r+5,  L_lookup=r+3,  w_lookup=n,  R_lookup<=2.

If a_j have p fractional bits, all coefficients lie on the grid with
max(p,r) fractional bits. Differences of dyadic values require no additional
fractional bits. Larger bounded signed outputs are handled by a fixed
rescaling. Carrying a fixed number of other channels changes only constants.

This proves efficient TABLE lookup at a correctly supplied discrete address.
It does not itself compute the multidimensional safe cell address, nor prove
the old dyadic-tree implementation. A continuously varying address interpolates
codes and can produce an invalid digit stream. That distinction is essential.

## 5. Boundary stitching and polynomial arithmetic

Let Q=2d+1, choose dyadic D_0=2^{ceil(log2 Q)}, and rho=1/(4D_0).
Translate the fine grid in every coordinate by t h/D_0, t=0,...,Q-1.
For a fixed coordinate, the Q grid boundary phases have pairwise circular
separation at least h/D_0. Hence at most one grid has boundary distance
<=rho h in that coordinate. The union over d coordinates excludes at most d
grids, so at least d+1 are safe at every point, including a point on an
unshifted grid boundary. This proof is exact for all real points.

If each safe grid's output is within E of f(x), their median is also within E:
at most d of 2d+1 outputs can lie strictly below f(x)-E or above f(x)+E.
No magnitude assumption on the bad outputs is needed. A fixed sorting circuit
uses O(Q^2) comparators, with max(a,b)=b+ReLU(a-b) and min=a-ReLU(a-b).
Signed channels cost a fixed factor. Thus stitching has O_d(1) parameter and
depth overhead and ZERO additional boundary error. Actual cube-endpoint cells
still need the domain treatment required by Section 7.

On [0,1], set g(x)=2ReLU(x)-4ReLU(x-1/2)+2ReLU(x-1). The function

    S_m(x)=x-sum_{j=1}^m 4^{-j} g^{compose j}(x)

is the linear interpolant of x^2 on the dyadic mesh 2^{-m}. This follows
inductively because each tent subtracts the midpoint interpolation excess.
On each interval, interpolation error is (x-a)(b-x), so

    0 <= S_m(x)-x^2 <= 4^{-(m+1)}.

Then 2S_m((x+y)/2)-(S_m(x)+S_m(y))/2 approximates xy with error at most
3*4^{-(m+1)}. Signed bounded inputs require fixed affine rescaling. Each
iteration needs constant width, O(1) coefficients of magnitude <=4, and the
coefficient 4^{-j}. Total size/depth is O(m), precision 2m+O(1).
Maintain the running sum and current tent in signed channels; do not retain
all m tent outputs in a wide layer.

Products for degree-k monomials need O(J(k+1)) multiplications. Clip intermediate
monomial values to [0,1] (or use a signed bounded interval for signed local
coordinates). The clipping is nonexpansive and preserves the exact monomials.
A primitive error eta <= tau/[2J(2k+1)] makes the resulting polynomial error
<=tau/2, with a constant-factor refinement for signed/rescaled products. Thus
m=O_{d,s}(ell) is sufficient. These constants grow with J and k; none is
asserted uniform in d or s.

## 6. Finite-precision jet propagation is polynomial, conditional on correct digits

Translations satisfy T(a)T(b)=T(a+b). This follows either from the binomial
identity or from translating a polynomial twice. Thus if arithmetic injects
physical-jet errors eta_i of infinity norm <=eta, then

    delta_i=T((y_i-y_0)/h)delta_0
            +sum_{j<i} T((y_i-y_{j+1})/h)eta_j.

In a block of K sites per side, all coordinate displacements are <=K. Every
matrix entry is bounded by 2^k max(1,K)^k, so

    ||delta_i|| <= J*2^k*(1+K)^k (||delta_0||+B eta).

This is polynomial in the block side/length, not the crude A^B bound. The
argument requires exact transport matrices and correct transmitted digits.
Code-read errors have the exponential amplification from Section 3 and must
be controlled separately. Choosing activation precision
O_{d,s}(B+ell) is sufficient for these components, up to constants for the
number of arithmetic operations and selector margins. No robustness theorem
for an unspecified hardware evaluator is being asserted.

## 7. Precise remaining interface obligation and conditional theorem

The missing result is a QUANTIZED SPATIAL INTERFACE, with a layered circuit
description and a complete count. It must do all of the following for each
of the Q shifted grids:

1. On its safe set, supply the exact coarse-table address, the fine snake
   index, and local coordinates xi=(x-y_i)/h with |xi|<=1.
2. Select only that block's packed code and J initial jets, carry the input
   and local coordinates through the entire decoder, and capture/evaluate
   the chosen fine jet with no multiplication by an unbounded index.
3. Handle partial blocks at the boundary of [0,1]^d. Either give a proved
   bounded-norm smooth extension and its explicit grid/circuit constants,
   or treat endpoint blocks directly without changing integer transport.
4. Use O_{d,s}(N+B+ell) scalar slots, O_{d,s}(ell+log N) layers,
   width <= C_{d,s}(N+B^{1/d}+1), weights/biases bounded by C_{d,s}, and
   O_{d,s}(ell) fixed-coefficient fractional bits. Its address must be
   EXACT on safe sets; a small nonzero address error is insufficient when
   interpolating packed codes. State its safe-set margins.
5. Specify a sparse edge list or variable-width dense layout, including every
   identity wire. In particular, do not put width N identity layers through
   B ell sequential layers, which costs Omega(NB ell), and do not place
   two width-N dense layers consecutively while claiming only O(N) slots.

The old package's unavailable tree-selector proof/test sources prevent an
independent check of this interface. The lookup lemma in Section 4 removes
the large-weight table-selection obstacle, but does not prove all five items.

CONDITIONAL THEOREM. If this interface exists with constant H for any needed
smooth extension, then for every f in the stated Hölder ball the explicitly
unrolled bounded-state jet decoder, polynomial circuits, and median give

    ||f-Phi||_infinity <= C_{d,s,H}(NB)^{-s/d},
    P_actual <= C_{d,s,H}[N+B ell],
    L <= C_{d,s,H}[B ell+log(N+1)],
    w <= C_{d,s,H}[N+B^{1/d}+J+d+1],
    max weight and bias <= R_{d,s,H},
    b_frac >= max(2[J a(B-1)+1], ceil(sq), 2m+O_{d,s}(1), C_{d,s}ell).

Equivalently sufficient precision is O_{d,s,H}(B+log(NB)). Packed corrections
use exactly J a(B-1) bits plus a sentinel; the J initial physical jets use
J ceil(sq)+O(J log(1+H)) bits per block. Q shifted copies multiply costs by
Q, and the median adds O(Q^2). Boundary error adds zero. The local error
constant is H*C_T+J/2 plus the allocated polynomial tolerance.

PROOF OF THE IMPLICATION. There are B-1 steps. Per step, transport needs at
most J^2 affine coefficients; the correction reader uses O(Ja) constant-size
gates; evaluating a degree-k polynomial uses O(J(k+1)m) gates. All carried
channels have dimension O_{d,s}(1), so their identity cost is O_{d,s}(1)
per layer. Therefore the sequential cost is
O(B[J^2+Ja+J(k+1)m])=O_{d,s}(B ell) and its width is constant. Add the
interface and N-entry lookup costs, then Q parallel copies and the median.
The stated precision includes the code, initial jets, fixed spatial constants,
and multiplication coefficients. Sections 2 and 5 prove the uniform error
on safe sets and then everywhere. This proves the conditional implication;
it does not discharge the interface premise.

The candidate P_actual and L estimates therefore survive the arithmetic,
packing, table lookup, and carry accounting, but remain UNPROVED for the full
cube construction until that interface is compiled and checked. Constants
include J=binom(d+k,d), A=2^{k+1}, Q=2d+1 and a=ceil(log2(2C+1)); a claim
with constants independent of dimension/smoothness would be unjustified.

## 8. What rate the conditional bounds would actually imply

Assume b and L are at least sufficiently large multiples of log P. Taking
N comparable to P and

    B comparable to min(b, L/log P, P/log P)

within sufficiently small d,s-dependent constants would yield

    epsilon <= C [P min(b,L/log P,P/log P)]^{-s/d}.

Dyadic rounding of the block sizes changes only d-dependent constants. In
particular, the (Pb)^{-s/d} rate would follow when
log P << b <= c min(L/log P,P/log P). The depth-limited estimate loses
two logarithms relative to the lower ingredient (PL log P)^{-s/d}.
There is no derived theorem here for every prescribed width, for b=O(1),
or for every pair (P,L). A depth-precision conversion must charge all
coefficients and layers again; quoting it does not preserve the same P,L.

The two lower-bound ingredients alone never imply their matching upper bound.
Counting also requires a fixed architecture/sparsity pattern or charging the
information in target-dependent architecture selection.

## 9. Literature and independently executed evidence

Ou, Schenkel, Bölcskei, [Three Quantization Regimes for ReLU Networks](https://www.mins.ee.ethz.ch/pubs/files/ca2024.pdf),
Theorem 5.1, proves one-dimensional Lipschitz upper bounds in the three regimes.
Its proper-quantization regime is (w^2 L b)^{-1}; its over-quantization upper
bound is (w^2 L^2 log w)^{-1}. The stated lower over-quantization bound contains
log w+log L, so a uniform constant-factor equivalence using only log w needs
an additional relation between L and w. Its Section 1 says the Lipschitz
results can be extended to d dimensions, but does not give the general
smoothness/precision theorem sought here. Proposition 4.1 is a conversion
with width factor 16 and depth factor k+2, not a free change of precision.

Yarotsky and Zhevnerchuk, [The phase diagram of approximation rates for deep neural networks](https://arxiv.org/html/1906.09477),
Appendix A.3-A.5, gives smooth Taylor encoding and explicitly charges the
polynomial-evaluation logarithm. Those results do not by themselves certify
the bounded-magnitude quantized spatial interface above.

`quantization_controls.py` uses Fraction arithmetic and seed 15112026. Its
83,362 assertions pass. They cover jet dimensions, integer translations and
their group identity, correction bounds, exhaustive finite bitstrings of
length <=10, sentinel perturbations, all lookup nodes and interpolation
samples, multiplication, grid coverage in dimensions 1-6, and median
corruption. The new packed quadratic example uses 248 bits plus a sentinel,
498 stored fractional bits, and 1,025 sample points; maximum error 1/1024
equals its registered bound. It is a new test, not a reproduction of the
unavailable historical 863-point test. None of these counts proves a theorem.

`MathCan/QuantizedJet.lean` formalizes exact scaled nearest rounding, the
corrected sign identity, one-step and whole-stream stability, a correction
alphabet bound, and the finite-dimensional predictor reset. It does not
formalize real differentiation, the matrix norm estimate, or a full network.
Its printed axiom dependencies are recorded in the replay report.
