# A compiled spatial interface and a uniform quantized approximation theorem

2026-11-15 continuation. This supplies the construction missing from Section 7
of QUANTIZATION_D_DIMENSION_AUDIT_BEFORE_INTERFACE.md. The proof below is a paper proof for all
real inputs in the cube, with explicit circuit and parameter accounting.
`spatial_circuit.py` generates the layered ReLU circuits and exact resource
certificates. Its finite regression tests are implementation checks, not the
proof of the real-domain assertion. The full analytic/network theorem is not
formalized in Lean. The scaled-integer clipping and capture identities are.

## 1. Parameters and statement

Fix d>=1, s>0, k=ceil(s)-1 and alpha=s-k in (0,1]. Let F_{d,s} contain functions
on [0,1]^d with all partial derivatives through order k bounded by 1 and all
order-k partials alpha-Hölder with constant 1 for the infinity norm. At integer
s this is C^{s-1,1}; the classical C^s unit ball embeds after a d-dependent
rescaling. The Euclidean-metric Hölder convention also changes only constants.
Using f/d suffices for these usual conventions (d=1 needs no rescaling), then
multiply the output by d and clip to [-1,1]. This adds bounded integer weights;
replace the weight bound by max(R_{d,s},d) for that rescaling.

Choose powers of two n0,K>=1. Define

    m=n0 K, h=1/m, A=2n0, N=A^d, B=K^d,
    Q=2d+1, D=2^{ceil(log2 Q)}, r_D=log2 D,
    mu=h/D, q=log2 m, tau=2^{-ceil(s(q+r_D))},
    I={beta in N^d: |beta|<=k}, J=binom(d+k,d),
    A_D=2^{k+1}D^k,
    R_0=2^{1+ceil(s r_D)}, C=R_0+A_D/2+1,
    a=ceil(log2(2C+1)), p=log2(D)+1,
    F=Ja+p, S=F(B-1), ell=1+log2(NB).

THEOREM. There is an architecture independent of f such that for each
f in F_{d,s} one can explicitly choose dyadically quantized weights to obtain
a ReLU network Phi satisfying

    ||f-Phi||_infinity <= E_{d,s} h^s,
    E_{d,s}=max(1,d^k/k!)+J/2+1/2,

with every affine coefficient and bias bounded by a constant R_{d,s}
independent of n0,K. Every scalar slot, including zeros and fixed carry
coefficients, can be charged in a variable-width dense layered architecture:

    P_actual = O_{d,s}(N+B(1+log B)+ell),
    L        = O_{d,s}(B(1+log B)+ell),
    w        = O_{d,s}(N+K+1),
    b_frac   = O_{d,s}(B+ell).

In particular the previous candidate bounds
P_actual=O_{d,s}(N+B log(NB)) and
L=O_{d,s}(B log(NB)+log N) hold. Since h=2(NB)^{-1/d}, the error is
O_{d,s}((NB)^{-s/d}). All constants and sufficient precision are specified
below. This is a sufficient resource family, not an optimal theorem for every
prescribed width/depth/bit triple or for one fractional bit per weight.

The new ingredients are clipped anchors on a finer dyadic lattice, a bounded
displacement tag in each jet frame, exact plateau addresses, and sequential
one-hot capture. No extension of f outside the cube is used.

## 2. Endpoint anchors and a uniform finite transport alphabet

For phase t=0,...,Q-1 put theta=t/D and, for integer r>=0, set

    y_t(r)=clip_{[0,1]}((r-1+theta)h).

All anchors lie in the cube. In microgrid units,

    y_t(r)/mu=clip_{[0,mD]}(Dr+t-D),

an integer. Monotonicity and the 1-Lipschitz property of clipping give

    0 <= [y_t(r+1)-y_t(r)]/mu <= D.

Thus the step is one of the INTEGERS 0,...,D, including the shortened first
and last steps. Positions beyond the cube are represented by repeated boundary
anchors; derivatives are evaluated only at points inside the closed cube.
`MathCan.SpatialInterface.anchor_step` formalizes the scaled-integer identity.

There are N coarse addresses c=(c_1,...,c_d) with 0<=c_j<A. Block c contains
the B sites i in {0,...,K-1}^d at anchors

    y(c,i)=(y_t(c_j K+i_j))_j.

The used blocks have c_j<=n0; other blocks are harmless padding. Padding from
n0^d to (2n0)^d blocks is a d-dependent factor only. Each block uses the same
serpentine Hamiltonian order: append the last coordinate j, using the forward
lower-dimensional path for even j and its reverse for odd j. Induction on d
proves that it visits each site once and consecutive sites change exactly one
coordinate by +1 or -1. The changed axis and sign are fixed by the layer index.
Only the actual clipped step magnitude depends on the block.

At anchor y define

    u_beta(y)=mu^{|beta|}D^beta f(y)/(beta! tau).

For an actual signed step q_i mu e_j, |q_i|<=D,

    u_next=T(q_i e_j)u+r,
    T(v)_{beta,gamma}=binom(gamma,beta)v^{gamma-beta}

when gamma>=beta, and zero otherwise. Since only one coordinate changes,
the entries outside that coordinate are zero unless gamma_l=beta_l.
All entries are integers and every row has absolute sum at most A_D.
Indeed the row sum is bounded by D^k times
sum_l binom(beta_j+l,beta_j)<=D^k 2^{k+1}.

Taylor expansion of D^beta f on that segment gives

    |r_beta| <= (mu^s/tau) D^{s-|beta|}/[beta!(k-|beta|)!]
              <=2D^s <=R_0.

The integral Taylor remainder proves this also for alpha=1; for zero Taylor
degree it is exactly the Hölder inequality. Every segment remains in the cube.

Round each u(y) componentwise to its nearest integer, with ties upward. Set
tilde u_next=T(q_i e_j)tilde u+c_i. The corrected recurrence yields

    ||tilde u-u||_infinity<=1/2,
    ||c_i||_infinity<=R_0+A_D/2+1/2<C.

Equivalently, the encoder computes c_i=round(u_next)-T(q_i e_j)round(u).
This is an explicit integer vector; it does not require rounding a matrix.
The physical state v=tau tilde u is dyadic and bounded by 3/2 in every component.

Each frame stores the unsigned magnitude |q_i| in p bits and the J values
c_{i,beta}+C in a bits each. The path layer already fixes the axis and sign.
Consequently each frame has exactly F=Ja+p bits, from an alphabet independent
of n0,K. A block has S=F(B-1) payload bits plus a sentinel. The J initial
physical jets are stored separately, each on the tau lattice.

The displacement tag is necessary: applying the same full-step matrix to
every clipped endpoint would give an unbounded normalized correction for
smoothness s>1. It is not a cost-free change of the old decoder.

Finite certified jet data also suffices; no exact comparison of an unknown
real derivative with a rounding threshold is necessary. Choose rational
normalized jet estimates with error at most

    epsilon_u <= min{1/[4(A_D+1)], 1/[8JD^k(k+1)]},

and round those estimates instead. The integer-state error is at most
1/2+epsilon_u. The correction bound becomes at most
R_0+A_D/2+1/2+(A_D+1)epsilon_u<C, so the same alphabet works. Physical states
remain strictly inside [-2,2]. The extra polynomial-coefficient error is at
most tau JD^k epsilon_u<=tau/[8(k+1)]. Section 6's multiplication budget is
at most tau(2k+1)/[4(k+1)], leaving more than this slack below tau/2.
Therefore the same final E_{d,s} bound holds. A derivative oracle needs only
finite precision |error in D^beta f|<=beta! tau epsilon_u/mu^{|beta|}.
Acquiring that jet information efficiently, or learning it by training, is
not part of the approximation theorem.

## 3. Continuous plateau circuits with exact addresses on safe sets

Let delta=h/(8D). A phase is safe at x if every coordinate is at distance
strictly greater than h/(4D)=2delta from the translated fine-grid boundaries
(j+theta)h. For such an x define

    r_j=floor(mx_j-theta)+1, c_j=floor(r_j/K), i_j=r_j-c_jK.

These floors DEFINE the desired indices; the network implements them with
the continuous formulas below. On safe sets 0<=r_j<=m, 0<=c_j<=n0 and
0<=i_j<K. This includes x_j=0 and x_j=1 whenever that phase is safe.

Write H_delta(z)=[ReLU(z+delta)-ReLU(z-delta)]/(2delta).
It is 0 for z<=-delta and 1 for z>=delta. The coarse thresholds are

    t_c=(cK-1+theta)h, c=1,...,n0.

Since these are fine-grid boundaries, safe x gives

    z_j=c_j/A=(1/A) sum_{c=1}^{n0} H_delta(x_j-t_c).

No large coefficient 1/delta is placed on every hinge. First form all 2dn0
hinges in ONE wide layer, aggregate into d registers in the next, and then
multiply those d registers by 2DK using log2(2DK) shared doublings. This equals
1/(2delta A). All these scalar coefficients have magnitude at most 2.

The normalized lexicographic address is

    z=sum_{j=0}^{d-1} A^{j+1-d} z_{j+1}
     =(sum_j c_{j+1}A^j)/N.

Every coefficient is dyadic and at most 1. On safe sets this is EXACTLY a
table node, with no address error. On all inputs, z is in [0,1] because
0<=z_j<=n0/A=1/2. Interpolated addresses on bad phases are permitted.

Apply the bounded-weight interpolation table from Section 4 of the earlier
audit at nodes r/N, simultaneously for the packed word and J initial jets.
The shared hinge layer has width N; aggregation returns to a fixed number
of registers BEFORE the log2 N amplification. Second differences of packed
words need their stored fractional precision but no extra fractional bits.
Second differences of the bounded signed physical jets have magnitude <=6.

Next compute normalized within-block coordinates

    u_j=n0[x_j-2z_j+(1-theta)h].

This uses a narrow affine stage and log2 n0 doublings. Clip to [0,1]. On safe
sets it equals (mx_j-theta+1-c_jK)/K, so clipping changes nothing. Fine
thresholds are l/K, l=1,...,K-1, with delta_f=1/(8DK)=n0 delta. Therefore

    v_j=i_j/K=(1/K) sum_{l=1}^{K-1} H_{delta_f}(u_j-l/K).

Again use one hinge layer, narrow aggregation, and shared amplification,
now by 4D. This costs O_d(K) slots and O_d(1) further amplification layers.
The safe-set distance is at least twice the ramp radius, so all these
selectors have a nonzero margin; no discontinuous floor is a circuit gate.

## 4. Exact snake rank and local coordinates

The parity of i_j is a fixed K-entry interpolation table evaluated at i_j/K.
It is exactly 0 or 1 on safe sets. Table size is O(K), not O(B). Define

    G_M(b,z)=ReLU(z+Mb)-ReLU(z-Mb)-Mb.

For |z|<=M, G_M(0,z)=0 and G_M(1,z)=z. For arbitrary b in [0,1] its value
is bounded by M. This gives exact multiplication by a binary selector using
bounded weights. The proof follows by checking the signs of z+M and z-M;
the scaled-integer versions are Lean checked.

If q_j is the normalized snake rank for the first j coordinates, then

    q_1=v_1,
    q_{j+1}=v_{j+1}+[q_j+G_2(parity(i_{j+1}),1-K^{-j}-2q_j)]/K.

Clip each result to [0,1]. For actual integer indices, induction gives
q_j=rank_j/K^j: the second term is the forward lower rank for even outer
coordinates and the reverse lower rank for odd coordinates. The argument
of the binary gate is in [-1,1] on safe sets. Hence final q=rank/B is exact.
For K=1 use the constant rank zero.

Compute selected anchors and local coordinates by

    y_j=clip_{[0,1]}(2z_j+v_j/n0-(1-theta)h),
    xi_j=clip_{[0,1]}(m(x_j-y_j)).

The multiplication by m uses shared doublings, never a weight m. On safe
sets y_j=y_t(c_jK+i_j), and 0<=x_j-y_j<=h, so both clipping operations
preserve the required anchors/local coordinates. Thus xi=(x-y)/h is exact
even in partial endpoint cells. The entire spatial setup has O_{d,s}(N+K+ell)
slots and O_{d,s}(ell) depth, including table weights and all carry wires.

## 5. Bounded-state stream decoding and capture

Encode the S bits beta_l in a single word

    w=sum_{l=1}^S (1+beta_l)4^{-l}+4^{-(S+1)}.

It needs 2(S+1) fractional bits. The previously proved bounded digit reader
returns the bits exactly and updates the word to its suffix. A trailing
sentinel leaves a margin of at least 1/48 before every required digit read.
Clip the suffix to [0,1] after each update; it is unchanged for a valid word.
This bounds invalid interpolated streams on bad phases.

Decode the p-bit displacement magnitude by a fixed one-hot Hamming test for
each q in {0,...,D}. The test is ReLU(1-HammingDistance), with coefficients
of magnitude <=1 and integer bias of magnitude <=p. Compute T(sign*q e_j)v
for the D+1 possibilities and select with G_{2A_D}. Since |v|<=2 after state
clipping, every candidate is in [-2A_D,2A_D]. Add the decoded tau*c and clip
v to [-2,2]. On safe sets the word is valid, so these are exactly the desired
transport/correction updates, and the true quantized physical state is bounded
by 3/2: clipping never changes it. Bad phases also stay bounded.

At site t=0,...,B-1 form

    b_t=ReLU(1-B|q-t/B|).

On safe sets q=r/B for an integer rank r, so b_t=1 if t=r and 0 otherwise.
The absolute value uses two ReLUs. The B amplification uses log2 B narrow
doublings, clipped at 1 after each doubling. Thus it computes
min(B|q-t/B|,1) with weights <=2; subtracting from 1 gives b_t. No bias t
or weight B appears: t/B is dyadic and in [0,1].

Maintain J accumulators z_beta, initially zero, and at each site add
G_2(b_t,v_beta), clipping the accumulators to [-2,2]. Exactly one site adds
its jet on a safe phase. Therefore the captured z_beta is exactly the selected
quantized jet. No polynomial approximation, large-index multiplication, or
all-sites-wide carry is used in capture. Its cost is O_{d,s}(B(1+log B)).

This also explains the difference from the former standalone interface
obligation: address setup remains O(N+B+ell), but temporal capture is explicitly
charged to the decoder at O(B log B). The resulting whole-network candidate
budget is preserved; a stricter O(B) capture claim is not made.

## 6. Polynomial approximation and global boundary coverage

The selected physical coefficients satisfy

    |z_beta-mu^{|beta|}D^beta f(y)/beta!|<=tau/2.

Consequently sum_beta D^{|beta|} z_beta xi^beta differs from the true Taylor
polynomial by at most J D^k tau/2<=J h^s/2. Taylor's theorem along the segment
from y to x gives error <=max(1,d^k/k!)h^s, exactly as in the earlier audit.

For completeness the polynomial circuit uses the triangular square

    S_n(x)=x-sum_{j=1}^n 4^{-j}g^{compose j}(x),
    g(x)=2ReLU(x)-4ReLU(x-1/2)+2ReLU(x-1),

whose error on [0,1] is <=4^{-(n+1)}. A product of signed inputs in [-2,2]
is obtained by u=(x+2)/4, v=(y+2)/4 and

    xy ~= 32 S_n((u+v)/2)-8S_n(u)-8S_n(v)-8u-8v+4,

with error <=12*4^{-n}. Use

    eta=tau/[4JD^k(k+1)],  n_poly>=1 with 12*4^{-n_poly}<=eta.

Build monomials by successive multiplication and clipping to [0,1], then
multiply by the captured jet and accumulate with integer coefficients D^{|beta|}.
The monomial error is at most k eta. The output error is at most
JD^k(2k+1)eta<=tau/2. Inputs to every multiplication remain in [-2,2].
Clip the scalar output to [-1,1]; f is in that interval, so error cannot grow.
Each safe phase now has the claimed E_{d,s}h^s error.

For each coordinate, the Q boundary phases on the circle modulo h have
pairwise circular separation >=h/D. A radius h/(4D) neighborhood can contain
at most one phase. Thus at most d phases are bad at any point in the cube,
including its boundary, and at least d+1 of Q=2d+1 are safe. The median of
the Q phase outputs has the same error bound. Boundary stitching adds ZERO
error. An insertion-sorting circuit uses Q(Q-1)/2 comparators, with
max(a,b)=b+ReLU(a-b) and min(a,b)=a-ReLU(a-b). This is explicitly generated
in build_approximant; it is not an external sorting oracle in the network.

## 7. All dense slots, widths, weights and bits

The generator compiles each real register z into (ReLU(z),ReLU(-z)). A linear
register assignment computes both signed rows; a ReLU register computes its
positive row and a zero row. Each row is an affine combination of the previous
layer's pairs. This is an ordinary ReLU network; no skip connections are free.
The first split and final scalar affine reconstruction are counted explicitly.

Let

    r=64(D+1)(J+1)(d+a+p+k+1).

This bounds all simultaneously live narrow registers, including the original
input, saved phase outputs, frame bits, D+1 transport alternatives, J jet and
capture registers, polynomial temporaries and the median. For example the
largest decoder transient is bounded by
4J(D+1)+a+p+10d+Q+20, which is smaller than r. Frames are released before the
next frame; the generator does not retain the history of every extracted bit.

Each wide hinge stage is followed immediately by a narrow aggregation stage.
Their total hinge-register count is at most

    2d n0+N+2d(K-1)+(d-1)K <= (d+1)N+3dK.

There are at most d+2 such pairs. A pair with n extra wide registers costs
at most 32(r+1)^2(n+1) dense scalar slots after signed compilation. Every
narrow hidden layer costs at most (2r+1)2r<=6r^2 slots. Define

    L_*=100(d+1)(ell+r_D+1)
        +B[2 log2 B+4F+4J+40]
        +30J(k+1)(n_poly+1).

The first term covers all spatial setup, parity tables and narrow amplifiers.
Per site capture uses at most 2 log2 B+9 layers. Per transition the tagged
frame and transport use at most 4F+2J+9 layers. Polynomial multiplication uses
at most J(k+1) calls, each at most 6n_poly+11 layers, including clipped
monomial updates. The loose coefficients in L_* dominate these counts.

For one phase a sufficient explicit ledger is

    L_grid <= L_*+2,
    P_grid <=32(r+1)^2[(d+1)N+3dK+d+2]+6r^2(L_*+2),
    w <=4d(N+K)+2r.

Compute the Q shifted copies sequentially while carrying only the input and
the previous Q scalar outputs. Then a sufficient whole-network ledger is

    L <= Q(L_*+4)+Q(Q-1)+3,
    P <= Q P_grid+12r^2(Q^2+Q+1),
    w <=4d(N+K)+2r.

Serialization avoids padding a width-N layer through all decoder steps. There
are no consecutive width-N dense layers, and every carry pair is charged.

One admissible weight/bias bound is the integer

    R_{d,s}=64+4A_D+2^{a+1}+p+4JD^k.

The largest special coefficients are second differences of physical jets
(<=6), signed-product coefficients (<=32), transport gates (2A_D), signed
correction digits (<2^a), and D^{|beta|}. All depend on d,s only.

Sufficient uniform fractional precision is exactly bounded by

    b_* = max{2(S+1), ceil(s(q+r_D)), 2n_poly,
              q+r_D+3, log2 N, (d-1)log2 K, 4}.

Every scalar coefficient and bias belongs to 2^{-b_*}Z. The factorials occur
only in the offline jet data before rounding; no non-dyadic factorial reciprocal
is a fixed circuit weight. Bounds involving eta determine n_poly but eta
itself is not a circuit coefficient. Actual storage is b_*+O_{d,s}(1) bits
because R_{d,s} is fixed. Packed length, initialization precision, spatial
threshold precision and arithmetic precision are all charged. Capture biases
t/B require log2 B bits, already dominated by 2(S+1) (also when B=1).

Since n_poly=O_{d,s}(ell), S=O_{d,s}(B) and K<=B, these explicit ledgers prove
the asymptotics in the theorem. They also give the constants' dependence
through D,Q,J,A_D,C,a,p,r. No dimension-independent constant is asserted.

## 8. Precision propagation and the scope of the rate conclusion

The theorem is about exact real evaluation of a network with quantized
parameters. It does not assert stability in a fixed IEEE format. For a
specified evaluator with an absolute error at most epsilon_act per affine/ReLU
node, the spatial setup and table stages amplify errors only polynomially
in N,K: they contain a fixed d-dependent number of wide stages and shared
doublings with products N, n0, K and m. The plateau safety margin is delta.

Code decoding is the separate sensitive part: over S reads, suffix error is
bounded by 4^j e_0+(4^j-1)e_update/3 while the digit margins hold. Thus the
lookup/code input and internal errors must be O(4^{-S}), up to polynomial
factors in N,K. Matrix routing and capture have fixed Lipschitz constants per
primitive, and their bounded states turn numerical selector errors into bounded
additive errors. As long as the physical state error stays <1/4, clipping is
inactive near the ideal state. The exact translation identity T(a)T(b)=T(a+b)
then bounds jet error by

    J 2^k(1+DK)^k [initial_error+B*step_error].

This supplies a bootstrap sufficient condition for the clipped decoder; an
arbitrary product A_D^B is not necessary. Capture errors accumulate at most
a further polynomial factor B^2 from the triangular selectors. Consequently
activation precision O_{d,s}(S+ell) is sufficient under the stated bounded
node-error model, with constants depending on that model. The generated finite
tests use exact arithmetic; they do not test arbitrary hardware rounding.

For parameter budgets P and sufficiently large b,L, choose dyadic N comparable
to P (within small d,s-dependent constants) and

    B comparable to min(b,L/log P,P/log P).

If b,L are at least sufficiently large multiples of log P, this gives

    epsilon <= C_{d,s}[P min(b,L/log P,P/log P)]^{-s/d}.

In the proper range b<=c min(L/log P,P/log P), the rate (Pb)^{-s/d} is
constructively attained for this architecture family. The depth-limited bound
has logarithmic losses against the lower ingredient (PL log P)^{-s/d}. The
construction specifies its sufficient width; it does not settle arbitrary
small prescribed widths, the whole under-quantization regime, or a generic
depth-precision conversion without recounting its edges.

## 9. Reproducible implementation evidence

`spatial_circuit.py` contains no floor or exact multiplication primitive in the
generated network. Floors appear only in the offline encoding, expected-index
oracle, and this proof's definition of the desired addresses. Runtime products
are triangular-wave ReLU circuits. The matrix materializer emits every hidden
ReLU layer, its exact rational rows, and the final affine output. Omitted matrix
entries are zero and still included in dense parameter counts.

The six regression suites cover snake ranks and adjacency, integer endpoint
microsteps, binary capture gates, safe addresses/codes/coordinates in dimensions
1-3, complete tagged decoding and selected-jet capture, polynomial circuits,
and the full shifted-median network. Inputs include cube corners, internal grid
boundaries and ramp margins. The materialized numeric-matrix evaluator is
checked against the register evaluator. Detailed counts, exact resource
certificates and source hashes are in spatial_circuit_results.json; a complete
small network is in spatial_relu_example.json.gz.

The tests currently pass 8,299 explicit assertions. That finite count is not
the reason the theorem holds: Sections 2-7 prove the domain, correctness,
boundary coverage and resource bounds for every permitted d,s,n0,K and real x.
