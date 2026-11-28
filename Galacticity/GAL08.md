# GAL-08: Structural certificate galacticity

Status: restricted access lower bound, structural query theorem, Boolean
certificate checker, explicit normal-form blowup control, and reuse-cost
campaign. Core-Lean candidates accompany the paper arguments.
No cryptographic implementation, native model training, or timing result.

## 1. Certificate access versus statement binding

Unknown f:{1,...,N}->Bool, promised all zero or one spike. A verifier has
a certificate string z and query access to f. Deterministic completeness:
there exists z accepted on all-zero f. Soundness: every z is rejected
on every spike f. Fix the accepting z and clean transcript. If a
coordinate was unread, insert a spike there. The same z and transcript
are preserved, contradicting soundness. Thus accepting a clean function
requires N queries even with arbitrarily long certificates, absent an
additional binding/access interface. Randomized extension: fix a
certificate with clean acceptance probability p; if every certificate
and spike has false acceptance at most e, coupling yields expected
distinct queries >=N max(0,p-e), exactly as GAL-07. This assumes the
certificate is fixed before private verifier randomness. An interactive
prover with a different protocol requires its own model.

A proof of a public circuit's behavior is different from a certificate
about an unbound black box. The circuit syntax must be the actual
implementation under consideration. Hashes/commitments can bind to
a representation under cryptographic assumptions, but a hash alone
does not certify semantics or prove that the running system uses that
representation. A succinct argument can verify a statement about a
committed computation under its own security/access contract; charge
commitment/prover setup, input binding, verification, and reuse.
No claim that an arbitrary black-box property admits a succinct proof.

## 2. Trusted structure changes query complexity

Consider unknown scalar f:GF(2)^n->GF(2), testing universal equality
to zero. For unrestricted f the deterministic exact worst case on the
zero function is 2^n queries (one-spike adversary). For linear f(x)=a.x,
query the n coordinate basis vectors. If the observed vectors have
rank less than n, a nonzero orthogonal a preserves all zero replies,
so n is necessary. For affine f(x)=a.x+b, lift each query to (x,1);
the lifted points span n+1, hence n+1 independent measurements are
necessary, and 0 plus the coordinate basis attains the bound.
The assumption is the true function class, not a claim supplied by an
untrusted prover. For dimension 40 the counts are 2^40,40,41.
General finite binary linear observation classes have exact query
complexity equal to the dimension of the class when evaluation
functionals span its dual (they do for distinct functions).

gal08_structure.mjs checks all 65,812 subsets of evaluation vectors
for n=1..4, with explicit annihilator search; all 1,020 affine
functions for n=1..8 and 502 nonzero linear coefficient vectors.
74,662 assertions. Wide profiles are formulas, not exhaustive runs.

## 3. Circuit-local structural proof

Input is the public, trusted-to-be-the-implementation acyclic Boolean
gate list with ordered wire references, n input bits, and output index.
The certificate supplies an affine pair (a,b) for every gate. The
checker validates all references and coefficient widths, compares
each row to the exact local consequence of earlier checked rows,
and binds the root to the requested affine target. Constants and
inputs have canonical forms; XOR combines coefficients and biases.
For AND of affine forms (a.x+b)(c.x+d), multiplication is affine iff
a=0, c=0, or a=c. The sufficiency cases are constant multiplication
and idempotence/complementarity. Otherwise the distinct nonzero
linear forms are independent over GF(2), the map x -> (a.x,c.x) is
surjective, and restricting to a 2D affine square shows the mixed
second difference of the product is 1 whereas any affine function
has mixed difference 0. Induction over topologically ordered gates
establishes soundness of accepted rows. Completeness holds for
circuits whose every gate computes an affine function; an affine
output alone need not admit this certificate. Builder returns
unsupported on the first non-affine gate. The verifier can recompute
the same forms, so this particular witness is not a shorter-time
proof than trusted recomputation. It is an explicit independently
checkable intermediate-state receipt.

With M gates and n inputs, dense coefficient payload is M(n+1)
bits and local symbolic work O(Mn), compared with M*2^n gate
evaluations for naive truth-table execution. Actual source and
certificate encodings, parsing, allocation, comparisons, indexing,
hashing and proof binding need separate accounting. The declared
construction units charge one per input/constant and n+1 per
binary gate; verification additionally charges n+1 per coefficient
row comparison and n+1 for the target comparison. These logical
units approximate dense bit operations, not an end-to-end time or
physical-memory bound. Builder and checker use BigInt coefficients;
truth table controls independently execute the circuit on small
inputs. Wide profiles only exercise the structural checker.

gal08_certificates.mjs checks all 5,456 pairs of affine functions
over n=1..5 for the AND criterion; 256 generated affine-at-every-gate
circuits on n=1..8 against full truth tables; 768 corrupt/truncated/
wrong-target controls; unrestricted random controls with 193
unsupported circuits. 29,010 assertions including the ANF controls.
The unsupported count means a nonlinear intermediate encountered,
not necessarily a nonlinear output.

## 4. Certificate discovery is representation dependent

Let P_n = product_i(1+x_i) in the Boolean polynomial ring GF(2)
with x_i^2=x_i. Its explicit algebraic normal form contains all 2^n
distinct squarefree monomials. Yet P_n XOR P_n is identically zero.
A bottom-up explicit ANF algorithm expands P_n before discovering
the cancellation. It therefore requires at least 2^n monomials at
that intermediate gate, independent of the tiny final result.
The fixture uses one shared wire for P_n and an output XOR of that
wire with itself; a local identity certifies zero without expanding
P_n. The gate-local affine checker rejects the nonlinear intermediate
for n>=2, as designed. With 65,536 monomial cap, the explicit ANF
algorithm succeeds at n<=16 (registered powers) and aborts at n=20,24.
The cap is for the size of an individual evolving term set; retained
DAG rows, transient duplicates, allocations and total memory are
not capped. The local rule is valid only against the authentic
syntax; structural well-formedness must still be checked. This is
a lower bound on materializing this explicit intermediate ANF, not
an irreducible proof-complexity lower bound across all proof systems.
It provides a negative control against conflating final algebraic
rank, certificate size, search work, and verification cost.

## 5. Economics of verified reuse

Fix a circuit, a checked representation, and repeated evaluation
requests. Reuse is valid until circuit/version/specification changes.
Let D be discovery/construction, V verification, B any additional
setup or binding, and u,v marginal direct/compact execution costs.
Then q-fold gain requires m(u-qv)>=q(D+V+B). All costs use the same
declared scalar meter; never divide counts of different operations
without a pricing assumption. For positive margin the least positive
integer horizon is max(1,ceil(q(D+V+B)/(u-qv))). At zero margin only
zero setup can attain equality; negative margin never attains it.

gal08_economics.mjs prices one gate evaluation at w, one output
at one, and dense affine execution at 2n+1 (one AND and XOR per
coefficient, output; bias used as initial accumulator). Direct
cost u=w*M+1, compact v=2n+1; setup is the previous construction
plus checking logical units, with additional binding B=0 only
for the declared trusted source. For n=40,M=202, setup=21,527,
first 2x horizons are 1,051,178,67 at w=1,2,4 respectively.
The direct arm is naive topological execution, not an optimal
algorithm lower bound; constant/sparse roots and dead gates may
have faster implementations. No runtime speedup is measured.
22,848 arithmetic parameter cases, 27,704 assertions.

## 6. Transfer and next research gate

BPD: exact fixed-version residual identity and lifecycle semantics
must be proven against actual tensor execution. A floating-point
approximation needs an error envelope, not a GF(2) identity.
Lean: check a small local proof before expanding a normal form;
charge elaboration, proof search, kernel checking and actual
definition/source binding separately. ZK: distinguish soundness
of the proof system from the semantics of the verified circuit,
commitment and public-input binding; charge prover and verifier
costs separately. No repo changes to BPD, Lean proof infrastructure
outside the companion module, or zk code in this stage.

The next gate is GAL-09, proof-search galacticity: construct an
explicit family with short independently checkable certificates but
high search cost under a stated discovery interface, and investigate
budgeted discovery with verified rewrites, shared subproofs, and
fallback. Avoid claiming unrestricted proof complexity from an
algorithm-specific ANF blowup.

Replay:
    node Galacticity/gal08_structure.mjs
    node Galacticity/gal08_certificates.mjs
    node Galacticity/gal08_economics.mjs
