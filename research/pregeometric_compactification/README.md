# PGC-01: Exact operational influence

PGC-01 supplies finite Boolean registers, a discrete update clock and a family of interventions. It recovers the resulting response functions and checks the conditions needed to interpret those responses as a relation. Exhaustive computation establishes the finite identities below. The accompanying [assumptions ledger](ASSUMPTIONS.json) keeps the proposed spacetime questions open. The [source audit](ASSUMPTION_AUDIT.md) examines the Lorentzian, dimension, representation and coupling assumptions behind the relevant reconstruction arguments.

The main executable result concerns cancellation. Two paths can transmit a source change to intermediate registers while their contributions to a later target cancel. The measured response relation then fails transitivity. An additional control on an intermediate register changes that result, so the permitted intervention family forms part of every influence claim.

## Run and replay

Python 3.10 or later is sufficient. The implementation uses the standard library.

```sh
python3 verify_pgc01.py --output receipt.json
python3 verify_pgc01.py --check receipt.json --output replay.json
```

The first command generates the canonical receipt. The second repeats every calculation and requires exact equality with the supplied receipt, including the hashes of the computational source files. A mismatch raises an exception and returns a failing process status. The separate runtime file records elapsed seconds; timing stays outside the canonical result.

| File | Role |
|---|---|
| `pgc01.py` | Boolean update engine, paired interventions, symbolic Boolean polynomials and relation diagnostics |
| `verify_pgc01.py` | Exhaustive comparisons, counterexamples, relabeling checks and receipt generation |
| `FIXTURES.json` | Declarative update rules and purposes for the nine named systems |
| `ASSUMPTIONS.json` | Supplied primitives, recovered quantities and unresolved physical targets |
| `receipt.json` | Exact results, counts, case digests and source hashes |
| `receipt.runtime.json` | Measured runtime for the most recent generation |

## Operational definitions

Let the state space be \(X=\{0,1\}^n\). A supplied map \(F:X\to X\) updates all registers simultaneously. Register indices label independently addressable factors of the state space. The integer \(h\geq1\) counts applications of this update map.

For a source register \(i\), let \(c\) specify every other initial register. Write \(c^{i,a}\) for the complete state obtained by setting register \(i\) to \(a\in\{0,1\}\). The paired response of target \(j\) after \(h\) updates is

\[
r_{ij}^{(h)}(c)
= (F^h(c^{i,1}))_j-(F^h(c^{i,0}))_j.
\]

This subtraction uses ordinary integers, so the response belongs to \(\{-1,0,1\}\). The program records

\[
I_{ij}^{(h)}=\max_c |r_{ij}^{(h)}(c)|,
\qquad
A_{ij}^{(h)}=\frac{1}{2^{n-1}}\sum_c |r_{ij}^{(h)}(c)|,
\qquad
S_{ij}^{(h)}=\frac{1}{2^{n-1}}\sum_c r_{ij}^{(h)}(c).
\]

Here \(I\) records the existence of a responding context. The absolute mean \(A\) gives the fraction of contexts that respond under the supplied uniform measure. The signed mean \(S\) equals the difference in target-one probabilities after averaging over that same measure.

For deterministic binary observations, the total variation distance between the two conditional point distributions equals \(|r|\), and averaging that distance gives \(A\). Comparing the two averaged target distributions gives \(|S|\). The inequality \(|S|\leq A\leq I\) explains why marginal distributions can conceal a context-specific response. With every context given positive probability, \(A>0\) is equivalent to \(I=1\).

Receipt matrices use **row source, column target**. Mean values remain exact because the JSON stores integer numerators and the common denominator `contexts_per_source`.

## Independent checks and proofs

### Boolean derivative identity

Every Boolean function has a unique multilinear polynomial over \(\mathbb F_2\), with the relations \(x_i^2=x_i\). Denote the polynomial of \((F^h(x))_j\) by \(P_j^{(h)}\). Its Boolean derivative is

\[
D_iP_j^{(h)}(x)
=P_j^{(h)}(x_i=1)+P_j^{(h)}(x_i=0).
\]

**Proposition.** The absolute paired response equals the value of this derivative at the context. Consequently, \(I_{ij}^{(h)}=1\) exactly when \(D_iP_j^{(h)}\) is a nonzero Boolean polynomial.

**Proof.** Write the polynomial uniquely as \(P=P_0+x_iP_1\), where both pieces exclude \(x_i\). Evaluation at zero gives \(P_0\); evaluation at one gives \(P_0+P_1\). Their sum over \(\mathbb F_2\) is \(P_1\), which equals one precisely when the two binary outputs differ. Uniqueness of the multilinear representation follows inductively from the same decomposition: the two restrictions recover both pieces. A nonzero multilinear polynomial therefore takes the value one somewhere on the Boolean cube. This gives the claimed equivalence. \(\square\)

The operational implementation evaluates prefix-expression trees and composes integer truth tables. The oracle uses sets of monomial bitmasks and symbolic substitution in the Boolean polynomial ring. The verifier compares every observed bit, every derivative value and every aggregate response entry through the declared horizons.

### Affine response and path cancellation

An affine system has the form \(F(x)=Ax+b\) over \(\mathbb F_2\). Repeated substitution gives

\[
F^h(x)=A^h x+\sum_{r=0}^{h-1}A^r b.
\]

Changing initial register \(i\) adds the vector \(e_i\). The difference between the two resulting states is \(A^he_i\), so

\[
I_{ij}^{(h)}=(A^h)_{ji}.
\]

This statement holds in every context and for every bias. A separate bit-matrix multiplication implementation supplies a third check of the affine influence entries. Since matrix powers use addition modulo two, two equal path contributions cancel. A structural directed path alone therefore leaves the measured source-to-target response undecided.

All \(2^{n^2+n}\) affine systems are enumerated for each of \(n=1,2,3\), with every initial state and lags one through three. These finite bounds describe the exhaustive part of the computation.

### Register relabeling

Let \(P_\pi\) permute register addresses, and define the relabeled dynamics by \(F_\pi=P_\pi F P_\pi^{-1}\). Composition gives \(F_\pi^h=P_\pi F^h P_\pi^{-1}\). A paired intervention on \(i\) becomes the paired intervention on \(\pi(i)\); its target becomes \(\pi(j)\). The same integer response follows after transforming the context. Uniform averaging preserves the correspondence.

The verifier applies every register permutation to each named fixture and compares complete trajectories and response matrices. This symmetry test concerns the supplied register addresses.

## Controlled systems and adverse cases

Every update below acts simultaneously. The symbol \(\oplus\) denotes XOR, multiplication denotes AND, and \(\lor\) denotes OR.

| Fixture | Update rule | Result checked |
|---|---|---|
| Chain | \((0,x_0,x_1)\) | The one-update relation omits the two-update response; the accumulated strict relation gives \(0\prec1\prec2\) with \(0\prec2\). |
| XOR diamond | \((0,x_0,x_0,x_1\oplus x_2)\) | Equal paths cancel at the final target; a middle-register control changes the effect. |
| AND context | \((x_0,x_1,x_0x_1)\) | The source response is zero when the other input is zero and one when it is one. |
| OR context | \((x_0,x_1,x_0\lor x_1)\) | Saturation removes the source response when the other input is one. |
| XOR mask | \((x_0,x_1,x_0\oplus x_1)\) | Every context responds, while the two uniformly averaged target distributions coincide. |
| Feedback swap | \((x_1,x_0)\) | The register relation contains a directed two-cycle. |
| Nonlinear feedback | \((x_1,x_2,x_0\oplus x_1x_2)\) | Nonlinear symbolic composition agrees with complete trajectory enumeration. |
| Causal observation model | \((U,U,A)\) on registers \((U,A,B)\) | A forced change to initial \(A\) changes the later \(B\) distribution. |
| Common-source observation model | \((U,U,U)\) on registers \((U,A,B)\) | The same unperturbed observations arise through the common source. |

### The diamond's transitivity obstruction

For the diamond, a change to \(x_0\) at tick zero changes both middle registers at tick one. Either middle register, when varied independently at tick one, changes target register three at tick two. The original source changes both middle values together, and the target computes their XOR. Its value at tick two is

\[
x_0\oplus x_0=0.
\]

Thus the source-to-target response is zero across every initial context. The lag-union relation contains \(0\to1\) and \(1\to3\), while \(0\to3\) remains absent. The same example applies to the event labels \((0,0),(1,1),(3,2)\), so retaining update times still leaves this response relation intransitive.

The verifier then clamps middle register two to zero immediately after the first update. For initial source settings zero and one, the final target values become zero and one. This expanded operation family exposes a source response. A transitive closure of the original response graph adds an edge that the original experiments lack; interpreting that closure requires a separate definition of the relation being reconstructed.

### Context dependence and averaging

For target \(x_0x_1\), changing \(x_0\) produces response \(x_1\). A probe performed only in the context \(x_1=0\) misses the dependence that appears at \(x_1=1\). The OR fixture gives the complementary saturation case.

For target \(x_0\oplus x_1\), the signed response is one when \(x_1=0\) and minus one when \(x_1=1\). Consequently, \(I=A=1\) and \(S=0\). Under a uniform context, both source settings produce target-one probability \(1/2\). The receipt preserves all these quantities separately.

### Feedback and supplied event time

The swap map sends register zero to register one and register one to register zero. Removing event times from the labels produces the cycle \(0\to1\to0\). When labels include the supplied clock, every update edge advances one tick, which makes that event graph acyclic by construction.

`relation_diagnostics` explicitly examines register labels after this time collapse. It records self responses separately, removes diagonal pairs for the candidate strict relation, then reports transitivity witnesses and cycles. The diamond shows the additional transitivity issue even for event labels. Both distinctions matter before an observed relation is treated as a causal partial order.

### Observations and interventions

Take \(U\) uniform on \(\{0,1\}\) and initialize \((U,A,B)=(U,U,U)\). Both observation models preserve these states. Their complete unperturbed histories of \((A,B)\) therefore agree: each consists of either all zeros or all ones, with equal probabilities.

Now set initial \(A=0\) and observe \(B\) after one update. The causal model gives \(P(B=1)=0\). In the common-source model, \(P(B=1)=1/2\). The target distributions have total variation distance \(1/2\). Their shared observational law leaves the model identity unresolved, and this intervention separates the declared pair.

## Finite-horizon identifiability

For any chosen horizon \(H\geq1\), use \(H+2\) registers. The first system shifts values along a chain:

\[
x'_0=0,\qquad x'_k=x_{k-1}\quad(1\leq k\leq H+1).
\]

The second system replaces the final update by \(x'_{H+1}=0\). The probe initializes every register except zero to zero, sets register zero to either zero or one, and observes only register \(H+1\).

**Proposition.** Both systems give identical transcripts through tick \(H\) for every permitted source setting. At tick \(H+1\), the setting one distinguishes them.

**Proof.** In the full chain, the source value moves one register per update, so it reaches the observed register after exactly \(H+1\) updates. All earlier values at that register come from zero initial values. In the severed system, the observed register is reset to zero at every update. At tick \(H+1\), the full chain carries the source value and the severed system carries zero. \(\square\)

The executable checks \(H\in\{1,2,3,4,6,8\}\). The number of surviving candidates within this declared two-system family is two through the observed horizon and one after the additional discriminating probe. These counts belong to the stated intervention and observation contract. Setting the penultimate register to one at tick zero would separate the models after one update. The construction establishes the need to carry the probe contract and candidate family into every finite identifiability claim.

## Verification receipt

The reference run reports `PASS` with the following counts:

| Verified quantity | Count |
|---|---:|
| Affine systems, widths one through three | 4,164 |
| Affine system-lag profiles | 12,492 |
| Affine paired-context bit responses checked against symbolic derivatives | 443,916 |
| Affine influence entries checked against independent matrix powers | 111,372 |
| Affine trajectory output bits checked against symbolic composition | 296,472 |
| Named fixtures | 9 |
| Named-fixture system-lag profiles | 36 |
| Named-fixture paired-context bit responses | 1,552 |
| Register permutations across named fixtures | 68 |
| Relabeled trajectory states compared | 2,912 |
| Relabeled response entries compared | 3,080 |
| Finite-horizon ambiguity cases | 6 |

Some named fixtures also belong to the affine family. Each row counts the stated comparisons, so the table leaves those sets distinct. Case digests cover the finite affine profiles and the relabeled fixture profiles. The receipt also stores the named systems, relation witnesses and ambiguity traces for direct inspection.

The receipt demonstrates consistency among the independent finite representations used here. The proof statements explain the mathematical identities and specify their scope.

## Assumptions and the spacetime questions

| Item | Status in PGC-01 |
|---|---|
| Subsystem addresses | Supplied as individually accessible register factors. |
| Clock and direction | Supplied through integer update ticks. |
| State alphabet | Chosen to be Boolean. |
| Allowed operations | Initial paired settings; one explicitly marked example adds an intermediate clamp. |
| Dynamics | Supplied as simultaneous Boolean expressions. |
| Context availability | Exhaustive for the small response tests; restricted by the declared horizon probe. |
| Influence | Recovered exactly within the stated operation class. |
| Partial order | Examined through explicit diagnostics; the accepted model family contains counterexamples. |
| Lorentzian manifold | A continuum limit, differentiable structure and metric remain open. |
| Null causal structure | A universal characteristic cone requires further dynamics. |
| Four dimensions | Dimension selection requires a new scale-dependent observable and competing model families. |
| Einstein variables | The experiment leaves the choice of gravitational representation open. |
| Newton coupling | A gravitational response observable and its normalization still need definitions. |

The supplied clock and register factorization already encode substantial operational structure. A proposal that derives them must start from a different primitive class or give an independent reconstruction theorem. Graph size, gate count and register count have clear meanings here; each requires a further argument before it can serve as a spacetime dimension or a physical coupling.

## Next experiment boundary

A subsequent experiment can ask whether a restricted response dataset admits a compact representation whose predictions survive new interventions. That experiment needs a stated candidate family and a fixed description language so that a representation cost can be reproduced. It should include the raw transition-table description as a baseline and carry observationally equivalent candidates together until a probe separates them.

Geometry selection would then require an observable that favors a geometric representation across scales and survives changes to the available operations. The cancellation and masking fixtures provide finite checks for that inference pipeline. PGC-01 supplies their exact responses and the counterexamples that such a pipeline must handle.

## Companion experiments and formal proof

[SelfModels' observer-budget experiment](https://github.com/MoralityLabAI/SelfModels/tree/research/pgc01-observer-budget-20261126/research/pregeometric_observer) exhausts all 16 Boolean response rules with two inputs. It checks all 81 partial truth tables and proves exact query bounds within that finite class. A fixed observer with eight states attains an expected three queries and a worst case of four under the declared uniform prior. Its state register requires three bits; the fixed program and external query interface have separate costs.

[The MathCan module](https://github.com/patrickdugan/MathCan/blob/research/pgc01-operational-identifiability-20261126/MathCan/OperationalIdentifiability.lean) proves four statements about observation fibers and deterministic decoding. All four compile under Lean 4.35.0 with empty axiom dependency lists. The mathematical obstruction applies whenever an explicit admissible pair agrees on accessible records and differs on the target property. A physical spacetime application still needs such a pair and a justified model class.
