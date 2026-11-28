# Task-preserving causal closure: exact finite-state companion

Status: handwritten proofs with exhaustive finite-state checks and exact rational arithmetic. This package exercises declared deterministic systems. Transformer measurements and proof-assistant verification remain outside its scope.

## Relationship to existing work

[EvalMath section 3](https://github.com/MoralityLabAI/EvalMath/blob/adeb45fbb6b611e0e1fc38989e28f0d4d621698b/research/interventional_compactification/RESEARCH.md#3-exact-quotient-theorem-all-operation-words) already defines the all-intervention behavioral quotient. It remains the canonical general derivation; this finite-state companion adds its constructive minimization, shortest distinguishing words, exact risk calculations, and a stronger reclosure counterexample. The [local bridge](../../note/intervention-preserving-compactification.md) and [IC-01 protocol](../../experiments/interventional-compactification/README.md) retain their ownership and scope.

These constructions specialize classical automaton minimization and bisimulation. Dean and Givan, *Model Minimization in Markov Decision Processes* (AAAI 1997), give coarsest homogeneous refinement: https://mlanthology.org/aaai/1997/dean1997aaai-model/ . Geiger et al., *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability* (JMLR 2025), develop the broader causal-abstraction setting: https://www.jmlr.org/papers/v26/23-0058.html . The basic quotient and conditional-variance identities are established mathematics. This sprint supplies an application-specific proof package and executable certificates, with no priority claim.

## Setup

Fix a nonempty finite domain X of size n, a finite intervention alphabet A, maps F_a:X->X, and a task readout h:X->Y. Every operation is total on the declared domain. For a word w=a_1...a_T, define F_w=F_(a_T) o ... o F_(a_1); the empty word acts as the identity. A code C:X->Z is considered on its attained image.

An exact autonomous task model requires maps G_a and q satisfying

    C(F_a(x)) = G_a(C(x)),       h(x) = q(C(x)).                 (1)

A behaviorally sufficient code supports predictions of h(F_w(x)) from C(x) and w for every word. Equation (1) implies behavioral sufficiency. Arbitrary sufficient codes can retain extra distinctions that break their own transition closure; section 4 gives an example.

The action maps specify the interventions. This setup supplies no causal identification from passive data. Actions have a common meaning across states, and action words are supplied externally. All guarantees are relative to these maps, this domain, and this task.

## 1. Minimal autonomous task model

Define

    x ~ y  iff  h(F_w(x)) = h(F_w(y)) for every w in A*.        (2)

**Theorem 1.** The quotient Q=X/~ is the unique coarsest task-preserving congruence, up to relabeling. It satisfies (1). Every code satisfying (1) refines Q, so it has at least |Q| attained codes.

**Proof.** Equality for every word makes (2) an equivalence relation. The empty word gives equal immediate readouts. Prefixing any suffix word by a shows F_a(x)~F_a(y) whenever x~y. Thus transitions and readouts descend to Q. For any C satisfying (1), induction on word length yields h(F_w(x))=q(G_w(C(x))). Equal codes therefore imply (2). This forces every such C-partition to refine Q. Q itself satisfies the conditions, giving minimality and uniqueness of the coarsest partition. QED.

**Corollary 1a.** A code C is behaviorally sufficient exactly when its partition refines Q. For sufficiency, map C(x) to [x] and execute the quotient. Necessity follows from any distinguishing word. This condition alone does not supply autonomous G_a on all of C's retained coordinates.

**Corollary 1b.** An exact fixed-length state code needs

    B >= ceil(log2 |Q|).                                      (3)

There are at most 2^B codewords. If 2^B<|Q|, two behaviorally distinct states collide, and some intervention word exposes the loss. The bound is attained by encoding quotient classes when transition/readout complexity is unrestricted. It counts state-code bits; learned weights, external lookup tables, and carried history have separate costs. A one-class quotient uses zero bits.

## 2. Constructive refinement and finite witness bound

Initialize P_0 by equality of h. Given P_t, define

    P_(t+1)(x) = class of (P_t(x), (P_t(F_a(x)))_(a in A)).     (4)

**Theorem 2.** P_t identifies exactly the states with equal task outputs for all words of length at most t. If r_0=|P_0|, this procedure reaches Q within n-r_0 strict refinement rounds. Every distinguishable pair has a distinguishing word of length at most n-r_0.

**Proof.** The claim for t=0 follows from the empty word. For the inductive step, (4) retains the earlier outputs and compares all words beginning with each action and followed by at most t actions. If P_(t+1)=P_t, the partition is a congruence preserving h, so it equals Q by theorem 1 and the finite-word interpretation. Every strict refinement increases the number of blocks by at least one. There are at most n blocks. By round n-r_0 the partition must have stabilized or become discrete, which is also stable. Any pair eventually separated has therefore separated by that round. QED.

The implementation records strict refinements and performs one final equality check. Breadth-first search on ordered state pairs returns a shortest distinguishing word. An equivalent pair exhausts a graph containing at most n^2 nodes. The global code witness returns an aliased pair and its shortest word; it makes no claim that this pair minimizes word length over all aliased pairs.

A finite-horizon P_t may still require refinement and hence lack an autonomous update on the same P_t. Horizon-indexed updates consume the remaining horizon. This reproduces the existing bridge's horizon qualification.

**Corollary 2a.** Enlarging A while keeping the shared maps and task fixed can only refine Q. It increases the set of words required by (2). The minimum state-code budget is therefore nondecreasing with intervention coverage.

## 3. A monotone recoverability threshold with a fixed target

For this section take h to be scalar and fix a probability measure mu on X. For each word w, set Y_w=h(F_w(X)). Keep its numeric scale fixed across codes. Define

    R_w(C) = inf_g E_mu[(Y_w-g(C(X)))^2]
           = E_mu[Var(Y_w | C(X))].                           (5)

**Theorem 3.** If D=f o C, then

    R_w(D)-R_w(C)
      = E_mu[(E[Y_w|C(X)]-E[Y_w|D(X)])^2] >= 0.               (6)

**Proof.** Conditional expectation onto D is a further projection of the conditional expectation onto C. Expand Y_w-E[Y_w|D] as the sum of Y_w-E[Y_w|C] and E[Y_w|C]-E[Y_w|D]. Conditional orthogonality makes the cross term zero. QED.

Consequently, for a fixed horizon H,

    R_H(C) = max_(|w|<=H) R_w(C)                              (7)

is nondecreasing along deterministic successive coarse-grainings C_(k+1)=f_k o C_k. For a fixed tolerance delta>=0,

    k* = min{k : R_H(C_k)>delta}                              (8)

is persistent: every later code also exceeds delta. When the set is empty use k*=infinity. R_H also grows monotonically with H because its word set expands. A fixed nonnegative weighting of word risks gives another monotone-in-compression score; renormalizing the weights as H changes needs a separate argument.

If mu has full support and H>=n-r_0, R_H(C)=0 exactly when C refines Q. With restricted support, the statement covers positive-mass initial states. Each risk optimizes over all readouts; fitted predictors give upper bounds on these optima. Restricted predictor classes need closure under composition with f before the same monotonicity argument applies. Changing mu, target scale, intervention semantics, or the coding side information changes the claim.

This is a behavioral loss threshold. The self-transition risk from the first sprint uses the changing target C(F(X)), so equation (6) does not apply to it.

## 4. Nonconstant task preservation still permits reclosure

Take four states and one action:

    F    = (0, 2, 0, 3)
    h    = (0, 0, 0, 1)
    C_0  = (0, 1, 2, 3)
    C_1  = (0, 0, 1, 2)
    C_2  = (0, 0, 0, 1).

These are successive deterministic coarse-grainings. Since h(F(x))=h(x), every code preserves all future task outputs. C_1 aliases states 0 and 1, whose successors have C_1 codes 0 and 1. Its own transition fails closure. C_2 has two states, remains nonconstant, and evolves by the identity. Under a uniform initial-state measure,

    self-transition risks:    0, 1/8, 0
    fixed-task future risks:  0,   0, 0.

Thus requiring a nonconstant preserved task leaves the reclosure phenomenon intact. Behavioral sufficiency and autonomous evolution of every retained coordinate require separate checks. Numeric self-transition risk depends on code labels; the zero/nonzero closure criterion is invariant under bijective relabeling.

## 5. Exact loss certificate for a discarded distinction

**Theorem 4.** For any scalar target Y:X->R and any code C on finite X,

    inf_g max_x (Y(x)-g(C(x)))^2
      = (1/4) max_z [max_(C(x)=z)Y(x)-min_(C(x)=z)Y(x)]^2.    (9)

**Proof.** On each fiber, the two endpoint outputs differ by Delta. Every prediction has absolute error at least Delta/2 on one endpoint. Their midpoint achieves that radius on the entire scalar interval. Choose this midpoint separately on every fiber and take the largest squared radius. QED.

Apply (9) to Y_w. Any aliased x,y with output gap Delta gives worst-case squared error at least Delta^2/4. Under the equal-weight distribution on that pair, its Bayes error has the same lower bound. A different predeclared measure can assign little or zero mass to the witness, so a large worst-case gap need not imply a large average risk there.

### Hidden-bit example

Let x=(y,b) range over two bits, let h(x)=y, and include identity plus reveal(y,b)=(b,b). With identity alone, Q has two states and needs one bit. Adding reveal yields four classes and needs two bits.

For C(y,b)=y, immediate task risk is zero. Under uniform states, the task risk after reveal is 1/4, which also equals the minimax loss. States (0,0) and (0,1) give an explicit equal-code witness. The word consisting of reveal separates their outputs by one.

## Reproduction and scope

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/dynamical_closure -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s research/dynamical_closure -v
```

The 14 tests include the first sprint's four tests, all 5,898 systems with 1-3 states, two indexed deterministic actions, and binary tasks, plus 29,290 candidate partitions in the minimality audit. They verify the finite-word semantics, quotient factorization, minimality, and shortest witnesses. Fixed-target Bayes and minimax monotonicity are exhausted over binary targets and nested partitions through four states. The Bayes checks use uniform and nonuniform positive rational weights; a separate example covers restricted support.

The source-hashed receipt records the executed commands and their results. These finite checks validate the implementation over their enumerated domains. The general statements above rely on the supplied proofs. Experiments on learned activations require a complete computational state, an intervention-closed domain, and independent evaluation of coverage. No model calls, training, paid compute, or changes to E10/IC-01 registrations occur in this sprint.
