# QO-04: The precise two-action complexity boundary for goal-aware quotient learning

**Date:** 2026-11-04. **Scope:** finite rational absorbing-goal hit/hold Markov systems, with a known transition table. Theorems below reduce their **uniform one-step projected total-variation defect**, with exact goal readout, to geometric point covering. The proof of two-action hardness is conditional on a fully specified classical geometric complexity theorem, not a claim to have invented geometric NP-hardness. There is no claim of physical xenobot or compact-model capability.

**Previous results:** [QO-01](GOAL_AWARE_RATE_DISTORTION.md) exact finite Pareto frontier, [QO-02](HIT_RATE_DP.md) polynomial one-action algorithm, and [QO-03](MULTIACTION_COLORING.md) variable-action graph-coloring reduction with only three transient macro labels. [MOR-05 Lean proof](https://github.com/patrickdugan/OntologicalProofs/blob/main/Neoplatonic/LTR/MorphospaceColoring.lean) certifies QO-03's edge-rate local equivalence.

## 1. Geometry of the declared stochastic quotient

Let transient states be \(S=\{1,\ldots,n\}\), with separate absorbing goal \(\dagger\). There are \(d\ge1\) actions, each specified by rational hit probabilities \(p_i^a\in[0,1]\), and the remainder of each transition returns to state \(i\):

\[
K_a(i,\dagger)=p_i^a,\qquad K_a(i,i)=1-p_i^a,\qquad
K_a(\dagger,\dagger)=1.
\]

All transient states have goal label zero and \(\dagger\) has goal label one, so every valid quotient assigns the goal a distinct macro label. Represent state \(i\) by the point

\[
v_i=(p_i^1,\ldots,p_i^d)\in[0,1]^d.
\]

For a partition \(\pi\) of transients into blocks \(B\), QO-03 establishes the exact macro-kernel optimum

\[
\epsilon(\pi)
=\frac12\max_{B,a}
\big(\max_{i\in B}p_i^a-\min_{i\in B}p_i^a\big)
=\frac12\max_B\operatorname{diam}_\infty\{v_i:i\in B\}.
\tag{1}
\]

The midpoint of each coordinate range specifies a valid stochastic macro transition kernel, attaining (1). The declared **uniform one-step** error \(\epsilon(\pi)\) has no implicit long-horizon or unobserved-reward guarantee.

Fix tolerance \(\tau\ge0\) and set \(r=2\tau\). A transient block has defect at most \(\tau\) exactly when its points fit inside an axis-aligned closed \(d\)-cube of side \(r\). Assigning each point to one covering cube gives a valid partition; every partition yields a cube for each block. Therefore:

**QO-04/T1 (exact geometric equivalence).**
A goal-visible quotient with at most \(k\) transient macrostates and defect at most \(\tau\) exists **if and only if** the \(n\) rational points \(v_i\) can be covered by at most \(k\) axis-aligned closed \(d\)-cubes of side \(2\tau\). The absorbing goal consumes one additional codeword. For a \(b\)-bit macrostate label, \(k\le2^b-1\).

This is a geometric reformulation of the finite QO-03 conflict-graph criterion, which interprets viable blocks as independent sets of a distance-threshold graph.

## 2. Polynomial algorithm when both action count and class budget are fixed

A covering cube \(C=\prod_j[\ell_j,\ell_j+r]\) containing a **nonempty** subset \(T\) can be moved to the canonical lower corner \(\ell'_j=\min_{i\in T}v_{ij}\). Every coordinate of every point in \(T\) remains inside \([\ell'_j,\ell'_j+r]\): the original cube contained the points, so their coordinate span is at most \(r\).

Thus every successful cover has a successful canonical cover in which each coordinate of every lower corner belongs to the corresponding coordinate list of the input points. There are at most \(n^d\) distinct candidate lower corners. Test all choices of at most \(k\) candidates and whether their union contains the input.

**QO-04/T2 (fixed-\(d,k\) exact algorithm).**
The decision problem is solvable using at most \(\sum_{j=1}^k\binom{n^d}{j}\) candidate-box combinations and \(O(dn)\) rational comparisons per selected cover. In particular its running time is

\[
O\big(d\,n^{dk+1}\big)
\]

for separately fixed \(d,k\), apart from polynomial rational-bit arithmetic. The implementation uses a bounded-\(k\) search with memoization and exact finite candidate corners; no learned geometry or floating-point rounding is assumed.

**One action:** rates lie on a line, so greedily place each next interval at the smallest uncovered rate. This gives an \(O(n\log n)\) covering decision algorithm for any \(k\) and fixed tolerance (the exact-rate optimizer of QO-02 also provides a \(O(kn^2)\) solution for the full error curve).

**Two actions and fixed \(b\):** at most \(k=2^b-1\) transient classes and \(n^2\) canonical candidate squares. The exact decision problem is polynomial in \(n\), even if \(\tau\) varies with input. For two fixed bits, \(k=3\), the direct certificate-enumeration bound is \(O(n^7)\).

**Fixed positive tolerance:** Since all rates are in \([0,1]^d\), \(C=\lceil1/(2\tau)\rceil^d\) fixed cubes suffice to cover the full unit \(d\)-cube. When \(d\) and positive \(\tau\) are *fixed constants*, the exact minimum class count can be found in polynomial time by considering at most \(C\) canonical boxes, regardless of the supplied class budget. If \(\tau=0\), grouping states by equality of their rate vectors is optimal and polynomial.

The special-case tractability here concerns **one-step** transition prediction. More general Markov kernels or multi-step policy-conditional rewards can present different computational problems.

## 3. NP-completeness with exactly two actions and variable tolerance/budget

**QO-04/T3 (two-action hardness).** When both the permitted number \(k\) of transient macrostates and rational tolerance \(\tau\) are part of the input, deciding if a two-action hit/hold system has a goal-preserving quotient with defect at most \(\tau\) is **NP-complete**.

**Published hardness premise.** Megiddo and Supowit, *On the Complexity of Some Common Geometric Location Problems*, SIAM J. Comput. 13(1), 1984, section 4, prove NP-hardness of the following *Square Covering* problem: given axis-aligned **unit squares** with rational coordinates and an integer \(k\), choose at most \(k\) points stabbing all those squares. See https://doi.org/10.1137/0213014 and https://rangevoting.org/MegiddoSupowit1984.pdf (pp. 187–189). The same square-stabbing problem is equivalent to covering the **centers** of the input squares by at most \(k\) freely placed axis-aligned unit squares: a stabbing point \(c\) lies inside a square centered at \(v\) iff \(v\) lies inside the unit square centered at \(c\).

**Polynomial reduction.** Take the rational centers \(v_i=(x_i,y_i)\) of the input squares. Let \(a_j=\min_i v_{ij}\), \(D_j=\max_i v_{ij}-a_j\), and choose rational \(\lambda=\max(1,D_1,D_2)\). Set probabilities \(p_i^1=(x_i-a_1)/\lambda\), \(p_i^2=(y_i-a_2)/\lambda\), and \(\tau=1/(2\lambda)\). These are valid rational probabilities in \([0,1]\). The unit-square covering question transforms exactly into whether the normalized rates can be partitioned into at most \(k\) blocks of coordinate diameter at most \(1/\lambda=2\tau\). By T1, the latter is the specified two-action Markov quotient decision. The transformation involves only polynomially many rational additions, subtractions and divisions of polynomial bit length.

**NP membership.** A candidate partition of the \(n\) transient states into at most \(k\) labels is a polynomial-size witness. Verify for every action and nonempty class that the rational coordinate maximum minus minimum is at most \(2\tau\). Reserve a distinct macro label for the goal. The midpoint of each class' rate range constructs a rational macro kernel with error at most \(\tau\). Thus the restricted decision problem belongs to NP and is NP-complete. \(\square\)

### Closing the fixed-length **bit budget** gap

The original geometric covering problem supplies an arbitrary class bound \(k\), whereas a \(b\)-bit macrostate representation permits \(2^b-1\) transient labels after reserving the absorbing goal. We prove hardness even when the **only** budget parameter supplied is \(b\).

Given an unnormalized geometric instance \(P\subset\mathbb Q^2\), side length \(r>0\), and \(1\le k\le n\), set

\[
 b=\lceil\log_2(k+1)\rceil,\quad
 k'=2^b-1,\quad m=k'-k.
\]

Since \(k'<2k+1\), \(0\le m\le k\), so only polynomially many extra states are needed. Append \(m\) artificial points on a horizontal ray to the right of all original points: for \(j=1,\ldots,m\),

\[
u_j=(x_{\max}+j(r+1),\ y_{\min}).
\]

Each added point is separated from every original point and every other new point by more than \(r\) in the first coordinate. Every such point **must** occupy its own square; it cannot share a square with any original point. Hence the augmented instance has a cover with at most \(k'\) squares iff the original one has a cover with at most \(k\). Translate and positively rescale **the entire augmented instance** to \([0,1]^2\), as above, and set the new rational tolerance \(\tau=r/(2\lambda)\). This preserves every pairwise box-compatibility relation and requires only polynomial bit length.

**QO-04/T4.** The two-action problem is NP-complete when the variable input budget is expressed solely as a fixed-length **\(b\)-bit total macrostate label** with exact goal visibility. Both \(b\) and rational tolerance \(\tau\) vary with the input.

An important qualification: this does **not** establish hardness for **fixed \(b=2\)** with two actions. T2 supplies a polynomial algorithm for that fixed capacity.

## 4. Machine-checkable witnesses and protocol

[geometric_quotient.py](geometric_quotient.py) provides an exact candidate-box enumerator, a bounded-class search, an unrestricted minimum-box DP, a 1D greedy oracle, sorted pairwise-distance radius search, a normalization map, a padding construction for \(b\)-bit budgets, and assignment certificates. Every coordinate and threshold uses Python \`fractions.Fraction\`.

[test_geometric_quotient.py](test_geometric_quotient.py) compares **729 two-action, three-transient-state rate matrices** (2,187 state-budget comparisons) against an independent exhaustive *partition*, repeats fixed-threshold feasibility checks, verifies the 1D greedy method against QO-02, checks a square requiring a **mixed-coordinate lower corner**, checks the worst-case four-square cover of \([0,1]^2\) at tolerance \(1/4\), and verifies a normalized/padded reduction against independent partition checks. Source code and tests are built by the existing read-only-token \`ingression-autonomy.yml\` workflow.

**Claims ledger:** T1 and T2 are explicit elementary deductions. T3/T4 are polynomial equivalence reductions from the classical geometric covering problem whose NP-hardness is cited. Python checks the **finite constructions and algorithms** and does not reprove the 1984 hardness premise. These statements concern a known, fully enumerated microstate transition model. A learned neural quotient must still undergo blinded tests on new intervention families with all representation/program costs accounted for.
