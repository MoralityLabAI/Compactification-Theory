# OIC-27: Six-phase one-substitution identifiability and affine gauges

**Date:** November 11, 2026. **Evidence:** two exact finite C++ decision-game searches, two independently validated winning trees, and a separate Python truth-path verifier. The fixed-schedule 21-query minimum is supported by a SciPy/HiGHS integer-feasibility calculation and a directly verified 21-query witness. These results concern the declared synthetic sensor model. They establish no theorem about consciousness, cosmology, or trained neural circuits.

## Definitions

A hidden world is \(w=(d,z_0)\in\mathbb Z_n^2\). At clock step \(t\), the controller chooses one publicly addressed sensor \(m_t\in\mathbb Z_n\), observes a binary report about \(z_t=z_0+td\pmod n\), and must identify the shift \(d\). The uncorrupted report is

\[
y_t(w,m_t)=[z_0+td\equiv m_t\pmod n].
\]

An adversary may substitute at most one report. The controller chooses each address using previously **received** bits and may stop when its posterior supports one shift. The initial phase and true shift are unknown. Exact correctness requires the right shift on every hidden world and legal online fault path.

At a received history maintain masks \(M_0,M_1\) of worlds consistent with zero and exactly one substituted report. For the set \(C_y(t,m)\) of worlds predicting bit \(y\), the exact successor is

\[
M_0'=M_0\cap C_y,\qquad M_1'=(M_1\cap C_y)\cup(M_0\cap C_{1-y}).
\]

A leaf succeeds iff all worlds in \(M_0\cup M_1\) share the same shift. The finite AND/OR recurrence over output branches and marker actions is as in [OIC-T39](OIC23_THEORY.md). Pairwise pruning uses the independently justified [OIC-T44](OIC25_THEORY.md): if two distinct-shift worlds cannot generate more than their combined remaining flip budgets' worth of different clean bits, they can follow a common corrupted transcript under every continuation policy.

## OIC-T47: The affine phase-label symmetry

Let \(u\in\mathbb Z_n^\times\) be a multiplicative unit and \(c\in\mathbb Z_n\). The affine relabeling

\[
\boxed{(d,z_0,m)\mapsto(ud,uz_0+c,um+c)}
\]

preserves every clean report at every time:

\[
[z_0+td=m]\iff[uz_0+c+tud=um+c].
\]

**Proof.** Multiplication by a unit is invertible modulo \(n\), and addition of \(c\) is a bijection. The equality of clean reports follows. A substitution acts on the observed bit, so the same corruption pattern yields the same received history in both labelings. By induction over time, transform every marker chosen by a policy and every corresponding posterior mask using the same affine map. The transformed policy has an isomorphic received-history tree and succeeds exactly when the original succeeds. All candidate shifts are transformed bijectively by \(d\mapsto ud\), preserving the terminal condition that the shift is unique. Consequently the Bellman feasibility value is constant on each affine orbit, making canonical memoization sound. QED.

For \(n=6\), the units are \(1,5\), giving twelve affine label permutations, whereas the earlier OIC-24 quotient used only the six translations. The independent affine solver canonicalizes the pair of 36-bit candidate masks under all twelve maps, with a separate inverse action transform during certificate extraction.

## OIC-T48: Exact six-phase one-substitution horizon

For \(n=6\) and at most one substituted binary sensor report,

\[
\boxed{Q^*_{\mathrm{flip}}(6)=15.}
\]

**Finite exhaustive lower bound.** Two implementations evaluate the full Boolean information-state game at \(Q=14\) with the sound pairwise obstruction and independent memoization/canonicalization code. Both return UNSAT without reaching their declared node caps:

| Implementation | Symmetry group | Completed distinct game states | Outcome at Q=14 |
|---|---|---:|---|
| `oic27_fast.cpp` | translations | 1,639,144 | UNSAT |
| `oic27_affine.cpp` | affine units and translations | 813,921 | UNSAT |

The implementations use 64-bit masks for the 36 hidden worlds in each corruption layer, avoiding the 32-bit truncation hazard noted in OIC-26. A separate unpruned run exceeded its execution limit and is **inconclusive**. The lower bound relies on OIC-T44's proved pruning condition and the finite complete searches.

**Constructive upper bound.** Both solvers find SAT at \(Q=15\), producing different complete query trees. The independent `oic27_verify.py` reconstructs beliefs directly as sets of \((d,z_0,\text{used flips})\), verifies every feasible report edge and unique-shift leaf, then separately replays every true world and all online legal fault paths.

| Policy | Tree nodes | Leaves | True-world fault paths replayed |
|---|---:|---:|---:|
| Translation quotient | 998 | 455 | 477 |
| Affine quotient | 994 | 453 | 471 |

Every replayed leaf identifies the actual \(d\). Thus \(Q^*\le15\), and the complete negative search gives \(Q^*\ge15\). These two inequalities establish the finite result. QED (computer-assisted).

The exact horizons for one substitution at \(n=2,3,4,5,6\) are therefore \(6,8,10,14,15\). The earlier conjecture \(2n+2\) was already refuted at five phases.

## OIC-T49: Even-cycle nonadaptive parity bound

For any even \(n=2k\) and an arbitrary **fixed** marker schedule that must correct up to \(e\) substituted bits across unknown shifts and phases, the query horizon obeys

\[
\boxed{Q_{\mathrm{fixed}}(n,e)\ge n(2e+1).}
\]

**Proof.** Compare hidden shift \(d=0\) against \(d'=k=n/2\). Take the first world's initial phase to be \(a\). If the second starts at \(a+k\), their hidden positions agree at odd times and occupy the opposite positions \(a,a+k\) at even times. Their clean sensor bits therefore differ only at even times when the fixed marker falls in the pair \(\{a,a+k\}\). For universal correction of \(e\) substitutions, their codewords need Hamming distance at least \(2e+1\). Thus for *each* of the \(k\) disjoint antipodal marker pairs, there must be at least \(2e+1\) addressed even-time measurements, requiring at least \(k(2e+1)\) even steps. If the second world starts at \(a\), the same argument applies to the odd-time steps. Therefore the total number of steps satisfies \(Q\ge2k(2e+1)=n(2e+1)\). QED.

For \(n=6,e=1\) this gives \(Q_{\mathrm{fixed}}\ge18\), already proving strict adaptive improvement at the 15-query exact threshold. The bound is a statement about **fixed, one-address-at-a-time schedules**, even if the decoder afterwards uses unlimited computation.

## OIC-C2: Fixed-schedule minimum at n=6

For the same six-phase one-substitution model, formulate a 0–1 integer program with variables \(x_{t,m}\in\{0,1\}\), exactly one addressed sensor per step,

\[
\sum_mx_{t,m}=1,
\]

and for every pair of worlds \((d,z),(d',z')\) with \(d\ne d'\), require a cross-shift Hamming distance of at least three:

\[
\sum_{t=0}^{Q-1}\sum_{m=0}^{n-1}
 x_{t,m}\left|[z+td=m]-[z'+td'=m]\right|\ge3.
\]

The SciPy/HiGHS mixed-integer solver reports **INFEASIBLE at Q=20**, after fixing \(m_0=0\) by the proven phase-translation symmetry. At Q=21 it returns the following fixed schedule, independently checked by integer arithmetic to have minimum cross-shift distance three:

`0,2,2,3,3,4,2,5,3,4,4,3,3,1,4,5,5,5,1,0,5`

This supports a solver-computed nonadaptive minimum of **21 queries**. The positive upper-bound witness is independently certified by enumeration of all 540 distinct-shift phase pairs. The twenty-query infeasibility result currently relies on the HiGHS integer solver; a separate proof-producing SAT/SMT check timed out. This distinction is retained in the evidence grade. The analytically proved 18-query lower bound above needs no MILP solver.

## Resource ledger and scope

The 15-query adaptive policy uses 15 returned sensor bits and 15 three-bit addresses if every address is transmitted online, for **60 online link bits**. Its finite policy tree contains 994 or 998 nodes and requires storage or equivalent policy-generation computation. Its live Boolean sufficient state has two 36-bit candidate masks, or 72 logical bits, aside from time, loop state, and implementation overhead. The solved 21-step fixed schedule has 21 returned sensor bits and can preconfigure its 21 addresses, with 63 raw address bits if stored in a direct binary encoding.

Training a neural policy, continuous-space hidden-state estimation, and physical sensor noise distributions are outside this exact reference. Its observation interface assumes one reliable address per query and at most one adversarial substitution in returned bits. Worst-case statements apply only to that contract.

## Verification commands

~~~sh
# Core, requires only Python 3 and a C++17 compiler
python research/observational_inwardness_v1/oic27_reference.py --check
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic27*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic27*.py' -v

g++ -O3 -std=c++17 research/observational_inwardness_v1/oic27_fast.cpp -o /tmp/oic27_fast
/tmp/oic27_fast 14 2500000 2 1
/tmp/oic27_fast 15 1000000 2 1 /tmp/oic27_translation_tree.json

g++ -O3 -std=c++17 research/observational_inwardness_v1/oic27_affine.cpp -o /tmp/oic27_affine
/tmp/oic27_affine 14 2000000
/tmp/oic27_affine 15 1000000 /tmp/oic27_affine_tree.json
python research/observational_inwardness_v1/oic27_verify.py /tmp/oic27_affine_tree.json --horizon 15

# Optional SciPy/HiGHS fixed-schedule MILP (not required in lightweight CI):
python research/observational_inwardness_v1/oic27_fixed_milp.py 20 --seconds 30
python research/observational_inwardness_v1/oic27_fixed_milp.py 21 --seconds 30
~~~
