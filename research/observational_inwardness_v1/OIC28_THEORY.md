# OIC-28: Seven-phase adaptive robustness, affine gauge, and a certified interval

**11 November 2026. Evidence grade:** finite computer-assisted UNSAT at 14 queries, a complete 20-query SAT policy verified by a separate truth-state interpreter, and explicit inconclusive resource-limited searches for intervening horizons. Exact model only.

## Registered model

Let \(n=7\). A world is \(w=(d,z_0)\in\mathbb Z_7^2\), with hidden phase \(z_0+td\pmod7\). The controller chooses one labeled sensor address \(m_t\) at each step and receives \(y_t=[z_0+td=m_t]\). An online adversary may invert at most one report. The controller must identify \(d\) for every hidden world and every allowed fault path.

The public-history belief is two 49-bit world masks \(S=(M_0,M_1)\), for zero and exactly one substitutions. If \(C_y(t,m)\) is the clean-report set, then

\[
M'_0=M_0\cap C_y,\qquad M'_1=(M_1\cap C_y)\cup(M_0\cap C_{1-y}).
\]

A verdict is valid precisely when all worlds remaining in the union have one common shift.

## OIC-T50: Affine gauge preserves the fault game

For \(u\in\mathbb Z_7^\times\), \(c\in\mathbb Z_7\), transform worlds and addresses by

\[
(d,z_0,m)\mapsto(ud,uz_0+c,um+c).
\]

Every report is invariant:

\[
[uz_0+c+tud=um+c]=[z_0+td=m].
\]

Thus the transformation preserves both fault-layer transitions and the terminal single-shift condition. All 42 affine transformations can be used to canonicalize a memo state. Reconstruct an original query marker using the modular inverse \(m=u^{-1}(m_{\mathrm{canonical}}-c)\). The independent Python verifier works directly in the untransformed coordinate system. Exhaustive unit tests cover all unit/translation combinations through the 20-step horizon.

## OIC-T51: Certified seven-phase query interval

The affine C++ solver with the sound OIC-T44 future-disagreement obstruction exhausts the 14-query game: UNSAT after 6,719,523 canonical states, including 5,045,224 pair-pruned states. Shorter queries are also impossible by monotonicity.

At Q=20, a separate action-ordering mode returns SAT after 1,097,803 states and emits a policy with 1,697 nodes, 773 leaves, maximum depth 20. An independent Python verifier traverses every feasible received-history branch and all 49 hidden worlds with zero or one substitution at each possible position, validating 1,029 paths.

\[
\boxed{15\le Q^*_{\mathrm{flip}}(7)\le20.}
\]

Searches at Q=15 through Q=19 reached declared state-expansion or wall-time limits. They remain inconclusive. The exact minimum is open.

## OIC-T52: Fixed-schedule integer formulation

For a fixed address schedule, binary variables \(X_{t,m}\) satisfy \(\sum_mX_{t,m}=1\). For every pair of worlds with different shifts, one-flip recovery requires

\[
\sum_{t,m} X_{t,m}
\left|[z_0+td=m]-[z'_0+td'=m]\right|\ge3.
\]

SciPy/HiGHS reports this integer system INFEASIBLE at Q=20. Its LP relaxation is feasible. This supports a strict adaptive-vs-fixed gap at that budget but lacks an independent integer unsatisfiability certificate; it remains solver-reported evidence. A Q=21 fixed test timed out inconclusively.

## Resources and verification

The Q20 policy consumes at most 20 sensor-response bits and 60 three-bit online address commands, 80 link bits in total. Its exact belief state uses two 49-bit masks, before counting the policy description and planner computation. Uncompressed policy JSON is approximately 39 KB.

Local exhaustive Q14 took approximately 11 seconds and 386 MB peak RSS; constructive Q20 approximately one second and 70 MB. Hardware-specific figures.

Run the Python unittest discovery pattern test_oic28.py in both normal and optimized Python. Running oic28_reference.py with --check independently audits the saved certificate, while --cpp-check recompiles the pinned C++ source and repeats both exact solver decisions, refusing to treat a resource-cap exit as impossibility. Source and canonical certificate hashes are pinned in results/oic28_reference.json.

The synthetic finite result does not establish any neural mechanism or physical inwardness.
