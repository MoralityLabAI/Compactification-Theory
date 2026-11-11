# OIC-31: Sharp proper-marker complexity under mixed substitutions and erasures

**11 November 2026.** Exact finite synthetic sensor model. Elementary written proof and pure-standard-library executable reference. The unrestricted seven-marker, 20-query feasibility problem remains open at proof-certificate level; a 20-query integer solver has reported infeasibility without a checkable exhaustive proof.

## Interface

For a cycle of size `n`, hidden shift `d` and phase `z₀`, at time `t` a precommitted sensor marker `m_t` returns `y_t=[z₀+t d = m_t (mod n)]`. An adversary may substitute at most `e` report bits and erase at most `r` reports, with erased positions explicitly marked. The decoder knows `n` and the marker sequence but neither hidden variable. The goal is exact shift `d` for every initial phase, with no assumptions on error placement. A *proper-marker* fixed policy uses strictly fewer than `n` distinct sensor labels.

Two clean binary report strings `a,b` belonging to different shifts can produce the same received string under these budgets exactly when their Hamming distance is at most `2e+r`: erase up to `r` disagreement positions and allocate the remaining differences between the two possible substitution patterns, at most `e` per string. Consequently, zero-error shift identification is possible exactly when the cross-shift code distance obeys `d_min >= 2e+r+1`.

## OIC-T62. General proper-marker lower bound

**Theorem.** For every integer `n>=2`, a fixed schedule using a proper subset of marker labels that corrects `e` substitutions and `r` marked erasures must use at least

\[
\boxed{Q \ge n(2e+r+1).}
\]

*Proof.* Choose an unqueried label `b`. The stationary trajectory `(d,z₀)=(0,b)` emits the all-zero string. Consider all `n` initial phases with rotating shift `d=1`. A query `(t,m_t)` contributes precisely one positive report across those `n` phase trajectories, namely for `z₀=m_t-t (mod n)`. Their total number of positive reports is therefore `Q`. One rotating initial phase has at most `floor(Q/n)` positives. If `Q<n(2e+r+1)`, its report is at distance at most `2e+r` from the stationary all-zero report, so a shared corrupted receipt exists. The two shifts cannot be guaranteed distinguishable. QED.

This lower bound is *conditional on label omission*. It gives no independent infeasibility certificate for Q20 when all seven marker positions may be queried.

## OIC-T63. Prime-cycle attainment

**Theorem.** Let `p` be prime, and `k=2e+r+1` satisfy `2<=k<p`. Query any `k` distinct marker labels in consecutive blocks of `p` identical queries. Then `Q=kp` and the minimum cross-shift Hamming distance is exactly `k`. Hence this fixed proper-marker policy corrects all allowed mixed errors and erasures, and attains OIC-T62:

\[
\boxed{Q^*_{\text{fixed,proper}}(p,e,r)=p(2e+r+1)}
\qquad (2\le 2e+r+1<p).
\]

*Proof.* As in OIC-T58, every nonzero-shift trajectory visits each selected marker once during its `p`-step block, giving exactly `k` positive reports. A stationary trajectory at an unqueried phase emits all zeros and is exactly Hamming distance `k` from a rotating trajectory. Stationary trajectories at a queried marker have `p` positives and distance `p+k-2 >= k` to a rotating trajectory.

Two different nonzero slopes meet at exactly one phase/time residue because their difference is invertible modulo prime `p`. The meeting point can coincide with the scheduled marker in at most one of the `k` blocks, whose labels are distinct. The two words therefore overlap in at most one positive position, giving distance at least `2k-2>=k`. Thus every cross-shift distance is at least `k`, and the all-zero stationary word achieves equality. The previous mixed-error criterion proves correction. OIC-T62 proves optimality among proper-marker fixed policies. QED.

The restriction `k>=2` is substantive: a single-marker, single-block schedule can have identical reports for two different nonzero shifts when their positive pulses coincide. When `k>=p` this construction uses every available marker and does not prove a matching proper-marker upper bound.

## Exact test cases

| p | e substitutions | r erasures | k markers | minimum queries | minimum distance |
|---:|---:|---:|---:|---:|---:|
| 5 | 0 | 1 | 2 | 10 | 2 |
| 5 | 0 | 2 | 3 | 15 | 3 |
| 5 | 1 | 0 | 3 | 15 | 3 |
| 5 | 1 | 1 | 4 | 20 | 4 |
| 7 | 1 | 0 | 3 | 21 | 3 |
| 7 | 1 | 1 | 4 | 28 | 4 |
| 7 | 2 | 0 | 5 | 35 | 5 |
| 11 | 1 | 1 | 4 | 44 | 4 |

The Python checker computes every cross-shift distance in these examples using independent integer report encodings. A separate routine constructs an ambiguous raw receipt for short proper-marker schedules, including erasure placement and substitution allocation; tests enumerate complete small query alphabets.

## OIC-31 fixed Q20 proof-certificate ledger

The OIC-30 49-integer occupancy formulation with the exact 1,029 cross-shift distance inequalities was independently examined by an SMT-LIA formulation through the installed Z3 library. Under a shell time limit, this attempt returned no conclusive answer. The direct and reduced HiGHS models previously reported Q20 infeasibility, but lack a stand-alone integer-UNSAT certificate. The global fixed optimum is therefore **at most 21**, with **Q20 still uncertified**. The adaptive seven-phase query optimum remains in the verified interval `15..20`.

The OIC-31 mixed-error theorem supplies a sharp general result for the proper-marker design class. It is an application of classical code distance to a periodic finite-state sensor channel. Claims about actual neural models would require a separately demonstrated observation map.
