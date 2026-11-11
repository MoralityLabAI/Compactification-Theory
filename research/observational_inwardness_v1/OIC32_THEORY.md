# OIC-32: A global fixed-sensor lower bound from affine-plane line moments

**11 November 2026. Status:** a uniform handwritten proof over all primes `p >= 5`, with a new standard-library exact-arithmetic reference and seven finite regression tests. This resolves the previously open **unrestricted** seven-phase fixed-schedule Q20 decision. No HiGHS, SMT, SAT or assumed numerical infeasibility certificate is used by the proof.

## Sensor contract and code distance

A world is `(d,z) ∈ F_p²`. At time `t`, the fixed sensor address `m_t ∈ F_p` reports `y_t=[z+t d=m_t]`. Exactly one unknown report bit may be substituted. The decoder must identify `d` for every possible `z`, and the address sequence is committed before observing outputs. As established in OIC-T55, the exact correction criterion is

\[
\min_{d\ne d',\,z,z'}d_H(c_{d,z},c_{d',z'})\ge3. \tag{1}
\]

OIC-T58 already constructs such a schedule in `Q=3p` queries with three distinct marker labels, each repeated for one `p`-step block. For `p>=5`, its minimum cross-shift Hamming distance is exactly three. Thus the *upper* bound is `Q*_fixed(p,1) <= 3p`.

## OIC-T64. Global prime-cycle optimality, including all marker labels

**Theorem.** For every prime `p>=5`, and for every fixed schedule allowed to use **any or all** `p` sensor addresses,

\[
\boxed{Q^*_{\mathrm{fixed},\mathrm{one\ substitution}}(p)=3p.} \tag{2}
\]

**Proof.** It is enough to rule out `Q=3p-1`: any shorter successful schedule could be extended arbitrarily to this horizon without reducing its inter-shift Hamming distances.

Reduce query times modulo `p`. Regard each query as a point `(r,m)` in the affine plane `F_p²`, with multiplicity retained when a phase/residue address is queried repeatedly. Every hidden trajectory `(d,z)` is the nonvertical affine line

\[
L_{d,z}=\{(r,z+rd):r\in\mathbb F_p\}.
\]

Let `w(L)` count the queries on `L`, including multiplicity. There are `p` slope classes (one per `d`), each with `p` pairwise parallel lines, and **each class has total weight `Q=3p-1`**. Thus every slope class contains a line with weight at most two.

Two lines `L,L'` of different slopes intersect in exactly one affine point `P`, and their report Hamming distance is

\[
d_H(c_L,c_{L'})=w(L)+w(L')-2c(P), \tag{3}
\]

where `c(P)` counts the queries at their intersection. Under (1), any two lines of weight at most two from *different* slopes must be disjoint in their query supports. Indeed, if `c(P)>=1` their distance is at most `2+2-2=2`. Lines of the same slope do not intersect. Therefore **each query occurrence belongs to at most one line of weight at most two**. In particular, if `a_j` denotes the number of weight-`j` lines,

\[
a_1+2a_2\le Q. \tag{4}
\]

A weight-zero line is impossible: it has Hamming distance at most two from the guaranteed light line in another slope class.

**Eliminate weight-one lines.** Suppose one line `L` has `w(L)=1`, with its unique query at `P`, so `c(P)=1`. Every line from another slope has weight at least two, because a weight-one line from another slope would have distance at most two from `L`. The unique line of that other slope through `P` must have weight at least four by (3), since `1+w-2>=3`.

Each other slope class therefore has one line of weight at least four and `p-1` others of weight at least two. If at most one of the latter had weight two, the class would have total weight at least

\[
4+2+3(p-2)=3p>Q.
\]

So **each of the other `p-1` slopes has at least two weight-two lines**. Equation (4) then forces

\[
Q\ge2a_2\ge4(p-1)>3p-1=Q
\]

for `p>=5`, a contradiction. All `p²` lines must consequently have weight at least two.

**Upper bound the line-pair moment.** Every slope class contains a weight-two line. Since two such lines of different slopes cannot share a queried point, (4) gives `a_2<=floor(Q/2)`. Write each heavier line's weight as `3+k_L`, with `k_L>=1`, and set `E=sum k_L`. Summing all line weights over the `p` slope classes gives `pQ=3p²-p`. Compared with a baseline weight of three on each of the `p²` lines, this implies

\[
E=a_2-p\le\left\lfloor\frac{3p-1}{2}\right\rfloor-p=\frac{p-1}{2}. \tag{5}
\]

The second factorial moment of the line weights is therefore bounded by

\[
\begin{aligned}
\sum_L \binom{w(L)}2
 &=3p^2-2a_2+\frac12\sum_{k_L>0}(5k_L+k_L^2)\\
 &=3p^2-2p+\frac{E+\sum k_L^2}{2}\\
 &\le 3p^2-2p+\frac{E+E^2}{2}\\
 &\le 3p^2-2p+\frac{p^2-1}{8}. \tag{6}
\end{aligned}
\]

**Lower bound the same moment.** In `Q=3p-1` consecutive time steps, `p-1` residues occur three times each and the final residue occurs twice. Thus the number of *pairs of queries in distinct time residues* is

\[
\sum_{r<s}R_rR_s
=\frac{(3p-1)^2-[9(p-1)+4]}{2}
=\frac{9p^2-15p+6}{2}. \tag{7}
\]

Each such pair determines exactly one nonvertical affine line, since its two time residues are distinct and `p` is prime. Therefore every pair contributes one unit to `sum_L binom(w(L),2)`. Repeated queries at the very same affine point contribute **additional** units to that sum, one in each of the `p` lines through the point. Pairs at the same residue but different markers contribute zero. In particular,

\[
\sum_L\binom{w(L)}2\ge\frac{9p^2-15p+6}{2}. \tag{8}
\]

Subtracting the upper bound (6) from the lower bound (8) gives

\[
\frac{11p^2-44p+25}{8}
=\frac{11(p-2)^2-19}{8}>0\qquad (p\ge5). \tag{9}
\]

The same quantity cannot be simultaneously above and below these limits. Consequently the supposition (1) at `Q=3p-1` is false. OIC-T58 attains `3p`, proving (2). **QED.**

### Exact seven-phase contradiction

For `p=7` and `Q=20`, (5) gives `a₂<=10` and `E<=3`, so (6) gives

\[
\sum_L \binom{w(L)}2 \le 139.
\]

The actual query-residue counts are `(3,3,3,3,3,3,2)`, forcing by (8)

\[
\sum_L \binom{w(L)}2 \ge 171.
\]

This is a purely integer contradiction. No assumption that the schedule omits a sensor marker enters the proof.

## OIC-T65. Cross-residue incidence identity, including repeated observations

For any length-`Q` query word on a prime cycle `p`, let `c_{r,m}` be the number of queries with time residue `r` and address `m`, let `R_r=sum_m c_{r,m}`, and let `w(L)` count queries lying on affine line `L`, with all multiplicities. Then

\[
\boxed{
\sum_{L\text{ nonvertical}}\binom{w(L)}2
=\sum_{r<s}R_rR_s+p\sum_{r,m}\binom{c_{r,m}}2.
} \tag{10}
\]

**Proof.** Expand each `binom(w(L),2)` into unordered query-occurrence pairs on `L`. Distinct-time-residue pairs determine one unique line. A pair with the same residue and the same marker belongs to all `p` nonvertical lines through that point. A pair with the same residue and different markers belongs to none. These cases partition every unordered pair of query occurrences. QED.

Identity (10) supplies an independent integer audit of the incidence counting that underlies OIC-T64, and explicitly handles repeated measurements at the same residue/marker point.

## Scope and comparison

The theorem resolves **fixed**, precommitted sensor schedules on prime cycles `p>=5` with one substituted bit. OIC-28's **adaptive** seven-phase bound `15<=Q*_adaptive(7)<=20` is unchanged. The fixed seven-phase minimum is now **21**, proven analytically, with a three-marker attaining schedule.

Its communication cost depends on how the marker schedule is described: 21 raw responses, up to 63 binary online address bits if all addresses are transmitted, or a preprogrammed three-block instruction charged separately. The observation model is a synthetic deterministic finite-state interface; this theorem implies no result about subjective experience, cosmological structure, or real transformer internal states.

## Machine verification

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic32_global.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic32_global.py' -v
python research/observational_inwardness_v1/oic32_run.py --check
~~~

These checks verify the exact arithmetic, incidence identity with duplicate query locations, analytic inequality, constructive block witnesses, and randomized negative instances. They supplement a **uniform mathematical proof** and do not treat random testing as evidence of an exhaustive negative result.

## OIC-T66. Complete prime-size classification, including the small exceptions

For a fixed one-bit sensor and one adversarial substitution, the smallest primes are exceptional:

\[
\boxed{Q^*_{\rm fixed,flip}(p)=
\begin{cases}
6,&p=2,\\
8,&p=3,\\
3p,&p\ge5\text{ prime}.
\end{cases}}
\]

For `p=2` and `p=3`, these are **finite computer-assisted exhaustive results**. Simultaneously translating every queried marker `m_t` and hidden initial phase `z_0` by a constant `c∈F_p` preserves every report trajectory, for each shift `d`. Hence we may fix the first marker to zero without changing the existence of a successful fixed schedule. At horizon `Q` the resulting search space contains exactly `p^(Q-1)` words.

The independent integer enumerator evaluates every such word at the immediately shorter horizons:

| p | Q | Exhaustively enumerated normalized fixed words | Largest attained cross-shift distance |
|---:|---:|---:|---:|
| 2 | 5 | 16 | 2 |
| 2 | 6 | 32 | 3 |
| 3 | 7 | 729 | 2 |
| 3 | 8 | 2,187 | 3 |

At `p=2,Q=6`, the six-address word `(0,0,0,0,0,0)` attains distance three. At `p=3,Q=8`, the word `(0,0,0,1,1,1,2,2)` attains distance three. Any shorter valid schedule could be extended to the immediately shorter tested horizon without reducing its minimum distance. Complete enumeration at `Q=5` and `Q=7`, respectively, establishes the exact minima.

The general analytical proof OIC-T64 covers every prime `p>=5`. The exceptional enumerations are independently reproducible through `oic32_small.py`, `test_oic32_small.py` and `oic32_small_run.py`, using only Python integers. These cases complete the classification over all primes without extending the uniform second-moment argument to `p=3`, where its strict inequality does not apply.

## OIC-T67. Certified strict benefit of adaptive queries under one substitution

The new unrestricted fixed lower bound combines with previously independently verified adaptive decision trees:

- `p=5`: OIC-T45 gives an adaptive minimum of 14 queries, while OIC-T64 gives a fixed minimum of 15.
- `n=6`: OIC-T47/48 give an adaptive minimum of 15; the previously proved even-cycle fixed lower bound OIC-T49 is at least 18.
- `p=7`: OIC-T51 supplies a verified adaptive winning tree with at most 20 queries; OIC-T64 now proves every fixed schedule needs at least 21.

Thus the adaptive strategy has a **strictly smaller returned-query budget** in each of these synthetic cases with one adversarially inverted report. At seven phases,

\[
Q^*_{\mathrm{adaptive,flip}}(7)\le20<21=Q^*_{\mathrm{fixed,flip}}(7).
\]

This is independent of the previous HiGHS fixed-Q20 infeasibility conclusion. It follows from the new analytic fixed lower bound and the complete OIC-28 winning policy replay. Adaptive strategies still pay live address-command and policy-description costs, so a lower *number of reports* does not imply lower total communication or hardware cost.

## OIC-T68. Global fixed one-erasure optimum for prime cycles

**Theorem.** For every prime `p>=5`, with one **marked erased report** and no substitutions, the minimum number of fixed, precommitted one-bit sensor queries is exactly

\[
\boxed{Q^*_{\mathrm{fixed,one\ erasure}}(p)=2p.}
\]

This lower bound permits **every** one of the p sensor addresses. The two-block word `0^p 1^p` achieves distance two (OIC-T58, k=2), so it suffices to rule out `Q=2p-1`.

**Proof.** As in OIC-T64, represent query occurrences as points `(r,m)` of the affine plane `F_p²`, with each hidden shift/phase a nonvertical line `L` and weight `w(L)` equal to the count of queries on it. Erasure recovery requires minimum cross-shift Hamming distance at least **two**.

At `Q=2p-1`, each slope's p lines have total weight `2p-1`, so at least one line in each slope class has weight at most one. A weight-zero line would have distance at most one from a light line in any other slope, contradicting erasure correction. Hence all lines have weight at least one.

If two weight-one lines of different slopes shared a selected query point, their report distance would be `1+1−2c(P)<=0`, violating distance two. Lines of the same slope have disjoint supports. Thus each query occurrence belongs to at most one weight-one line. If `a_1` is their number,

\[
a_1\le Q=2p-1. \tag{11}
\]

Write the weights of all remaining lines as `2+k_L`, `k_L>=0`; positive k identifies the heavier lines. Across all `p` slope classes, the sum of the `p²` line weights is `pQ=2p²-p`. Comparing with baseline weight two gives

\[
E:=\sum_{k_L>0}k_L=a_1-p\le p-1. \tag{12}
\]

The line-pair moment is therefore bounded above by

\[
\begin{aligned}
\sum_L\binom{w(L)}2
&=p^2-a_1+\frac12\sum_{k_L>0}(3k_L+k_L^2)\\
&=p^2-p+\frac{E+\sum k_L^2}{2}\\
&\le p^2-p+\frac{E+E^2}{2}\\
&\le\frac{3p(p-1)}{2}. \tag{13}
\end{aligned}
\]

The time-residue row counts at `Q=2p-1` are `p−1` rows of size two and one row of size one. The incidence identity OIC-T65 independently forces

\[
\sum_L\binom{w(L)}2 \ge \sum_{r<s}R_rR_s
=2(p-1)^2. \tag{14}
\]

The lower bound (14) exceeds the upper bound (13) by

\[
2(p-1)^2-\frac{3p(p-1)}2
=\frac{(p-1)(p-4)}2>0\qquad(p\ge5). \tag{15}
\]

Therefore no `2p-1` fixed schedule can correct a marked erasure. Any shorter successful schedule could be extended to that length without reducing its minimum code distance. The two-block code attains `Q=2p`, proving the theorem. QED.

## OIC-T69. One-erasure optimum over every prime size

For `p=2` and `p=3`, the normalized finite enumerator verifies the immediately shorter horizons exhaustively: all four length-three schedules on `p=2` have cross-shift distance at most one, and all 81 normalized length-five schedules on `p=3` also have distance at most one. A two-block word attains distance two at `Q=4` for `p=2` and `Q=6` for `p=3`. Combined with OIC-T68:

\[
\boxed{Q^*_{\mathrm{fixed,one\ erasure}}(p)=2p
\quad\text{for every prime }p\ge2.}
\]

The p=2 and p=3 lower bounds are finite exhaustive certificates; the p>=5 lower bound is uniform and analytic. The general statement applies to a known-position erased binary report, leaving the other reported bits intact. It is distinct from the model with a substituted bit, whose exact prime thresholds are classified in OIC-T64/T66.

See `oic32_erasure.py`, `test_oic32_erasure.py`, `oic32_erasure_run.py`, and `results/oic32_erasure.json` for the arithmetic and small-instance checks. All calculations are integer-exact.
