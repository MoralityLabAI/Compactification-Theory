# OIC-30: Prime-cycle block codes and sharp resource-constrained fixed sensing

**November 11, 2026.** Evidence: elementary proofs, a deterministic standard-library Python reference, exhaustive two-address count enumeration, all 1,029 cross-shift codeword pairs, and every zero/one-substitution report for the seven-state witness. These results concern a finite synthetic sensor problem. There is no model-training or physical/cosmological inference.

## Sensor model

A hidden world is a constant modular shift `d` and an initial phase `z₀` on `Z_n`; at step `t` its phase is `z₀ + td (mod n)`. A fixed query schedule `m₀,...,m_{Q−1}` receives the binary report

\[
y_t(d,z_0)=[z_0+t d\equiv m_t\pmod n].
\]

The observer must identify `d` for every `z₀`, and may face up to `e` adversarial substitutions among these reports. It can decode the correct shift exactly when the minimum Hamming distance **between codewords belonging to different shifts** is at least `2e+1` (OIC-T55). The initial phase itself may remain ambiguous.

## OIC-T58. Prime-cycle parallel-block construction

Let `p ≥ 5` be prime, and choose `k` distinct sensor labels `b₀,...,b_{k−1}` with `2 ≤ k < p`. Use the fixed schedule

\[
\underbrace{b_0,\ldots,b_0}_{p\text{ queries}},
\underbrace{b_1,\ldots,b_1}_{p\text{ queries}},\ldots,
\underbrace{b_{k-1},\ldots,b_{k-1}}_{p\text{ queries}}.
\]

Then **the minimum cross-shift Hamming distance is exactly `k`**. In particular, it corrects `e` substitutions whenever `k ≥ 2e+1`; the schedule uses `kp` readings from `k` distinct labeled sensor positions.

*Proof.* For shift `d=0`, a phase `z` outside the selected marker set yields the zero word, and a phase at marker `b_j` yields exactly `p` ones in block `j`. For each nonzero `d`, primality makes `r↦z+rd` a permutation of all phases inside every block, so each such trajectory yields exactly one positive report per block, hence `k` positive reports overall.

Two different nonzero shifts `d,d'` can produce simultaneous positive reports only when their phases agree:

\[
 z+rd\equiv z'+rd'\pmod p.
\]

Since `d−d'` is invertible, this equation has exactly one residue `r`. At the corresponding query offset within each block, their common phase has one fixed value, while the block addresses `b_j` are distinct. Therefore the two positive supports intersect in **at most one** time step. Their Hamming distance is at least `2k−2`.

For the stationary phase `z=b_j`, its `p` positive reports and a rotating codeword have exactly one common positive report in block `j`, yielding distance `p+k−2`. For an unselected stationary phase, the all-zero word has distance `k` to every rotating codeword. Since `k≥2`, `2k−2≥k`, and since `p≥2`, `p+k−2≥k`. An unselected stationary phase exists because `k<p`; hence the minimum is precisely `k`. QED.

For `p=3` with `k=p=3`, the same construction has minimum cross-shift distance 4 and also corrects one substitution. The `k<p` formula above is stated for the parameter range in which the zero-word witness exists.

## OIC-T59. Two labeled sensors cannot correct one flip in 3p fixed queries

Let `p≥3` be an odd prime. For a fixed word of length `3p` that queries at most **two** distinct sensor addresses, any two different labels may be normalized to `0` and `1` by an invertible affine relabeling of the phase space. This relabeling permutes the shifts and preserves all report strings.

For each residue `r∈Z_p`, the query instants `r,r+p,r+2p` contribute three reports. Let `c_r∈{0,1,2,3}` of those report address 0; the others query address 1.

The stationary world `(d=0,z₀=2)` returns an all-zero word. For any distinct residues `r,s`, choose a nonzero shift `d=(s−r)^{-1}` and starting phase `z₀=−rd`. This world occupies address 0 at time residue `r` and address 1 at residue `s`, and visits each address exactly once per `p`-step period. Its report Hamming weight is

\[
 w= c_r+(3-c_s)=3+c_r-c_s.
\]

If the sensor code corrects one flip, every rotating word must have distance at least three from the stationary zero word. It follows that `c_r≥c_s` for every ordered pair of distinct residues. Therefore all `c_r` equal a constant `c`.

For this constant-count schedule, *every* nonzero-shift word has exactly three positive reports. At every residue `r`, one of the two marker positions appears at least twice among the three queries. Choose distinct nonzero shifts `d,d'` and initial phases such that both rotating worlds occupy that same repeated marker at residue `r`. Their positive supports then intersect in at least two positions, so the Hamming distance between two weight-three codewords is at most

\[
 3+3-2(2)=2.
\]

This contradicts one-flip correction. Thus two labels are insufficient for **every** fixed length-`3p` schedule. QED.

The proof covers a schedule with a single distinct label as a special case. For `p=7`, all `4^7=16,384` normalized two-marker count vectors are additionally checked by an independent executable witness constructor. No search heuristics are used in that check.

## OIC-T60. A general query lower bound when some sensor addresses are omitted

Let `n≥2` be any integer, and consider a fixed schedule of `Q` one-bit marker queries that uses fewer than `n` distinct sensor positions. For `e≥0` adversarial substitutions, guaranteed identification of every constant shift requires

\[
\boxed{Q\ge n(2e+1).}
\]

*Proof.* Select an unqueried phase `z_*`. The stationary world `(0,z_*)` emits the all-zero report. For shift `d=1`, every query `(t,m_t)` yields a positive report for exactly one initial phase `z₀=m_t−t (mod n)`. Summing the number of positive reports over all `n` initial phases gives exactly `Q`. Therefore some rotating initial phase emits at most `⌊Q/n⌋` positive reports. If `Q<n(2e+1)`, its report has distance at most `2e` from the stationary all-zero word, so one error-correcting decoder cannot distinguish the shifts. QED.

For `n=7`, `e=1` and any proper subset of the seven addresses, the minimum possible fixed-schedule query horizon is at least **21**. OIC-T58 attains that bound with three marker positions. By OIC-T59, three is also the minimum number of marker positions for a 21-query schedule.

**Important scope:** This conditional lower bound does not exclude a 20-query schedule that is permitted to address **all seven** marker positions. The earlier reported Q20 HiGHS integer infeasibility has no independently checkable UNSAT certificate. An independently derived count-per-residue MILP has also remained inconclusive at Q20 under short runtime limits. The globally shortest fixed-schedule horizon is therefore still unresolved.

## OIC-T61. Exact phase-residue occupancy compression

For a fixed schedule of `Q` queries on `Z_n`, let `c_{r,m}` count queries at address `m` whose time satisfies `t≡r (mod n)`. Each phase is periodic modulo `n`, so for two candidate worlds `(d,z)` and `(d',z')`, let

\[
 a_r=z+rd\pmod n,\quad b_r=z'+rd'\pmod n.
\]

Their report Hamming distance is exactly

\[
\boxed{
 d_H=\sum_{r=0}^{n-1}[a_r\ne b_r]
 \big(c_{r,a_r}+c_{r,b_r}\big).
}
\]

*Proof.* The two clean report bits differ precisely when the queried address equals one of the two different phase positions. Both phase positions are unchanged when a query time is increased by a multiple of `n`. Summing the corresponding query counts proves the formula. QED.

For `n=7,Q=21`, the occupancy matrix has just 49 integer variables with seven row-sum constraints, `Σ_m c_{r,m}=3`. Every valid occupancy matrix expands to a fixed sensor word, so the integer representation is exact. A SciPy/HiGHS search on this formulation found a 21-query witness; the final three-marker **block schedule** was independently derived and checked using only Python integers.

## Seven-phase exact certificate and communication

The minimal-three-marker 21-query word is:

~~~text
0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2
~~~

The independent checker examines all `C(7,2)·7²=1,029` cross-shift world pairs. Its distance histogram is `{3:168,4:315,6:420,8:126}`. It also enumerates `49·(21+1)=1,078` zero-or-one-flip world/report cases, obtaining 1,012 distinct received words with no cross-shift collision.

A direct online address command requires three bits per query, yielding `21+63=84` online link bits. A literal offline arrangement uses `3·3=9` bits to identify the markers, `21·2=42` bits to describe the schedule by its three-label indices, and 21 response bits for the first episode, totaling **72 bits**. A much shorter generator program could describe the three blocks, but its encoding must be charged separately; the 72-bit figure uses the explicit literal convention of OIC-29.

The 21-query construction improves the OIC-29 three-marker 22-query schedule, whose corresponding literal first-episode budget was 75 bits. The two-marker 27-query schedule from OIC-29 remains an alternate communication/resource point at 60 first-episode bits.

## Sources and evidence level

The result is a finite construction and an elementary application of affine-plane incidence geometry. It does not establish previously unknown general coding theory. A distinct, run-time-limited search for 20 queries using the full marker set remains inconclusive, and the adaptive seven-phase interval `15–20` from OIC-28 is unchanged.

Reproduce with:

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic30_prime_blocks.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic30_prime_blocks.py' -v
python research/observational_inwardness_v1/oic30_run.py --check
~~~
