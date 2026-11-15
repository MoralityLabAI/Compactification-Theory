# NOR-08: enlarged second attractor basin and source-conditioned capture

**15 November 2026.** Builds on the NOR-04/05 fixed point, NOR-06 signed-family exact capture, NOR-07 two distinct attracting fixed points, and MathCan's merged multibasin code-capacity result.

The original trained RRC-25 GRU has 1,000 parameters and twelve hidden coordinates. The dynamics in this study are the zero-input real-analytic extension of its original trained decimal-rational checkpoint, SHA-256 54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b.

## Executed NOR-08A: larger certified second basin

The [source-locked radius sweep](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43699717963) passed all five regression tests and evaluated ten proposed radii using the exact-rational, global-Hessian, stationary-center Volterra enclosure. The nominal center is the second fixed-point candidate already certified by NOR-07.

| Radius | Upper bound on the true 24-step derivative norm | Certified invariant ball? |
| --- | ---: | :---: |
| \(10^{-20}\) | 0.804280138442 | Yes |
| \(10^{-16}\) | 0.804280138445 | Yes |
| \(10^{-12}\) | 0.804280163878 | Yes |
| \(10^{-10}\) | 0.804282682016 | Yes |
| \(10^{-8}\) | 0.804534553418 | Yes |
| \(10^{-7}\) | 0.806829536493 | Yes |
| \(10^{-6}\) | 0.830309495372 | Yes |
| \(10^{-5}\) | 1.130995802941 | Inconclusive |
| \(10^{-4}\) | \(4.01\times10^{48}\) | Inconclusive |
| \(10^{-3}\) | Arithmetic envelope exceeded the bounded numeric representation | Inconclusive |

The largest radius certified by this preregistered sweep was \(R_2=10^{-6}\), with

\[
K_2\le
\frac{20757737384308237464323670318943}
     {25000000000000000000000000000000}
=0.83030949537232949857294681275772,
\]

and a center residual bounded by

\[
s_2\le\frac{37}{5000000000000000000000000000000}
=7.4\times10^{-30}.
\]

The **exact invariant-ball clearance** is

\[
R_2-K_2R_2-s_2
\ge
\frac{4242262615691762535676144681057}
     {25000000000000000000000000000000000000}
\approx1.69690504628\times10^{-7}>0.
\]

These rational quantities are retained in [results/nor08_expanded_second_ball.json](results/nor08_expanded_second_ball.json). The second attractor is already known to be a fixed point of the original one-step map by the stronger NOR-07 period-one theorem; its enlarged invariant ball is a contracting neighborhood for the 24-step nonlinear map.

A failure of the sufficient bound at larger radii does not establish noncontraction there. The huge \(10^{-4}\) bound and the \(10^{-3}\) arithmetic overflow are explicitly treated as inconclusive certificates.

## Executed FP64 public-input comparison

The same [NOR-08 radius survey](nor08_radius_survey.py) replayed the original squared-family source through the nine public RRC-25 feature rows and then 3,600 zero-input recurrent transitions. The candidate center distance crossed the newly certified \(10^{-6}\) radius at **651 zero-input steps**, while the original tiny \(10^{-20}\) and \(10^{-16}\) radii could not be distinguished by float64 at that horizon.

The first measured crossings, for selected radii, were \(10^{-3}\) at step 329, \(10^{-4}\) at 436, \(10^{-5}\) at 543, \(10^{-6}\) at 651, \(10^{-8}\) at 865 and \(10^{-12}\) at 1293. The crossing times alone are FP64 discovery results.

## NOR-08B: exact rational source-to-second-basin test

The [independent capture script](nor08_squared_capture.py) starts from the exact-rational nine-row public input enclosure, reusing the NOR-06B source-locking and interval propagation. It tracks a rational reference hidden state \(c_k\), a certified Euclidean error radius \(\rho_k\), and a uniform ordered-Jacobian bound on each subsequent twelve-step block.

The proof condition is

\[
\|F^{12k}(h_9)-c_k\|_2\le \rho_k,
\qquad
\|c_k-c_2\|_2+\rho_k\le 10^{-6},
\]

where \(h_9\) is the original squared-family state and \(c_2\) is the exact decimal-rational center from NOR-07. Every inequality deciding capture uses exact fractions and upward-rational square roots. The code searches at most 70 blocks, a bounded 840-step campaign selected before running the interval test.

A successful witness transfers the known second-ball contraction to the **real nonlinear continuation of that specific naturally generated squared-family input** for all later depths, just as NOR-06 did for the signed-family input at the first attractor. A failed interval condition is an inconclusive result, with the original FP64 crossing preserved as separate discovery evidence.

The [NOR-08B workflow](../../.github/workflows/nor08-squared-capture.yml) has separate jobs for five source and interval regression tests and for the full bounded exact capture. Its completed artifact records every block radius, the final positive clearance, source hashes and an exact trace checksum.

## Executed NOR-08B: squared-family natural source captured

The complete [source-locked independent interval run 43700133484](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43700133484) passed both regression and full rational-capture jobs. It proves that the original squared-family RRC-25 input state, after its nine public feature rows and then **55 blocks of twelve zero-input steps**, enters the second NOR-08 invariant ball.

The certified step count is **660** (compared with FP64's first crossing at step 651). At the terminal rational center \(c_{55}\), the exact proof gives

\[
\begin{aligned}
\|c_{55}-c_2\|_2
&\le \frac{2019911314217696479857059}
{2500000000000000000000000000000}
\approx8.079645256871\times10^{-7},\\
\|F^{660}(h_9)-c_{55}\|_2
&\le \frac{11725051219}
{50000000000000000000000000000000}
=2.3450102438\times10^{-22},\\
R_2-\|c_{55}-c_2\|_2-\rho_{55}
&\ge\frac{9601773715646058677807601}
{50000000000000000000000000000000}
\approx1.92035474313\times10^{-7}>0.
\end{aligned}
\]

Here \(R_2=10^{-6}\), \(c_2\) is the exact 55-decimal stationary center from NOR-07, and \(h_9\) is the real-analytic trained hidden state after the original nine public RRC-25 input rows. All acceptance decisions use rational upper bounds. The full certified trajectory trace has SHA-256 bbfb0c62be1c9989b9c100d1540450e776ffb997a158c99f585dc7a0cfeb73cc.

The [frozen dual-source receipt](results/nor08_dual_source_capture.json) binds this second source to the earlier NOR-06B signed-family first-basin entry at step 1,308. [NOR-08E](nor08_dual_capture_audit.py) independently checks both exact fractional clearances and their distinct-ball separation without rerunning the expensive 55- and 109-block intervals.

This gives two distinct **naturally generated source-conditioned** GRU trajectories, each with mathematically certified entry into a different attracting basin. Let \(p_2\) be NOR-07's certified period-one fixed point in the expanded second ball. For any \(m\ge0\) and \(0\le j<24\),

\[
\|F^{660+24m+j}(h_9)-p_2\|_2
\le L_{\rm cube}^{j}
  (0.830309495372330)^m
  \left(\|c_{55}-c_2\|_2+\rho_{55}+\frac{s_2}{1-K_2}\right),
\]

where NOR-05 provides a conservative invariant-cube one-step Lipschitz bound \(L_{\rm cube}\), and \(K_2,s_2\) are the exact NOR-08A block bounds. The fixed point is stationary under the original one-step map, so convergence applies to every subsequent integer depth.

For the two source states at their respective certified capture times, NOR-08D additionally proves a cross-basin Euclidean distance greater than 5.020911335789842. Identical terminal codes would incur at least that additive cross-distance error. The merged MathCan NOR-08 injectivity theorem makes the required one-bit terminal attractor label explicit.

## Implication for compact recurrent representations

MathCan [NOR-08 basin-code capacity theorem](https://github.com/patrickdugan/MathCan/pull/60) proves that if two late hidden streams approach distinct attracting fixed points and their allowed distance distortion is below the separation minus both tail radii, any faithful terminal-history representation must assign distinct codes. Two distinct attractors were certified in NOR-07; the signed-family source has a certified entry to attractor one. NOR-08B has now completed the independent squared-family source-to-second-basin proof at step 660.

Any evaluation of proof-search competence, native model calibration or broader mechanistic semantics requires separate held-out experiments. The geometric certificate concerns a small real-analytic trained GRU under a declared zero-input intervention.


## NOR-08C: held-out family and intervention audit

The [independent 72-case CI run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43700460668) passed four source-lock and public-only tests and evaluated twelve newly seeded test cases per RRC-25 family. Each case was run through its original nine public input rows and then 720 zero-input steps. A mode was assigned from the closest of the two certified fixed-point centers when the candidate was within a preregistered Euclidean distance of 0.05.

| Public synthetic family | First fixed-point mode | Second fixed-point mode |
| --- | ---: | ---: |
| signed | 7 | 5 |
| sparse | 9 | 3 |
| product | 5 | 7 |
| rational | 9 | 3 |
| squared | 3 | 9 |
| late_source | 12 | 0 |

All 72 sample trajectories resolved to a mode by this FP64 diagnostic. The prespecified classifier mapping second mode to the squared family and first mode to all other families achieved 70.8% accuracy, below the 83.3% base rate of always predicting not-squared. The geometry captures an attractor distinction, with substantial within-family variability.

A separate deterministic intervention flipped the sign of hidden coordinate zero directly after the nine original input rows, then repeated the same 720-step zero-input continuation. It switched the attractor-mode label in **2/72 cases**, while 70 retained their original FP64 mode. This is one coordinate intervention, not a global causal analysis of all learned features. No hidden training suffixes or future labels were consulted by the mode assignment.

The [NOR-08C implementation](nor08_basin_partition.py) and [hosted workflow](../../.github/workflows/nor08-basin-partition.yml) retain per-case public-feature fingerprints, family counts, original native-logit parity, both distances and intervention outcomes. Source case identities are synthetic; overlap against the full original training roster has not been exhaustively audited.

## NOR-08D: an exact terminal-code separation obligation

The [cross-basin arithmetic audit](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43700705224) passed three tests using the two certified fixed-point centers, the original NOR-04 first ball of radius \(10^{-11}\), and the enlarged second ball of radius \(10^{-6}\).

It computes an exact-rational lower bound on the 12D Euclidean center distance by integer square root with outward rounding. Consequently **every pair of states in the two certified balls** has separation strictly greater than

\[
\boxed{5.020911335789842.}
\]

Collapsing such a pair to the same terminal code incurs at least this additive distance error. Under any smaller cross-basin error budget, a faithful late-history encoder needs at least two distinct attractor labels, requiring one fixed-width bit.

This is a deterministic geometric lower bound, using [NOR-08D code](nor08_cross_basin.py) and the merged [MathCan basin-label capacity theorem](https://github.com/patrickdugan/MathCan/pull/60). Both the signed-family first-basin entry and the squared-family second-basin entry now have separate successful rational interval proofs, enabling this quantitative two-natural-source comparison. The two-attractor geometry itself was already certified in NOR-07.
