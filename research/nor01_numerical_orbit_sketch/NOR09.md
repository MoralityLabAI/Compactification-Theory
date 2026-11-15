# NOR-09: certified dual-source, two-anchor recurrent history compression

**Date:** November 15, 2026. This is the executable quantitative continuation of the merged NOR-06–08 source, invariant-ball and multibasin results. The interval and Banach premises are already separately verified by source-locked CI; this campaign combines their **exact rational receipts** into a cross-source infinite-horizon compression certificate for 24-step checkpoint histories.

## Two genuine capture witnesses

The same frozen trained RRC-25 GRU with twelve hidden coordinates and 1,000 checkpoint parameters supplies two naturally generated synthetic public-input states. The trained checkpoint SHA-256 is:

54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b.

- **Signed input:** [NOR-06B run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43695479529) certifies entry into the radius-\(10^{-11}\) ball around first fixed-point center \(c_1\) after 1,308 additional zero-input steps. NOR-04 certifies a uniform 12-step nonlinear contraction factor \(K_1=0.928242792263\).
- **Squared input:** [NOR-08B run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43700133484) certifies entry into the radius-\(10^{-6}\) ball around second fixed-point center \(c_2\) after 660 additional zero-input steps. The separately verified expanded basin has a 24-step contraction factor \(K_2=0.8303094953723295\).

Both attracting points were proven to be **fixed by the original one-step map**, using distinct invariant-ball and one-step-return certificates. The [NOR-08D cross-basin audit](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43700705224) proves that any point in the two different certified balls is more than 5.020911335789842 Euclidean units from any point in the other ball.

The data are interpreted as the real-analytic decimal-rational extension of the unmodified original trained GRU under a declared zero-input continuation.

## Exact asynchronous 24-step clocks

Start each trajectory at **its own** verified capture time. Index the later states by an extra integer \(m\) of 24-step blocks:

\[
z_1(m)=F^{1308+24m}(h_1),\qquad
z_2(m)=F^{660+24m}(h_2).
\]

The first basin contracts over two verified 12-step blocks per epoch, giving \(\rho_1=K_1^2\approx0.861634681388211\); the second contracts in one verified 24-step block, \(\rho_2=K_2\approx0.830309495372330\).

For the **known rational centers** \(c_i\), the source-lock and Banach receipts imply

\[
\|z_i(m)-c_i\|
\le
\left(D_i+\epsilon_i+\frac{s_i}{1-K_i}\right)\rho_i^m
+\frac{s_i}{1-K_i}
=:B_i(m),
\]

where \(D_i\) is the captured state-to-center distance upper bound, \(\epsilon_i\) is its independently propagated state-error upper bound, and \(s_i\) is the certified center residual for its respective nonlinear block.

The program in [nor09_dual_history.py](nor09_dual_history.py) uses exact Fraction multiplication for these bounds and prints **outward-rounded** rational upper limits at checkpoints \(m=0,8,16,32,64,96,128,256\). These are genuine nonlinear recurrence bounds rather than sampled Jacobian extrapolations.

## Two-anchor history theorem

At a cutoff \(M\), retain the first \(M\) 24-step states of each captured trajectory plus the two rational anchors \(c_1,c_2\). Encode the pooled finite set with any distance map having maximum additive finite-pool error \(E_M\). All later states of source \(i\) map to its **own** anchor image.

The [MathCan NOR-09 theorem](https://github.com/patrickdugan/MathCan/tree/research/nor09-asynchronous-two-basin-history-20261115/research/nor09) proves the all-horizon pairwise error bound

\[
\boxed{E_M+2\max\{B_1(M),B_2(M)\}}
\]

for every pair of post-capture **24-step sampled** states, including cross-basin pairs and unequal source capture times. Its Init-only Lean module checks the finite metric replacement and late-clock indexing.

With a Euclidean identity representation of the finite observed prefix, \(E_M=0\), giving certified additive tail errors:

| Retained extra blocks per stream | Error bound for all clipped pair distances |
| ---: | ---: |
| \(0\) | \(1.6160\times10^{-6}\) |
| \(32\) | \(4.2083\times10^{-9}\) |
| \(64\) | \(1.0960\times10^{-11}\) |
| \(96\) | \(3.3079\times10^{-14}\) |
| \(128\) | \(3.3067\times10^{-14}\) |

The bound approaches a floor from the finite rational-center approximation of the first fixed point, approximately \(3.30667\times10^{-14}\) at late times. This certified floor is not a lower limit on achievable representation error.

OpenAI Math #094 (*Subpolynomial dimension reduction in Lp*, September 23, 2026) supplies a separate existential finite set embedding into \(\ell_p^d\) for \(2M+2\) anchor-and-prefix points:
\[
d\le\exp(C_{p,D}(\log(2M+2))^{\gamma(p)}).
\]
The existence result does not automatically yield an efficient low-dimensional mapping or the actual finite-pool error \(E_M\). For any concrete embedding such as a Gaussian \(L_4\) sketch, that error must be measured and checked independently.

## A one-bit obstruction to collapsing both modes

At every extra 24-step checkpoint, the captured source states remain in their respective invariant balls. Their true Euclidean cross-distance is **always greater than 5.020911335789842**. Mapping both terminal states to one identical code would assign them zero encoded separation and therefore incur an additive error exceeding that lower bound.

A low-distortion history compactifier on these two captured sources must retain at least **two terminal codes**, corresponding to one binary basin label. The general finite-metric injectivity theorem was merged into [MathCan NOR-08 PR #60](https://github.com/patrickdugan/MathCan/pull/60).

This identifies a representation constraint in recurrent hidden-state dynamics. It does not establish that a basin label encodes mathematical correctness, human-interpretable semantic content, or general safety-relevant behavior. The six-family 72-case probe showed overlap between source families and attractor modes.

## Reproduction and next experiment

Run from repository root with Python 3.12 standard library:

    python -m unittest research.nor01_numerical_orbit_sketch.test_nor09_dual_history -v
    python research/nor01_numerical_orbit_sketch/nor09_dual_history.py --check --out /tmp/nor09_dual_history.json

The [NOR-09 hosted workflow](../../.github/workflows/nor09-dual-history.yml) checks source hashes, exact capture clocks, contraction fractions, and the all-horizon clipping budget before publishing the receipt.

Useful next step: apply an explicit, frozen \(L_4\) sketch to the pooled prefix states from both captured source histories, measure \(E_M\) on held-out time pairs, and compare its attainable error and binary label fidelity against a single-anchor compressor. The analytic certification of true states and a trained end-to-end reasoner performance gain remain separate tasks.
