# NOR-05: trained GRU one-step fixed point and full-depth convergence

**Executed 15 November 2026.** [Successful hosted NOR-05 proof audit](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43693151218) · [source-locked exact receipt](results/nor05_period_one.json) · [MathCan theorem and proof](https://github.com/patrickdugan/MathCan/blob/research/nor01-orbit-stitching-20261115/research/nor05/PERIOD_ONE_NONLINEAR_COMPACTIFICATION.md).

The original RRC-25 1,000-parameter trained GRU has a real-analytic zero-input transition \(F\) on twelve-dimensional hidden states. NOR-04 already certified that \(G=F^{12}\) is strictly contractive on an explicit closed Euclidean ball \(B(c,10^{-11})\), with \(K=0.928242792263\), center residual \(s=1.186388155\times10^{-15}\), and a unique fixed point \(p\) of \(G\) within the ball.

## Verified period-one exclusion

The new exact-rational one-step certificate proves that \(F(p)\) also belongs to that certified ball. A uniform operator-norm enclosure on the entire ball and a separate one-step residual bound give
\[
L_1\le1.697603419511,\qquad
s_1\le1.81733347\times10^{-16},\qquad
\|p-c\|\le s/(1-K)\le1.65333656704\times10^{-14}.
\]
Thus
\[
\boxed{\|F(p)-c\|\le s_1+L_1\frac{s}{1-K}
\le2.82488314450335\times10^{-14}<10^{-11}.}
\]
Since \(G\circ F=F\circ G\), \(F(p)\) is another fixed point of \(G\) inside the ball. By NOR-04's proven uniqueness, \(F(p)=p\). The point is therefore a **fixed point of the original one-step nonlinear trained GRU**, rather than merely a possible twelve-cycle.

The [NOR-05 Python script](period_one.py) verifies these exact rational inequalities after checking the original RRC-25 checkpoint SHA-256:
54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b.

## Every single recurrence depth

The true GRU map sends its hidden unit cube into itself because each coordinate is a sigmoid-weighted convex combination of the previous hidden coordinate and a tanh candidate. A rational interval enclosure of its Jacobian on the entire invariant cube certifies the conservative bound
\[
L_{\mathrm{cube}}\le\frac{39111216023139}{5\cdot10^{11}}
=78.222432046278.
\]
For every starting state \(x\in B(c,10^{-11})\), and time \(t=12m+j\) with \(0\le j<12\),
\[
\boxed{\|F^t(x)-p\|_2
\le L_{\mathrm{cube}}^j(0.928242792263)^m\|x-p\|_2.}
\]
This is a full-depth nonlinear convergence theorem. The factor \(L_{\mathrm{cube}}^{11}\approx6.71\times10^{20}\) is very conservative and makes the bound weak at moderate depths. Its value is a certified upper envelope, not a measured expansion.

The paired MathCan paper combines this infinite-horizon nonlinear convergence with OpenAI Math #094 finite-set real \(L_p\) dimension reduction. It yields an existential all-time metric embedding for trajectories started inside the ball, with a tail error controlled by \(4R M K^{\lfloor T/12\rfloor}\). Complete Crouzeix remains available for matrix-valued polynomial bounds on the frozen Jacobian; the full nonlinear result follows from the independent invariant-ball and one-step uniqueness proof.

## Reproducibility and epistemic boundary

Run from repository root with Python 3.12 and its standard library:

    python -m unittest discover -s research/nor01_numerical_orbit_sketch -p 'test_period_one.py' -v
    python research/nor01_numerical_orbit_sketch/period_one.py --check --out /tmp/nor05_period_one.json

The [NOR-05 GitHub Actions workflow](../../.github/workflows/nor05-period-one.yml) runs four tests, replays the exact check, and compares the output JSON byte-for-byte in parsed content to the committed receipt. The abstract commutation-and-uniqueness argument is Lean-checked in MathCan/PeriodOne.lean. The general Banach and Gaussian \(L_p\) theorems are written paper-level mathematics.

The certified ball has radius only \(10^{-11}\). Entry into this ball by naturally sourced RRC-25 hidden states has **not** been proved, and the original native proof-ranking performance has not been reevaluated by this certificate. The interval arithmetic analyzes the real analytic map defined by the checkpoint's decimal-rational weights; floating-point runtime behavior is a separate implementation model.
