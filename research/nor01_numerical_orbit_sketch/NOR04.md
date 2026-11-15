# NOR-04: source-locked nonlinear fixed-ball certificate

**Executed 15 November 2026** on the original RRC-25 trained GRU. [Exact GitHub Actions verification](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43691939525) succeeded with four tests and an independent execution of the trained-state interval certificate. [Full pure-math theorem and proof](https://github.com/patrickdugan/MathCan/blob/research/nor01-orbit-stitching-20261115/research/nor04/NONLINEAR_INVARIANT_BALL.md).

The [frozen JSON receipt](results/nor04_invariant_ball.json) retains all exact rational quantities, source hashes, the chosen 17-decimal coordinate center, and precise scope fields.

## Validated result

For the *real-analytic zero-input transition* \(F:\mathbb R^{12}\to\mathbb R^{12}\) of the trained RRC-25 1,000-parameter GRU, let \(G=F^{12}\). A numerical search through the frozen checkpoint identified an approximate attractor \(c\). Its twelve coordinates are stored as exact rational decimal literals in the script. The search is independent of the subsequent arithmetic proof.

The exact-rational interval engine certifies for the entire closed Euclidean ball \(B(c,10^{-11})\) that
\[
\sup_{h\in B}\|DG(h)\|_2\le
K=\frac{928242792263}{10^{12}}=0.928242792263<1.
\]
A separate full twelve-step rational interval replay yields
\[
\|G(c)-c\|_2\le
s=\frac{237277631}{2\cdot10^{23}}
=1.186388155\times 10^{-15}.
\]
The exact invariant-ball clearance is
\[
R-s-KR=
\frac{143277137843}{2\cdot10^{23}}
=7.16385689215\times10^{-13}>0,
\quad R=10^{-11}.
\]
Thus \(G\) is a contraction of \(B(c,R)\) into itself. By Banach's theorem, \(G\) has a unique fixed point \(p\) inside that ball, and for every \(x,y\in B(c,R)\), all \(m\ge0\),
\[
\boxed{\|G^m(x)-G^m(y)\|_2\le K^m\|x-y\|_2.}
\]
The fixed point obeys \(\|p-c\|_2\le s/(1-K)\le 1.65334\times10^{-14}\).

This is a **true nonlinear all-depth local certificate** for the specified trained weights, rather than a power of a single frozen Jacobian. It establishes a fixed point of \(F^{12}\); period-one status of \(F\) itself has not been certified. The radius is small, and no unrelated input-conditioned RRC-25 starting point is claimed to enter this certified basin.

## Audit method

The program in [invariant_ball.py](invariant_ball.py) imports the source-locked trained weights with the original bench25.checkpoints() gate and propagates outward-rounded rational boxes through each actual sigmoid/tanh gate. The 32-decimal interval engine employs an adaptive Taylor series for exponential with a rigorous remainder bound. The derivative of \(G\) is enclosed by multiplying the twelve interval Jacobians in the correct temporal order, and its interval-matrix Frobenius norm bounds the operator norm for **every** initial hidden state in \(B(c,R)\). The separate point trace supplies the residual \(s\). All inequalities in the CI clearance gate are checked over Python's fractions.Fraction, with no floating-point threshold decision.

The source-lock checkpoint SHA-256 is:

54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b

The numerical-range disk of the twelve-step derivative is also bounded by \(q\le0.937877390996\) throughout the ball, so the complete Crouzeix theorem can control matrix-valued polynomial filters on each frozen derivative. A uniform nonlinear-block derivative bound supplies the stronger local contraction statement.

## Reproduction

Python 3.12 and standard library only:

    python -m unittest discover -s research/nor01_numerical_orbit_sketch -p 'test_invariant_ball.py' -v
    python research/nor01_numerical_orbit_sketch/invariant_ball.py --check --out /tmp/nor04_invariant_ball.json

The [NOR-04 workflow](../../.github/workflows/nor04-invariant-ball.yml) additionally verifies exact equality of the generated JSON receipt with the committed copy. A change in trained weights, analytic code or arithmetic behavior must therefore update the source-locked record explicitly. The mathematical proof uses standard Banach contraction theory and remains written mathematics, while the arithmetic source code supplies a reproducible exact-rational certificate.

## Open gates

The next useful control is a validated basin-membership certificate for a naturally sourced hidden state, with interval propagation that retains cancellation over long recurrent histories. This would connect the local all-depth theorem to particular RRC-25 input-conditioned continuations. The native suffix-template ranking task has a separate exact-verifier baseline and should be retested directly before any capability claim.

The other theoretical direction is the finite-dimensional \(\ell_p\) embedding from OpenAI Math #094. NOR-04 supplies exponentially convergent *nonlinear* block-state histories, and the MathCan note proves how the NOR-01 finite-history stitching argument gives an all-horizon metric bound from that contraction. The existential embedding is not an efficient learned model.
