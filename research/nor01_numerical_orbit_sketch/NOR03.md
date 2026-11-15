# NOR-03: trained GRU block-Jacobian numerical-range certification

**Execution:** November 15, 2026. [Pure-math theorem and proof](https://github.com/patrickdugan/MathCan/blob/research/nor01-orbit-stitching-20261115/research/nor03/TRAINED_BLOCK_JACOBIAN.md). Extends NOR-01/02 beyond rational Jordan surrogates to a previously trained recurrent checkpoint. The tested zero-input continuation is an intervention, not the RRC-25 native proof-ranking task.

## Source and frozen protocol

The original checkpoint lives at research/rrc_compactification/results/rrc25_recurrent.json. It has 1,000 trained parameters, twelve hidden coordinates, and a sixteen-output head. Expected original SHA-256: 54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b. The existing RRC-25 source checkpoint reader validates the model, training steps, source hashes, and original checkpoint shape before this study runs.

The canonical positive case begins at exact hidden zero, advances through eight zero-input GRU steps, then certifies the true **changing-Jacobian product** of the next twelve nonlinear steps. The negative control computes a distinct eight-step block starting from hidden zero. The direct product of the different Jacobians is necessary for the chain rule.

Six post-training public source-family cases are generated independently using family index seeds compound25.SEED + 903900 + 1009*i. Each selects the first case with a public header and a visible input distinct from the others. Hidden suffix labels and future outputs are excluded. After the source's original nine feature rows, the GRU continues with zero input for warmups 8 and 32. Each anchor is separately audited for a twelve-step block disk.

## Certified arithmetic

Model JSON decimals and public feature float literals are lifted to real rational decimal values. Taylor series with an exact remainder bound enclose exponential, sigmoid, and tanh. Every interval operation rounds outward to denominator \(10^{16}\). Because sigmoid updates are convex combinations with tanh candidates, the hidden cube \([-1,1]^{12}\) is invariant. The program intersects propagated intervals with this cube, sharpening bounds without weakening soundness.

For each block step the **analytic** GRU Jacobian is enclosed on the actual hidden-state interval. The program computes two sufficient matrix disk bounds: one from a rational midpoint product and a telescoping error budget, and the other from direct interval matrix multiplication, which retains some cancellation. For the true twelve-step block derivative \(A\), both methods certify
\[
W_{\mathbb C}(A)\subseteq \overline D(a,b),
\quad a=1/32,\quad
b\ge\|A-aI\|_2.
\]
The smallest verified rational radius is used. The contraction gate checks \(a+b<1\) exactly, and the global Hessian-derived quadratic remainder is calculated independently for perturbations of radius \(10^{-9}\) within the invariant hidden cube.

## Executed baseline and negative result

The first hosted checkpoint audit passed the six trained-GRU mathematical tests. Its **coarse telescoping** method certified \(a+b=0.942200589137\) at the positive anchor, while the initial-zero eight-step control had \(a+b=27.669071261948\). The analytic one-step Hessian bound was 523.4880814132176, and the resulting twelve-step nonlinear remainder budget at radius \(10^{-9}\) was \(6.764913275852784\times10^{-11}\).

The original retrospective six-case roster subset produced 0/6 contracting disks under the coarse telescoping certificate. Hidden interval widths were small, and the telescoping bound often dominated the result. The direct interval-product method and later-warmup sweep constitute the follow-up campaign. A failed sufficient disk test does not prove instability.

## Reproduce

Python 3.12, standard library only:

    python -m unittest discover -s research/nor01_numerical_orbit_sketch -p 'test_trained_gru_cert.py' -v
    python research/nor01_numerical_orbit_sketch/trained_gru_cert.py --check --out /tmp/nor03_8.json
    python research/nor01_numerical_orbit_sketch/trained_gru_cert.py --check --warmup 32 --out /tmp/nor03_32.json

The [NOR-03 GitHub Actions workflow](../../.github/workflows/nor03-trained-gru.yml) runs both tests and uploads source-bound JSON receipts with every successful/failed test-family certificate.

## Boundaries and next gate

The disk controls powers and matrix-valued polynomials of **the frozen derivative of one trained nonlinear block**. Full nonlinear repeated-block convergence requires uniform derivative and invariant-tube control. This intervention makes no native RRC-25 ranking-performance claim. A tested Gaussian L4 sketch also remains distinct from the general existential dimension bound of OpenAI Math #094.
