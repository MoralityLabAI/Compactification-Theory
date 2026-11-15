# NOR-03: completed source-locked trained GRU audit

**Executed 15 November 2026.** Exact rational interval certification of the changing Jacobians of a real trained 1,000-parameter RRC-25 GRU, with an independently sourced GitHub Actions receipt. The complete archived run is [Actions #43691049653](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43691049653), commit `c6940b4123ec71ab313366b3c51a96b872fbacf8`, artifact `nor03-train-locked-rational-interval-receipt`. Its receipts are named `nor03_trained_gru.json` and `nor03_warm32.json`.

## Certified derivative disks

The real analytic model lifts the checkpoint's JSON decimal weights to exact rationals. The source checkpoint SHA-256 is `54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b`. After the original public nine-feature input sequence and a further eight zero-input GRU steps, the calculation uses interval enclosures of the actual nonlinear chain-rule derivative of the next twelve recurrence steps. The disk center is exactly 1/32. Its radius and q=a+b are computed by rational outward interval arithmetic.

| Input family | Certified q=a+b, warmup 8 | Contracting disk? | Coarse telescoping q |
|:---|---:|:---:|---:|
| Signed | 0.950758814230 | Yes | 0.963965169286 |
| Sparse | 0.960808756508 | Yes | 0.975354390751 |
| Product | 0.731708641506 | Yes | 0.948387259243 |
| Rational | 0.576830356675 | Yes | 0.581178168023 |
| Squared | 1.384527970358 | Uncertified | 2.036290679778 |
| Late-onset | 0.960411442890 | Yes | 0.969690304159 |

All six source-family states use public inputs from a deterministic, post-training generator cohort. The native final model logits agree with independent portable replay to its specified tolerance; no suffix-class labels or future oracle outcomes enter the derivative computation. This cohort was not exhaustively checked for overlap with the original model's 720 training examples. The source's frozen trained weights are unchanged.

For the positive reference from hidden zero, eight zero-input steps followed by one twelve-step nonlinear block give q=**0.937535987328**. The initial-zero eight-step negative control has q=**27.669049913028**. The positive block has a global Hessian upper bound 523.4880814132176 and a Taylor remainder upper bound **6.764903298421893e-11** for input perturbations of Euclidean radius 1e-9 within the invariant cube.

## Longer-depth failure mode

At warmup 32, the same six-family experiment yielded 0/6 contracting disks. In the order listed above, the computed sufficient disk bounds q were 2205.044748, 3572.211481, 1568264898.186089, 8.814303, 1422.853818 and 452.378240. The corresponding hidden-state interval boxes widened substantially through repeated recurrence; direct interval multiplication reduced wrapping relative to a coarse telescoping estimate yet was insufficient to certify these cases. Failure of a sufficient enclosure does not establish a dynamical instability.

## Provenance, limits and research consequences

Both mathematical paths are recorded. The generic product/tail and quadratic shadowing theorem is in [MathCan NOR-03](https://github.com/patrickdugan/MathCan/blob/research/nor01-orbit-stitching-20261115/research/nor03/TRAINED_BLOCK_JACOBIAN.md); the implementation is in [trained_gru_cert.py](trained_gru_cert.py). Exact rational arithmetic certifies pointwise block derivatives of the decimal-lifted source model, using the complex numerical range required by Crouzeix. A bound on A=D(F^12)(h) controls powers and operator-coefficient polynomials of that frozen derivative. Distinct nonlinear blocks have changing derivatives and need a uniform tube/invariant-set argument for global transfer.

The paper #094 Lp embedding is existential. No general constructive subpolynomial embedding algorithm or native reasoning improvement is demonstrated. The 32-step failures identify interval dependency growth as the next numerical-analysis bottleneck. A recentered affine-arithmetic propagation or an interval Newton fixed-point enclosure would be useful independent next gates.

Machine-readable summary with the exact archived run reference: [results/nor03_trained_summary.json](results/nor03_trained_summary.json). Original archived JSON artifacts remain the controlling record.
