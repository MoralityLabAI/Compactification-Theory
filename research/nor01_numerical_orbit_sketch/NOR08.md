# NOR-08: expanded certified second-attractor basin in the trained RRC-25 GRU

**Executed November 15, 2026.** The original NOR-07 certificate established two distinct locally attracting period-one fixed points of the real-valued zero-input recurrence of the frozen 1,000-parameter RRC-25 GRU. Its second invariant ball had radius only \(10^{-20}\). This experiment rigorously enlarged that second ball.

[Successful hosted proof sweep](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43699240154) · [exact source-bound receipt](results/nor08_second_basin_summary.json) · [fast rational receipt auditor](nor08_receipt_audit.py) · [original two-attractor theorem](NOR07.md).

## Protocol

The [NOR-08 executable](nor08_expand_ball.py) uses the 55-decimal rational second center from the frozen NOR-07 proof. Its model checkpoint SHA-256 remains

54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b.

It tests a fixed descending set of radii \(10^{-5},3\times10^{-6},10^{-6},3\times10^{-7},10^{-7},10^{-8},10^{-9}\). For every radius \(R\), the source-locked stationary-center Volterra/Hessian verifier computes a rational upper bound \(K_R\) on the operator norm of the derivative of **the actual nonlinear 24-step map** \(G=F^{24}\) everywhere in the entire Euclidean ball \(B(c_2,R)\). It independently bounds the center residual \(s_R=\|G(c_2)-c_2\|\).

All pass/fail decisions are over exact fractions:
\[
K_R<1,\qquad s_R+K_RR<R.
\]
A one-step return bound also verifies that the unique \(G\)-fixed point is a period-one fixed point of the original \(F\). The candidate second ball must remain disjoint from the NOR-04 first fixed-point ball, using exact rational coordinate separation.

## Executed results

| Tested second-ball radius | Certified 24-step derivative upper | All proof gates |
| --- | ---: | :---: |
| \(10^{-5}\) | 1.130996 | Failed sufficient gate |
| \(3\times10^{-6}\) | **0.886155** | Passed |
| \(10^{-6}\) | 0.830309 | Passed |
| \(3\times10^{-7}\) | 0.811964 | Passed |
| \(10^{-7}\) | 0.806830 | Passed |
| \(10^{-8}\) | 0.804535 | Passed |
| \(10^{-9}\) | 0.804306 | Passed |

The largest passing radius on this registered grid is \(R_2=3\times10^{-6}\), a factor of \(3\times10^{14}\) larger than NOR-07's original tiny ball. The exact uniform block contraction bound is
\[
K_2\le\frac{44307764237386472464759585113983}{5\times10^{31}}
=0.8861552847477294\ldots,
\]
and the center residual upper is \(s_2\le37/(5\times10^{30})\).

The strict invariant-ball clearance is
\[
R_2-s_2-K_2R_2=
\frac{17076707287840582605720874658051}{5\times10^{37}}
\approx3.415341457568117\times10^{-7}>0.
\]
The independent one-step return estimate is approximately \(1.54144\times10^{-28}<R_2\). The two fixed-point balls remain separated by more than \(1.8389124\) units using a single coordinate, with full Euclidean center distance approximately 5.020912.

Since the original NOR-07 small ball lies inside this enlarged ball, the uniqueness of the 24-step attracting fixed point on the enlarged ball establishes that it is the **same second fixed point** as before.

## Next experiment: squared-family source capture

The NOR-06 FP64 survey showed that the public squared-family RRC-25 hidden state approaches the second center rather than the first. NOR-06B separately demonstrated that rational recentered shadowing can certify a different source trajectory entering the first attractor after 1,308 zero-input steps.

The new \(3\times10^{-6}\) ball is a far more practical target for an independent **squared-family** rational capture calculation. This requires the original nine public feature rows, exact rational interval propagation and a finite terminal clearance. A numerical first crossing must remain discovery-only until that interval proof passes.

The NOR-08 radius sweep changed no model weights, did no training, and used no hidden source labels to select a candidate radius. Its inputs were the frozen NOR-07 fixed-point center and a predeclared rational radius list.
