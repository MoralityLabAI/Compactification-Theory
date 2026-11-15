# NOR-07: certifying a second zero-input attractor in the trained RRC-25 GRU

**Status:** **source-locked rational interval certification succeeded**, 12/12 tests passed in [GitHub Actions run 43698679035](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43698679035). The candidate comes from the independently executed [NOR-06 six-family discovery survey](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43695670674), which showed that the squared-family input approached a stationary hidden vector approximately 5.020912 units from NOR-04's already certified fixed point.

## Exact model and separation question

The original trained recurrent model has 1,000 checkpoint parameters and 12 hidden coordinates. The checkpoint SHA-256 is

54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b.

NOR-04/05 proved a unique locally attracting *period-one* fixed point \(p_1\) of the real-analytic zero-input GRU in a certified radius-\(10^{-11}\) Euclidean ball centered at \(c_1\), with a 12-step block derivative norm at most \(K_1=0.928242792263\).

The NOR-06 squared-family source trajectory numerically settled near
\[
c_2\approx(0.14515780418,-0.89936969686,-0.99235391608,-0.99713057751,\ldots),
\]
with its terminal FP64 one-step residual zero at machine precision. The candidate centers satisfy \(\|c_1-c_2\|\approx5.0209123358\). This numerical candidate supplies an input to an *independent* rational validation and does not establish an attractor by itself.

## Validation strategy

The [NOR-07 executable](nor07_two_attractors.py) first Newton-refines the squared-family candidate using Python Decimal at 80-digit working precision. The result is converted to exact decimal-rational coordinates with 55 decimal places. This is candidate selection only.

At the rational candidate \(c_2\), it attempts to prove using interval arithmetic that a Euclidean ball \(B(c_2,R_2)\), with \(R_2=10^{-20}\), is invariant and uniformly contracting for the **actual nonlinear 24-step map** \(G_2=F^{24}\). The latest implementation uses a [stationary-center Volterra proof](nor07_stationary_tube.py), which bounds the real one-step Jacobian error and global Hessian remainder while retaining the exact rational powers of a single nominal Jacobian. These powers capture nonnormal cancellations between steps. The general [recentered nonlinear Volterra method](nor07_recentered_tube.py) is preserved separately and formalized at the semiring identity level in [MathCan NOR-07T](https://github.com/patrickdugan/MathCan/pull/58). For a uniform norm upper bound \(K_2\) on \(DG_2\) throughout the entire ball and a validated center residual \(s_2\), the required exact rational tests are
\[
\boxed{K_2<1,\qquad s_2+K_2R_2<R_2.}
\]
Banach's theorem then establishes a unique attracting fixed point \(p_2\) of \(F^{24}\) in this ball.

A second one-step interval Jacobian bound \(L_{1,2}\) and residual \(s_{1,2}\) must satisfy
\[
s_{1,2}+L_{1,2}\frac{s_2}{1-K_2}<R_2.
\]
This ensures \(F(p_2)\) lies in the same invariant ball, and uniqueness plus commutation \(F^{24}F=FF^{24}\) yields \(F(p_2)=p_2\).

Finally the proof checks an exact rational coordinate separation \(|c_{2,1}-c_{1,1}|>R_1+R_2\). Thus the fixed points lie in disjoint balls and cannot coincide. The resulting model would have **at least two distinct locally attracting fixed points** with independently certified nonlinear convergence neighborhoods.

## Executed two-attractor result

The independent [NOR-07 CI run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43698679035) completed all twelve tests on the original checkpoint. The 80-decimal Newton refinement of the squared-family candidate reached a maximum residual of \(5\times10^{-80}\) before the center was frozen to 55 decimal places and independently certified by rational Taylor intervals.

The [stationary-center Volterra implementation](nor07_stationary_tube.py) then proved that the *actual* nonlinear 24-step zero-input GRU map is uniformly contracting on a radius-\(10^{-20}\) ball around the second candidate. The exact certified quantities are:

| Obligation | Certified upper or lower bound |
| --- | ---: |
| Uniform \(\|D(F^{24})\|\) upper | \(0.80428013844207623058831195827191\) |
| Center 24-step residual upper | \(7.4\times10^{-30}\) |
| Invariant-ball clearance lower | \(1.9571986081792377\times10^{-21}\) |
| One-step operator norm upper on ball | \(2.366151213621491\) |
| One-step center residual upper | \(2.4\times10^{-31}\) |
| One-step image bound for fixed point | \(8.970214677153268\times10^{-29}\) |
| Guaranteed inter-basin separation using one coordinate | \(1.8389154011509952\) |

The one-step return test places the second 24-step fixed point strictly within its own ball after a single original GRU step. Commutation with \(F^{24}\) and Banach uniqueness prove it is **a period-one fixed point of \(F\)**. Since the first NOR-04/05 fixed point lies in a disjoint certified ball, the real-analytic trained model has at least two distinct attracting fixed points.

The [committed rational summary](results/nor07_two_attractors_summary.json) includes the 55-decimal center, all exact fractions, the original checkpoint and Actions artifact identifiers, and an explicit scope flag that the squared-family source trajectory has *not yet been interval-certified to enter the second ball*. [NOR-07C receipt audit](nor07_receipt_audit.py) rechecks the rational inequalities against the NOR-04/05 receipts independently of the long proof run.

The previous 24-step axis-aligned interval propagation was computationally impractical due to enclosure wrapping and large rational denominators. The successful stationary-center method uses the cancellation-preserving rational powers of the nominal Jacobian and a global Hessian-controlled nonlinear remainder, rounding every scalar recurrence **upward** to a fixed 32-decimal rational denominator. The full noncommutative theory is in [MathCan NOR-07T](https://github.com/patrickdugan/MathCan/pull/58), now merged.

## Reproduce

Python 3.12 standard library only:

    python -m unittest research.nor01_numerical_orbit_sketch.test_nor07_two_attractors -v
    python research/nor01_numerical_orbit_sketch/nor07_two_attractors.py --check --out /tmp/nor07_two_basin.json

The [NOR-07 workflow](../../.github/workflows/nor07-two-attractors.yml) uploads its complete exact-rational receipt only after the 24-step uniform derivative bound, self-map, one-step-period, and disjointness tests pass. It preserves source weight hashes and sets squared-family input basin entry to false until an independent finite-time trajectory enclosure proves it.

[Companion MathCan NOR-07](https://github.com/patrickdugan/MathCan/pull/56) proves a corresponding multibasin \(L_p\) compression theorem and a lower bound on the distortion of merging both terminal histories into one label. Its Lean component checks finite metric separation and terminal-code arithmetic. The analytic Banach and Gaussian \(L_p\) derivations remain written mathematical arguments.
