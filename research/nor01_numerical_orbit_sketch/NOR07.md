# NOR-07: certifying a second zero-input attractor in the trained RRC-25 GRU

**Status:** source-locked rational interval certification in progress. The candidate comes from the independently executed [NOR-06 six-family discovery survey](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43695670674), which showed that the squared-family input approached a stationary hidden vector approximately 5.020912 units from NOR-04's already certified fixed point.

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

## Reproduce

Python 3.12 standard library only:

    python -m unittest research.nor01_numerical_orbit_sketch.test_nor07_two_attractors -v
    python research/nor01_numerical_orbit_sketch/nor07_two_attractors.py --check --out /tmp/nor07_two_basin.json

The [NOR-07 workflow](../../.github/workflows/nor07-two-attractors.yml) uploads its complete exact-rational receipt only after the 24-step uniform derivative bound, self-map, one-step-period, and disjointness tests pass. It preserves source weight hashes and sets squared-family input basin entry to false until an independent finite-time trajectory enclosure proves it.

[Companion MathCan NOR-07](https://github.com/patrickdugan/MathCan/pull/56) proves a corresponding multibasin \(L_p\) compression theorem and a lower bound on the distortion of merging both terminal histories into one label. Its Lean component checks finite metric separation and terminal-code arithmetic. The analytic Banach and Gaussian \(L_p\) derivations remain written mathematical arguments.
