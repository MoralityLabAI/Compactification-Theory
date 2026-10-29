# WS-07: A self-gravitating charged bulk and the static two-brane obstruction

**Research prototype, 30 October 2026.** This is a stacked continuation of [WS-06 PR #13](https://github.com/MoralityLabAI/Compactification-Theory/pull/13). The new package solves the **five-dimensional Einstein equations coupled to a Goldberger–Wise stabilizing scalar and a globally charged complex scalar**, with explicit Israel stress reconstructions and field Robin conditions at **both** branes. We quantify a new ansatz-specific staticity obstruction and compute the **decoupled, gauge-invariant transverse-traceless gravitational sector** at nonzero three-momentum.

**Finding:** A static two-brane compactification with a canonical, stationary, finite-density bulk charged condensate, three-dimensional translation/rotation invariance and both branes satisfying the null energy condition (NEC) is impossible in this ansatz. An exact bulk solution exists if the IR brane instead supplies negative `(rho+p)`, here explicitly realized by a **wrong-sign brane scalar**. This is a diagnostic counterexample, **not** a viable model of matter or dark energy. The gravitational-wave speed also differs dramatically from its observational bound. The scalar/radion/phase coupled *physical* spectrum on an evolving charged two-brane background remains open.

## Reproduce

Python 3.12+ and CPU:

```bash
cd research/warped_superfluid_v7
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python run_experiment.py
```

This runs **26 regression tests**, writes `results/summary.json`, and generates `results/static_background.png`, `results/charge_sweep.png`, and `results/tensor_modes.png`. Set `OPENBLAS_NUM_THREADS=1` for predictable small dense generalized eigensolves. The experiment needs no earlier WS package at runtime and changes no existing research modules.

## Bulk action, metric and exact equations

Let `M3=M5^3`. Work on an orbifold interval `0<=y<=L` with `Z2` jumps. Take the five-dimensional bulk action

\[
S=\int d^5x\sqrt{-g}\left[\frac{M_3}{2}R-\frac12(\partial\chi)^2-\frac12|\partial\Psi|^2-V(\chi,|\Psi|)\right],
\]

with `Psi=F(y) exp(i omega*t)`, `chi=chi(y)`, and

\[
V=\frac12u^2\chi^2-\frac{2}{3M_3}W(\chi)^2+\frac{m^2F^2}{2}+\frac{\lambda F^4}{4},\qquad
W(\chi)=3M_3k+\frac{u\chi^2}{2}.
\]

The **anisotropic** static metric is

\[
ds^2=-e^{2N(y)}dt^2+e^{2A(y)}d\mathbf x^2+dy^2.
\]

It cannot maintain `N=A` when `omega*F` is nonzero. Write `R_y=chi'^2+F'^2`, `Q=omega^2 exp(-2N) F^2`. Then

\[
T^t{}_t=-\tfrac12(R_y+Q)-V,\quad
T^i{}_i=-\tfrac12R_y+\tfrac12 Q-V,\quad
T^y{}_y=\tfrac12(R_y+Q)-V.
\]

The Einstein tensor components are

\[
G^t{}_t=3A''+6A'^2,\qquad
G^i{}_i=N''+2A''+N'^2+3A'^2+2N'A',\qquad
G^y{}_y=3A'(A'+N').
\]

The solver evolves the Einstein `tt` equation, `ii-tt` difference, and both scalar equations while independently monitoring the `yy` Hamiltonian constraint:

\[
3M_3 A'(A'+N')=\frac12(R_y+Q)-V.
\]

It also checks all three Einstein components against the stress tensor independently. Scalars obey

\[
\chi''+(N'+3A')\chi'=V_{,\chi},\qquad
F''+(N'+3A')F'+\omega^2e^{-2N}F=V_{,F}.
\]

For zero charge `F=0` the numerical solution recovers the exact backreacted Goldberger–Wise wall `A=N=-ky-chi_0²(exp(2uy)-1)/(12M3)`, `chi=chi0 exp(uy)`.

## Exact stationary-brane sum rule and no-go statement

Define `D=N'-A'`. Subtracting the mixed `tt` Einstein equation from `ii` gives the **pointwise identity**

\[
\boxed{(e^{N+3A}D)'=\frac{\omega^2}{M_3}e^{-N+3A}F^2.}
\]

For Z2 Israel conditions, write the UV and IR surface stresses as `(rho_0,p_0)` and `(rho_L,p_L)` respectively. Then

\[
D(0)=\frac{\rho_0+p_0}{2M_3},\qquad
D(L)=-\frac{\rho_L+p_L}{2M_3}.
\]

Integrating yields

\[
\boxed{e^{N_0+3A_0}(\rho_0+p_0)
+e^{N_L+3A_L}(\rho_L+p_L)
+2\omega^2\int_0^L e^{-N+3A}F^2dy=0.}
\]

For positive `omega` and nonzero real amplitude `F`, the last integral is strictly positive. Both NEC inequalities `rho_i+p_i>=0` therefore cannot hold simultaneously. This **does not prohibit** time-dependent compactifications, moving branes, noncanonical kinetic terms, different topologies, or sources that violate the theorem's assumptions. It is related in spirit to the Gibbons–Kallosh–Linde braneworld consistency constraints and to the static matter restrictions discussed by Enqvist, Keski-Vakkuri and Räsänen; the displayed identity is explicitly derived here for the stated canonical charged scalar.

## A self-consistent but exotic stationary solution

Fix `M3=k=L=1`, `u=.25`, `chi(0)=.4`, `F(0)=.08`, `m²=.5`, `lambda=.3` and both charged-field Robin stiffnesses `kap_uv=kap_ir=1`:

\[
F'(0)=\frac12F(0),\quad F'(L)=-\frac12F(L),\quad
\chi'(0)=u\chi(0).
\]

The solver chooses UV `A(0)=N(0)=0` and `A'(0)=N'(0)<0` from the exact Hamiltonian constraint. It shoots on `omega` to satisfy the infrared charged Robin condition with a positive nodeless amplitude, and derives both brane scalar potentials and surface stress values from the resulting profiles. The scalar potentials have independent linear `chi` couplings and quadratic U(1)-invariant `F` couplings, sufficient to implement all field jumps. Their constant terms reproduce the Israel surface energy densities.

**Reference numerical results** (801 output points; high-accuracy DOP853 integration):

| Quantity | Result |
|---|---:|
| Condensate frequency `omega` | 1.088007652353 |
| IR `rho+p` | -0.673250329248 |
| Bulk conserved U(1) charge | about 0.005402 |
| Maximum Hamiltonian error | 2.85e-13 |
| Independent integrated identity (Simpson) error | 2.92e-15 |
| IR charged-field Robin residual | 3.38e-13 |
| Tensor effective speed `c_T` | 1.0051921593 |

A concrete local action for the IR NEC-violating surface stress adds a homogeneous brane scalar `zeta=v t` with the **wrong-sign kinetic** action

\[
S_{\mathrm{ghost},L}=\int d^4x\sqrt{-\gamma}\left[+\frac12\gamma^{ab}\partial_a\zeta\partial_b\zeta\right],
\quad \rho_g+p_g=-v^2e^{-2N_L}.
\]

The generated brane-action parameters set `v² e^{-2N_L}=0.673250...`, and the remaining potential terms reproduce all Israel and scalar jump conditions. This explicitly realizes the mathematical background **at the price of a propagating ghost with an unbounded Hamiltonian**. The solution is not a viable stable cosmology and offers no acceptable superfluid dark-energy mechanism.

## Finite-momentum gauge-invariant tensor sector

Three-dimensionally transverse-traceless (TT) metric perturbations form a gauge-invariant sector under the unbroken three-dimensional rotation symmetry. With isotropic spatial brane stress and no TT source, their equation is

\[
-\partial_y(e^{N+3A}h')+p^2e^{N+A}h
=\Omega^2 e^{-N+3A}h,\qquad h'(0)=h'(L)=0.
\]

The normal-mode problem has positive kinetic weight, nonnegative radial-gradient energy and nonnegative `p²` gradient energy. P1 finite elements with consistent mass matrices reproduce the Neumann zero mode and show positive `Omega²` for each `p>0`, with second-order convergence of the first massive graviton mode. This establishes **only TT sector stability**, leaving the radion–phase–brane-ghost scalar sector uncontrolled.

The low-momentum tensor speed in the UV-brane reference clock is

\[
\boxed{c_T^2=\frac{\int_0^L e^{N+A}dy}{\int_0^L e^{-N+3A}dy}}
=1.010411...,\qquad c_T=1.005192... .
\]

Since the background enforces `N-A>0` throughout the bulk away from the UV surface, this tensor mode propagates faster than an ordinary photon confined to the UV brane. If applied as a present-day low-energy gravity prediction, the difference `c_T-1≈5.2e-3` conflicts dramatically with the GW170817/GRB170817A empirical constraint, which is of order `10^-15`. The model's normalization is dimensionless; this is a **conditional phenomenological exclusion**, assuming the low-momentum mode maps to observed gravitational waves without an additional mechanism restoring luminality.

## Validation gates (26 tests)

* Positive nodeless rotating field, finite conserved charge and UV/IR Robin conditions.
* Independently audited `tt`, `ii`, `yy` Einstein components, the UV Hamiltonian constraint and its bulk propagation.
* Local/weighted-integrated anisotropy identities, convergent quadrature, and inferred Israel NEC signs.
* Background charged-amplitude sweep and exact zero-charge Goldberger–Wise recovery.
* Explicit ghost IR action stress, GW linear jumps, and charge-flux check.
* TT normalizable zero mode, healthy finite-p tensor energy, Neumann boundary conditions and FE convergence.
* Weighted graviton speed agreeing with direct low-p eigenvalue, and exact luminal zero-charge control.

## What remains (WS-08 candidate)

1. Seek a **time-dependent** compactified charged solution with both branes NEC-respecting. WS-04/05 evolving numerical initial data offer a first seed; realistic brane scalar actions and time-dependent junctions must be treated together.
2. Derive the complete five-dimensional quadratic constraint system at finite three-momentum, including charged phase, amplitude, Goldberger–Wise radion, scalar metric lapse/shift, brane bending and physical Israel conditions. Solve the **physical**, constraint-reduced kinetic/gradient spectrum and time-evolution transfer.
3. Replace the global complex scalar analogue with a microscopic helium-3 BdG A-phase calculation only in a separate laboratory-analogue study. The helium-3 superfluid transition scale is millikelvin, distinct from the meV dark-energy fourth-root scale.
4. Produce a vacuum energy dynamically (the present action has an adjusted negative five-dimensional potential and surface constants), and match observed gravitational-wave propagation and cosmological expansion.

## Literature

- Gary Gibbons, Renata Kallosh, Andrei Linde (2001), [Brane World Sum Rules](https://arxiv.org/abs/hep-th/0011225).
- Kari Enqvist, Esko Keski-Vakkuri, Syksy Räsänen (2001), [Constraints on brane and bulk ideal fluid in Randall-Sundrum cosmologies](https://doi.org/10.1103/PhysRevD.64.044017).
- Lisa Randall and Raman Sundrum (1999), [A Large Mass Hierarchy from a Small Extra Dimension](https://doi.org/10.1103/PhysRevLett.83.3370).
- W. Goldberger and M. Wise (1999), [Modulus Stabilization with Bulk Fields](https://doi.org/10.1103/PhysRevLett.83.4922).
- LIGO–Virgo/Fermi (2017), [Gravitational Waves and Gamma-rays from a Binary Neutron Star Merger](https://dcc-lho.ligo.org/LIGO-P1700308/public).