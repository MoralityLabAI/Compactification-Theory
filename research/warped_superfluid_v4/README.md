# WS-04: gauge-invariant radion and charged bulk superfluid analogue

**Research prototype (29 October 2026).** Continues [WS-03](https://github.com/MoralityLabAI/Compactification-Theory/pull/8), where the charge was restricted to the ultraviolet brane. This package adds a canonical complex scalar carrying a conserved **global U(1)** charge throughout the bulk, including its full nonlinear Einstein backreaction. It independently computes the coupled **metric–Goldberger–Wise scalar gauge-invariant radion spectrum** on the *uncharged static* reference background.

The radion eigenproblem has **one static GW scalar** and flat four-dimensional slices. The evolving charged background has **two bulk scalars** and is time-dependent. The former spectrum cannot certify stability of the latter, whose coupled gauge-invariant perturbation equations remain an open problem.

## Run

From `research/warped_superfluid_v4/` (with sibling `research/warped_superfluid_v3/` from the WS-03 PR available):

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
```

CPU-only NumPy, SciPy, and matplotlib. The 22 tests and generated figures run in GitHub Actions. Reproducible JSON and plots are written to `results/`.

## Gauge-invariant metric–scalar sector

We implement Eqs. (6)–(10), (16)–(17) of [Lesgourgues and Sorbo (2004)](https://arxiv.org/abs/hep-th/0310007), for the exact two-brane GW background:

```text
W(chi) = 3 M5³ k + u chi²/2
chi(z) = chi0 exp(u z)
warp(z) = k z + chi0² (exp(2 u z)-1) / (12 M5³)
a(z) = exp(-warp(z)),   dy = dz/a(z).
```

In conformal y let `H=a_y/a`, `r=chi_yy/chi_y`, and
`Pi=Psi_y+2H Psi`. The physical coupled radion satisfies

```text
Psi_y = Pi - 2 H Psi
Pi_y  = (-H + 2r) Pi + [2 chi_y²/(3 M5³) - m²] Psi.
g_uv Pi(0) + m² Psi(0) = 0
g_ir Pi(L) + m² Psi(L) = 0
g_uv = H-r+a U''_uv/2 = +a(0) kappa_uv/2
g_ir = H-r-a U''_ir/2 = -a(L) kappa_ir/2
Phi=-2 Psi,  Delta=-3 Pi/chi_y.
```

The correct Robin conditions depend on the eigenvalue `m²`. For positive GW stiffness on both branes, `H<0`, `chi_y>0`, `g_uv>0`, and `g_ir<0`. These satisfy the necessary and sufficient two-brane radion stability condition proved by Lesgourgues and Sorbo for a monotone one-scalar background. The numerical tachyon scan confirms the expected sign. A deliberately negative IR stiffness flips the sign condition and produces a tachyonic mode.

At `k=M5³=L=1, u=.25, chi_uv=.4, kappa_uv=kappa_ir=8`:

| Index | Coupled gauge-invariant radion m² | Fixed-metric GW scalar probe m² (WS-03) |
|---:|---:|---:|
| 1 | **0.001180654105** | 2.423279 |
| 2 | 2.431109734 | 8.124458 |
| 3 | 8.132671781 | 18.934577 |
| 4 | 18.941529665 | 35.841749 |

The physical lightest radion mass is `0.03436 k`. Its squared mass is more than two thousand times below the first fixed-metric scalar probe. Replacing IR stiffness `+8` with `-3` produces `m²=-0.001625596158`.

## Full five-dimensional global charge and backreaction

The **same** Einstein–GW background now includes two real fields `x,v` representing a complex scalar, with kinetic terms `-((dx)²+(dv)²)/2` and potential

```text
V_bulk(x,v) = mB² (x²+v²)/2 + lambdaB (x²+v²)²/4.
U_uv, U_ir each gain kappaB (x²+v²)/2.
```

The finite-density initial phase is `x(t=0,y)=F(y)`, `v=0`, `x_t=0`, `v_t=omega F(y)`. Both fields satisfy the same endpoint Robin condition, so `e^{3A}(x v_y-v x_y)=0` on each brane. The global Noether charge is

```text
Q(t) = integral_0^yL exp(3A) (x v_t - v x_t) dy.
```

The canonical stress tensor contributes `T^t_t - T^i_i=-exp(-2B) omega² F²` at the initial slice. The metric must therefore evolve: simply inserting finite charge into a static four-dimensionally Lorentz-invariant warp would violate its Einstein equations.

Metric ansatz is exactly the one used by WS-03:

```text
ds² = exp(2B(t,y))(-dt²+dy²) + exp(2A(t,y)) dvecx².
```

The PDE contains `A,B,chi,x,v` and their first time derivatives. It solves a constraint-satisfying initial hypersurface by adjusting B and chi profiles to enforce both temporal and scalar brane conditions, interpolating F with the two scalar Robin slopes, integrating the Einstein Hamiltonian constraint for A, and solving for H² from the remaining IR spatial junction. The code then evolves with the full total scalar potential and both scalar kinetic terms in Einstein's equations, applies **ten** junction conditions, and measures independent Hamiltonian, momentum, current-flux, and boundary-slope errors.

Test values: `kL=1`, `bulk_amp=.02`, `omega=.4`, `mB²=.5`, `lambdaB=.3`, `kappaB=1`, evolution `0<=t<=.2`.

| Grid nodes | max Hamiltonian constraint | max momentum constraint | max relative Q drift |
|---:|---:|---:|---:|
| 65 | 2.19636e-3 | 2.06687e-3 | 3.88752e-6 |
| 129 | 6.10114e-4 | 5.83422e-4 | 4.78461e-7 |
| 257 | 1.62701e-4 | 1.56713e-4 | 1.02140e-7 |

The initial *smooth* constraints satisfy the equations to roundoff, with independent grid derivatives showing second-order convergence in the evolved residuals. The endpoint Noether flux vanishes. Under `omega -> -omega`, the charge changes sign and the initial H² is unchanged. The zero-amplitude zero-charge limit returns the WS-03 BPS solution up to conformal-coordinate interpolation tolerance.

## Scientific boundaries

The full charged system is a **global-U(1) coherent scalar**, representing a superfluid analogue. No electromagnetic gauging, microscopic helium-3 quasiparticle calculation, phase transition, or dark-energy data fit has been performed. The tests prove **finite-time numerical consistency**, alongside the **linear radion stability of the uncharged BPS reference**. Long-time stability, radion–phase mixing on the charged evolving background, physical hierarchy scales, observable cosmology, and generation of the renormalized vacuum energy remain open.

Useful comparisons: [Giovannini (2001)](https://doi.org/10.1103/PhysRevD.64.064023), [Goldberger and Wise (1999)](https://doi.org/10.1103/PhysRevLett.83.4922), [Mukohyama (2002)](https://doi.org/10.1103/PhysRevD.65.084036), [Olechowski (2025)](https://doi.org/10.1016/j.nuclphysb.2025.116807).
