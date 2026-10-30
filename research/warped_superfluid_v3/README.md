# WS-03: Coupled 5D superfluid-brane cosmology with Goldberger–Wise stabilization

**Research prototype, 2026-10-29.** A CPU-runnable 5D Einstein–scalar system with a finite-density, charged phase **confined to one brane**, two dynamical brane junctions, and a bulk Goldberger–Wise-type stabilizing scalar. The full nonlinear **time-dependent bulk PDE** is evolved and the independent Hamiltonian and momentum constraints are measured under spatial-grid refinement. This project continues [WS-01](https://github.com/MoralityLabAI/Compactification-Theory/pull/6) and [WS-02](https://github.com/MoralityLabAI/Compactification-Theory/pull/7) without depending on either PR being merged.

The computation is dimensionless and uses a modest warp `kL=1` to ensure a well-conditioned, independently testable numerical baseline. The construction **does not solve the cosmological-constant problem**. A gauge-invariant, coupled radion and metric-scalar eigenmode spectrum remains an open research gate; the positive bulk-scalar *probe* spectrum is explicitly labeled as such. The charged condensate is UV-brane-localized, rather than a charged complex scalar propagating throughout the five-dimensional bulk. All reported stability statements refer to the tested finite time interval.

## Reproduce

```bash
cd research/warped_superfluid_v3
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
```

`run_experiment.py` creates `results/summary.json` and five independent diagnostics plots. GitHub Actions runs these commands on pull requests. Numerical tests require only NumPy, SciPy and matplotlib on CPU.

## 1. One 5D action

Use the physical five-dimensional metric, bulk scalar and brane matter

\[
\begin{split}
 ds_5^2 &=e^{2B(t,y)}(-dt^2+dy^2)+e^{2A(t,y)}d\vec{x}^{\,2},\\
 S &= \int d^5 x\sqrt{-g_5}\left[\frac{M_5^3}{2}R_5-\frac12(\partial\chi)^2-V(\chi)\right]\\
 &\quad-\sum_{i=0,L}\int_i d^4x\sqrt{-h_i}\,\lambda_i(\chi)
 +\int_{y=0}d^4x\sqrt{-h_0}\,P(X).
\end{split}
\]

The gravitational action assumes the Gibbons–Hawking boundary terms and the usual `Z2` identification at each brane. The phase `theta(t)` is confined to the UV brane, with `X=-(h0^μν ∂μ theta ∂ν theta)/2` and `P(X)=X²` for the baseline. Consequently the phase charge is `Q=a_UV³ P_X (dtheta/dtau_UV)`, a conserved quantity. Its fluid has

\[
 s(t)=s_0 e^{-4[A(t,0)-A(0,0)]},\qquad \rho_0=3s,\quad p_0=s,
 \quad\rho_0+p_0=4s>0,\quad c_s^2=\frac13.
\]

Choose a superpotential

\[
W(\chi)=3M_5^3 k+\frac{u}{2}\chi^2,
\quad V(\chi)=\frac12 W_\chi^2-\frac{2}{3M_5^3}W^2.
\]

The brane potentials are fixed and explicit:

\[
\lambda_0=2W(\chi)+\frac{\kappa_0}{2}(\chi-v_0)^2+\delta\lambda_0,
\quad\lambda_L=-2W(\chi)+\frac{\kappa_L}{2}(\chi-v_L)^2.
\]

`v0=chi0`, `vL=chi0 exp(u Lproper)`. Positive scalar stiffness values `κ0=κL=8` anchor the background values at both branes. `δλ0` is a separate UV vacuum-energy detuning parameter used to test vacuum sensitivity. The proper fifth-dimensional length evolves as `D(t)=∫_0^yL exp(B(t,y)) dy`.

### Exact tuned static reference

In proper transverse coordinate `z`, the full nonlinear bulk Einstein–scalar equations admit

\[
\chi(z)=v_0e^{uz},\quad
 ds^2=e^{-2\mathcal{W}(z)}\eta_{\mu\nu}dx^\mu dx^\nu+dz^2,\quad
\mathcal{W}(z)=kz+\frac{v_0^2}{12M_5^3}(e^{2uz}-1).
\]

`W` is the superpotential and `mathcal W` is the physical warp exponent. The chosen two brane actions satisfy both Israel conditions and the scalar boundary conditions exactly at `z=0` and `z=Lproper`. Conformal coordinates are obtained through `dy=exp(mathcal W(z)) dz`. The baseline values are `k=M5³=1, u=.25, v0=.4, Lproper=1`.

## 2. Full time-dependent bulk equations

Writing `dot=∂t`, `prime=∂y`, the evolved equations are

\[
\begin{split}
 \ddot A&=A''-3\dot A^2+3A'^2+\frac{2e^{2B}}{3M_5^3}V,\\
 \ddot B&=B''+3\dot A^2-3A'^2-\frac{\dot\chi^2-\chi'^2}{2M_5^3}-\frac{e^{2B}V}{3M_5^3},\\
 \ddot\chi&=\chi''-3\dot A\dot\chi+3A'\chi'-e^{2B}V_\chi.
\end{split}
\]

Independent Einstein constraints used to validate the solution are

\[
\begin{split}
 C_{00}&=3[-A''+\dot B\dot A+B'A'+\dot A^2-2A'^2]
 -\frac{\dot\chi^2+\chi'^2+2e^{2B}V}{2M_5^3},\\
 C_{0y}&=-3[\dot A'-B'\dot A-\dot B A'+\dot A A']
 -\frac{\dot\chi\chi'}{M_5^3}.
\end{split}
\]

### Both boundary junctions

At `y=0`, the `Z2` Israel and scalar conditions are

\[
\begin{split}
 e^{-B}A' &=-\frac{\lambda_0+\rho_0}{6M_5^3},&
 e^{-B}B'&=-\frac{\lambda_0-2\rho_0-3p_0}{6M_5^3},&
 e^{-B}\chi'&=\frac{\lambda_{0,\chi}}{2}.
\end{split}
\]

At `y=L`, with no separate IR fluid, they are

\[
 e^{-B}A'=e^{-B}B'=\frac{\lambda_L}{6M_5^3},
\qquad e^{-B}\chi'=-\frac{\lambda_{L,\chi}}{2}.
\]

The PDE boundary stencil applies these time-varying conditions; an independent fourth-order one-sided derivative checks them. A fixed, empty IR brane was inconsistent in WS-02's *pure AdS* UV-matter background. With the coupled bulk stabilizer and the deformed initial geometry here, both IR conditions can be satisfied without adding a fitted IR fluid.

## 3. Constructing constraint-satisfying charged initial data

Set initial `Bdot=chidot=0`, choose `B(y)` near the exact BPS background with its slopes satisfying the time-component Israel junctions on both ends, and choose a smooth `chi(y)` satisfying both scalar junctions. An optional interior scalar bump vanishes together with its first derivative at both branes, leaving the action fixed. Momentum conservation then fixes

\[
 \dot A(0,y)=H_{\rm UV}\exp[B(0,y)-A(0,y)].
\]

The Hamiltonian constraint becomes a nonlinear ODE for `A(y)`, with its UV initial slope fixed by the UV spatial Israel condition. `H_UV²` is determined by the remaining IR spatial Israel condition through a bracketed scalar root solve. This constructs a legitimate constraint-satisfying initial hypersurface for the full time-dependent system. The bulk variables are then evolved using method of lines, consistent Neumann/Robin brane slopes, third-order one-sided boundary second derivatives and a high-accuracy explicit ODE integrator.

For `s0=.001`, the result is

| Initial quantity | Dimensionless value |
|---|---:|
| Charged fluid density `rho_UV=3s0` | 0.003 |
| `H_UV²` from 5D constraints and both boundaries | 0.00116238226284 |
| `H_UV` | 0.03409372762 |
| `H²/rho` measured in 5D | 0.3874607543 |
| `H²/rho` predicted by 4D warped-gravity reduction | 0.3875425067 |
| Maximum analytic initial Hamiltonian residual | 2.22e-16 |
| Maximum analytic initial momentum residual | 2.08e-17 |
| Maximum of six initial junction residuals | 5.9e-15 |

The effective four-dimensional Planck scale is computed from this **backreacted** static background,

\[
 M_4^2=2M_5^3\int_0^{L_{\rm proper}} e^{-2\mathcal{W}(z)}\,dz
 \approx0.86012 M_5^3/k.
\]

The agreement at initial finite density is approximately 0.021% in `H²/rho`. The integral's factor of two counts both orbifold sides.

## 4. Nonlinear numerical evolution: convergence

The independent bulk Einstein constraints converge as the uniform spatial mesh is refined during `0≤t≤.4`:

| Grid nodes | max `abs(C00)` in interior | max `abs(C0y)` in interior | independent junction slope mismatch |
|---:|---:|---:|---:|
| 65 | 3.324e-3 | 3.380e-3 | 3.436e-5 |
| 129 | 9.402e-4 | 9.610e-4 | 4.175e-6 |
| 257 | 2.537e-4 | 2.597e-4 | 8.50e-7 |

The constraint decrease is consistent with the second-order interior finite-difference scheme. The boundary estimate uses an independent one-sided derivative formula and converges separately. Conserved brane phase charge has relative arithmetic drift below `5e-16`. The exact uncharged BPS geometry is independently checked as the zero-charge limit under mesh refinement.

### Coupled radion response

With the **same bulk and brane action**, vary the initial Goldberger–Wise scalar in the interior by a small, smooth `±0.001` bump that vanishes with its first derivative at the branes. Re-solve the entire Einstein constraint and the IR junction for each perturbation, then evolve the 5D PDE to `t=1`.

\[
\frac{\partial D(1)}{\partial\varepsilon}
\approx\frac{D_{+0.001}(1)-D_{-0.001}(1)}{0.002}.
\]

Results for `N=65,129,257` are `0.00686458`, `0.00688543`, and `0.00689089`. The early-time response remains bounded over the tested interval and converges under refinement. This verifies a finite-time dynamical radion response, **not** the complete physical radion perturbation spectrum or long-term stability. At `t=1`, the independent PDE constraints still converge but are larger than at `t=.4`.

## 5. A direct vacuum-energy sensitivity test

Introduce a constant positive UV brane tension detuning `δλ0`, set finite charge to zero, and solve the same 5D initial-value constraints and both brane junctions. The resulting initial cosmological curvature is:

| `δλ0` | `H_UV²` | `H²/δλ0` |
|---:|---:|---:|
| 1.0e-4 | 3.874959e-5 | 0.387496 |
| 1.0e-3 | 3.875467e-4 | 0.387547 |
| 1.0e-2 | 3.876314e-3 | 0.387631 |

The low-detuning slope closely matches `1/(3M4²)≈0.387543`. In this construction a positive vacuum tension detuning produces positive cosmological curvature. The stabilizing scalar does not remove vacuum-energy sensitivity. This is a controlled counterexample to strong self-tuning claims for this specific action.

## 6. Fixed-metric Goldberger–Wise fluctuation probe

For a **fixed** BPS background, a canonical scalar perturbation obeys the weighted Sturm–Liouville equation

\[
-(e^{-4\mathcal W} f')'+e^{-4\mathcal W}V_{\chi\chi}f
 =m^2e^{-2\mathcal W}f,
\]

with Robin conditions from the finite stiffness terms. The lowest four squared eigenmasses at `N=257` are approximately

`[2.4232791, 8.1244583, 18.9345774, 35.8417489]`.

The eigenvalues are positive and converge with P1 finite elements. They measure the fixed-background **bulk scalar probe** only. A physically complete radion stability verdict requires linearizing the 5D scalar metric fluctuations together with the Goldberger–Wise scalar, identifying gauge-invariant variables and checking their self-adjoint boundary conditions.

## What remains to be derived

1. The gauge-invariant coupled radion/scalar metric spectrum and possible tachyonic modes. Positive probe eigenvalues are insufficient to certify full stability.
2. Long-time evolution with constraint preservation and diagnostics that converge at the level required for cosmological inference. The current verified interval is short and dimensionless.
3. A complex **bulk** condensate with charge, including its phase, bulk stress, gauge interactions and boundary current flux. The present phase stays on the UV brane.
4. A physical hierarchy (`kL~35` or a dark-energy scale), realistic boundary potentials and a renormalized 4D effective cosmological constant. Here the existing tuned vacuum is an input.
5. Fits to H(z), structure growth, fifth-force, radion KK and gravitational-wave speed constraints.

## Main literature

- Randall and Sundrum (1999), [large mass hierarchy from small extra dimension](https://doi.org/10.1103/PhysRevLett.83.3370).
- Goldberger and Wise (1999), [modulus stabilization](https://doi.org/10.1103/PhysRevLett.83.4922).
- DeWolfe, Freedman, Gubser and Karch (2000), [fully backreacted scalar-gravity warped domain walls](https://doi.org/10.1103/PhysRevD.62.046008).
- Kanti, Lee and Olive (2003), [time-dependent brane models and radion stability](https://doi.org/10.1103/PhysRevD.67.024037).
- Martin, Felder, Frolov, Peloso and Kofman (2005), [BRANECODE full nonlinear two-brane evolution](https://doi.org/10.1016/j.cpc.2005.04.009).
- Olechowski (2008), [radion stability with nonstandard boundary conditions](https://doi.org/10.1103/PhysRevD.78.084036).

This experiment independently derives a small, numerically tractable version of coupled 5D bulk evolution. BRANECODE already established the wider class of full nonlinear simulations; the novelty of WS-03 is its combined charged brane phase, two-potential initial-data construction, quantitative convergence gates and explicit vacuum sensitivity analysis within this repo's experiment series.
