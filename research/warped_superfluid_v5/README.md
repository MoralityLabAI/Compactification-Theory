# WS-05: Charged-bulk tangent modes, Goldstone physics, and stability gates

**Research prototype, 29 October 2026.** Builds on the five-dimensional nonlinear charged bulk Einstein–scalar initial-boundary evolution of [WS-04 (PR #11)](https://github.com/MoralityLabAI/Compactification-Theory/pull/11), itself stacked on [WS-03 (PR #8)](https://github.com/MoralityLabAI/Compactification-Theory/pull/8). This package calculates the *full homogeneous tangent response of all ten WS-04 method-of-lines variables*, and separately solves the exact flat-space equilibrium two-branch spectrum of a canonical globally charged complex scalar.

**Scientific status:** full five-dimensional *gauge-fixed* tangent response for three-dimensionally homogeneous perturbations, plus an analytically controlled Goldstone benchmark. The fully gauge-invariant, momentum-dependent radion–condensate spectrum of the charged expanding two-brane spacetime has **not** yet been derived. No helium-3 experimental data or cosmological observations are fitted. No microphysical dark-energy mechanism is established.

## Reproduce

Requires the neighboring `research/warped_superfluid_v3` and `research/warped_superfluid_v4` folders from the stacked PRs. Python 3.12 or newer; CPU-only:

```shell
cd research/warped_superfluid_v5
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
```

Run outputs `results/summary.json` and five independent PNG diagnostic plots. The package has **20 locally passing regression tests** and a scoped GitHub Actions workflow. It imports `BulkChargedSystem` from WS-04 without modifying WS-03/04 source.

## 1. Exact homogeneous Goldstone control

For two real fields `x,v` with global U(1), use metric signature `(-,+,+,+)` and

\[
 {\cal L}=-\tfrac12\partial_\mu x\,\partial^\mu x-\tfrac12\partial_\mu v\,\partial^\mu v
 -\tfrac12m_B^2(x^2+v^2)-\tfrac14\lambda_B(x^2+v^2)^2 .
\]

A constant-amplitude charged solution of the *flat-space* bulk field equations is

\[
 x+iv=F e^{i\omega t},\qquad\omega^2=m_B^2+\lambda_B F^2.
\]

Writing the amplitude and phase as `F+r` and `omega*t+pi`, the quadratic Lagrangian is

\[
 {\cal L}_2=-\tfrac12\partial_\mu r\,\partial^\mu r+
 -\tfrac12F^2\partial_\mu\pi\,\partial^\mu\pi+2 F\omega r\dot\pi
 -\lambda_B F^2r^2.
\]

Its exact real-frequency determinant is

\[
 (p^2+2\lambda_BF^2-\Omega^2)(p^2-\Omega^2)
 -4\omega^2\Omega^2=0.
\]

With `M²=2 lambda_B F²`, the two positive squared frequencies are

\[
 \Omega_\pm^2=p^2+\frac{M^2+4\omega^2}{2}
 \pm\frac12\sqrt{(M^2+4\omega^2)^2+16\omega^2p^2}.
\]

At `p=0`, the Goldstone branch has `Omega_minus=0` and the gapped amplitude/phase branch obeys `Omega_plus²=M²+4omega²`. At small momentum,

\[
 c_s^2=\frac{M^2}{M^2+4\omega^2}
 =\frac{\omega^2-m_B^2}{3\omega^2-m_B^2}.
\]

This independently matches the thermodynamic susceptibility `c_s²=n/(omega dn/domega)` for `n=omega(omega²-m²)/lambda` along equilibrium. A numerically stable product-of-roots formula preserves the tiny lower-branch frequencies.

The WS-04 input uses `F=0.02`, `m_B²=0.5`, `lambda_B=0.3`, `omega_input=0.4`. Flat homogeneous equilibrium would require **omega_eq=0.7071916289**. In the reference conformal UV frame, `omega_input²-omega_eq²=-0.34012`. This quantitative mismatch flags a genuinely **nonstationary** charged initial configuration. Radial gradients, 5D expansion, and warp redshift also enter its full local equations, so the homogeneous benchmark is an independent control, not a claimed global warped equilibrium condition.

## 2. Tangent evolution of all ten coupled fields

In WS-04's conformal gauge,

\[
 ds^2=e^{2B(t,y)}(-dt^2+dy^2)+e^{2A(t,y)}d\vec{x}^2,
\]

and the discretized state is

`U = (A,B,chi,x,v,A_t,B_t,chi_t,x_t,v_t)`, over `n` y nodes, with brane-localized scalar and Israel junction derivatives **enforced inside** WS-04's RHS `F(U)`.

The new solver evolves the *entire* first variation of this one nonlinear system:

\[
 \partial_t U=F(U),\qquad
 \partial_t\delta U=[\partial_U F(U)]\,\delta U.
\]

The Jacobian-vector product uses an imaginary complex step, `Im[F(U+i h deltaU)]/h` with `h=10^-24`; WS-04's RHS and state-dependent Robin/Israel boundaries are holomorphic in the variables tested. Direct, independent symmetric finite-difference comparisons check that this derivative agrees to `1e-6` absolute accuracy or better for the tested perturbations. This computes the tangent of the **semi-discrete** PDE system, with its spatial truncation errors still present.

To generate physically consistent initial tangents, WS-05 differentiates two nearby *Hamiltonian- and momentum-constraint-solving* WS-04 initial slices in **only** `bulk_amp` (or `bulk_omega`). These vary initial data, while all bulk and brane action parameters remain the same. It avoids arbitrary metric initial perturbations that violate the constraints at leading order.

We measure:

* **UV Hubble rate in UV proper time:** `H_UV = exp(-B(0)) A_t(0)`, with tangent `delta H_UV = exp(-B(0)) [delta A_t(0) - A_t(0) delta B(0)]`.
* **Global charge:** `Q = integral exp(3A)(x v_t - v x_t) dy`, including its first variation and exact U(1) Ward symmetry.
* **A radion-related length proxy:** `L_t = integral exp(B)dy` along the specified constant-conformal-time slice. Its first variation is `integral exp(B)delta B dy`. This depends on the time slicing; it is **not** a fully gauge-invariant radion observable.

The exact internal phase-rotation tangent is `delta x=-v`, `delta v=x`, `delta x_t=-v_t`, `delta v_t=x_t`, with all other components zero. The nonlinear U(1) equivariance condition gives

\[
 J_F(U)\,T_{U(1)}U=T_{U(1)}F(U).
\]

This Ward identity holds at the evolving state to `1.55e-14` maximum absolute residual. The phase rotation also changes neither `H_UV`, global charge, nor the chosen length proxy, and satisfies the linearized brane boundary map and Einstein constraints to roundoff.

## 3. Measured results

Reference `k=M5³=1`, proper background length `1`, GW parameters `u=0.25`, `chi_uv=0.4`, globally charged bulk scalar `bulk_amp=0.02`, `omega_input=0.4`, `mass²=0.5`, `lambda=0.3`, Robin stiffness 1 at both branes. Perturbation `d/d(bulk_amp)`. Evolve from conformal `t=0` to `t=0.12`, 7 samples.

| Grid nodes | Max linearized Hamiltonian residual | Max linearized momentum residual | dH_UV/dF at t=.12 | dQ/dF at t=.12 |
|--:|--:|--:|--:|--:|
| 33 | 1.19335e-3 | 7.80585e-4 | 0.544463 | 0.00822307 |
| 65 | 3.52418e-4 | 2.33759e-4 | 0.544536 | 0.00821752 |
| 129 | 1.00086e-4 | 6.71215e-5 | 0.544558 | 0.00821612 |

Both linearized Einstein constraints decrease under refinement. The charge derivative drifts minimally (for 129 nodes, initial 0.008216124349, final 0.008216122012). The relative errors against *independently re-solved full nonlinear* WS-04 trajectories at `F ± 5e-4` are:

| Observable | Max relative difference |
|--|--:|
| dH_UV/dF | 7.34e-9 |
| dL_t/dF | 1.85e-8 |
| dQ/dF | 6.16e-8 |

The central finite-difference error decreases quadratically with the amplitude perturbation, another separate acceptance gate. These results are within a short-time numerical regime; they do not establish a long-duration non-tachyonic charged background.

## 4. Boundaries and what is still missing

The gauge-invariant Lesgourgues–Sorbo radion calculation in WS-04 applies to **one static uncharged background scalar**. In contrast, WS-05 evolves three-dimensionally homogeneous perturbations of an **expanding** geometry with two active scalar sectors (GW plus complex charge). A time-dependent 5D background and finite charge produce both metric gauge modes and physical scalar mixing; their fully gauge-invariant decomposition requires a 3D Fourier scalar-sector ansatz and systematic elimination of the linearized Einstein constraints and brane bending modes. A positive Goldstone dispersion in the *flat homogeneous equilibrium* control does not imply stability of the full curved, charged two-brane system.

Next acceptance gates:

1. Construct a charged **stationary** equilibrium or periodic phase background satisfying both bulk scalar boundary conditions, fixed charge, and all Einstein/brane constraints. Its existence requires an actual nonlinear boundary-value or periodic orbit solution, not simply a choice of phase frequency.
2. Add 3D Fourier scalar perturbations with `p≠0`, derive gauge-invariant phase/radion/isocurvature combinations and the physical boundary perturbations, and test no-ghost/gradient eigenvalues and zero-mode limits.
3. Compare full Hamiltonian + momentum constraints and Ward charge identity across long times; establish convergence and gauge-independence of physical observables.
4. Fit laboratory helium-3 BdG textured backgrounds independently of the cosmological solver, then test cosmological H(z) and gravity bounds. The Planck/electroweak and dark-energy hierarchies remain inputs, not predictions.

## Literature

- Lesgourgues & Sorbo, *Goldberger-Wise variations: stabilizing brane models with a bulk scalar* (2004), [arXiv:hep-th/0310007](https://arxiv.org/abs/hep-th/0310007).
- Bridgman, Malik & Wands, *Cosmological perturbations in the bulk and on the brane* (2002), [Phys. Rev. D 65, 043502](https://doi.org/10.1103/PhysRevD.65.043502).
- Cline & Firouzjahi, *5-dimensional warped cosmological solutions with radius stabilization by a bulk scalar* (2000), [Physics Letters B](https://doi.org/10.1016/S0370-2693(00)01254-5).
- Alford, Mallavarapu, Schmitt & Stetina, *From a complex scalar field to the two-fluid picture of superfluidity* (2013), [Phys. Rev. D 87, 065001](https://doi.org/10.1103/PhysRevD.87.065001).