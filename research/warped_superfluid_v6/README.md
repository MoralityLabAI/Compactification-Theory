# WS-06: finite-momentum charged bulk probe and gauge-invariant 4D radion–phase transfer

**Status: reproducible intermediate result, 29 October 2026.** This is a stacked research continuation of [WS-05, PR #12](https://github.com/MoralityLabAI/Compactification-Theory/pull/12). The work contains (i) a **self-consistent, stationary charged scalar probe** on a fixed two-brane Randall–Sundrum metric, including both Robin boundary conditions and arbitrary positive three-momentum, and (ii) the **exact gauge-invariant scalar metric/matter perturbation equations in a declared four-dimensional effective action** incorporating a radion, charged amplitude and phase, and the Einstein constraint corrections. Together these are substantial but distinct pieces of a full 5D gauge-invariant perturbation problem.

**Crucial distinction:** The stationary 5D probe **does not solve the 5D Einstein equations with its charge stress included**. The 4D effective theory is defined separately with a radion coupling chosen phenomenologically. Neither is the 5D gauge-invariant, finite-momentum, charged, two-brane **coupled** physical spectrum. The next step is to derive that system from the full 5D action, including metric lapse/shift and brane-bending perturbations, and to check its gauge-invariant junction conditions. The dark-energy scale and helium-3 microscopic physics are not derived.

## Reproduction

Python 3.12+, NumPy, SciPy and Matplotlib on CPU:

```bash
cd research/warped_superfluid_v6
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
```

The `results/summary.json` is a checked-in numerical reference. The runner independently recreates it and the four diagnostic PNG figures. The scoped workflow runs 26 tests and generates the plots. It does not overwrite earlier WS-01 through WS-05 results.

## A. Full fifth-dimensional **probe** condensate and brane physics

Set `k=M5³=L=1`. Take the **fixed** RS background

\[ds^2=e^{-2ky}(-dt^2+d\mathbf x^2)+dy^2,\qquad 0\le y\le L.\]

For a canonical two-real-field complex scalar, with bulk potential `V=m²|Ψ|²/2+λ|Ψ|⁴/4`, seek an oscillating solution `Ψ=F(y) exp(iωt)`. The exact bulk equation on this background is

\[
F''-4kF'+e^{2ky}\omega^2F-m^2F-\lambda F^3=0.
\]

Brane quadratic potentials give proper-coordinate Robin derivatives

\[
F'(0)=\tfrac12\kappa_{UV}F(0),\qquad
F'(L)=-\tfrac12\kappa_{IR}F(L).
\]

The P1 finite-element Galerkin system (two-point Gaussian quadrature) is

\[
[K_0+\lambda\,T(F^2)]F=\omega^2 MF,\quad F^TMF=q^2,\quad
M_{ij}=\int_0^L e^{-2ky}N_iN_jdy.
\]

`K0` includes the bulk gradient, mass term and both warp-weighted Robin boundary terms. The quartic term is assembled with the same shape functions and quadrature, rather than pointwise collocation. A damped Newton method solves for the field profile and eigenfrequency simultaneously at fixed `q=0.32`. The nodeless positive solution converges in the expected second-order manner as nodes increase through 33, 65, 129 and 257. Both one-sided second-order boundary slope residuals decrease approximately fourfold under each grid doubling.

For `m²=.5`, `λ=.3`, `κ_UV=κ_IR=1`, at 65 nodes:

- `ω=1.09254802425`, with `ω²=1.19366118529`.
- Linear lowest eigenvalue at zero charge: `ω₀²=1.16259337203`.
- Stationary phase residual `||K_s F||/||K_0 F|| < 5e-12`.
- First eigenvalue of radial stiffness `K_r` relative to M: `0.06212202`.
- Lowest eigenvalue of `K_s` relative to M: numerical zero, representing global U(1).

### Fully coupled amplitude and phase within the **fixed-metric probe**

Let `Ψ=e^{iωt}(F+r+is)` and Fourier transform with 3D comoving momentum `p`. The exact quadratic probe evolution is

\[
\begin{aligned}
M\ddot r-2\omega M\dot s+(K_r+p^2M)r&=0,\\
M\ddot s+2\omega M\dot r+(K_s+p^2M)s&=0,
\end{aligned}
\]

where `K_s F=0` and `K_r=K_s+2 λT(F²)`. With a strictly positive background nodeless ground state, positive λ, and positive Robin stiffness, `K_s` is positive semidefinite and `K_r` positive definite. The energy

\[
E_2=\tfrac12\left[\dot r^TM\dot r+\dot s^TM\dot s
+r^T(K_r+p^2M)r+s^T(K_s+p^2M)s\right]
\]

is conserved exactly by the gyroscopic coupling. This proves **linear probe stability** in the semidiscrete Galerkin theory for p>0, and neutral phase motion at p=0. Finite-element eigenvalues, direct energy conservation and an deliberately destabilized radial stiffness are independent tests. For p=.2, the largest numerical real part among all time-evolution eigenvalues is ~6e-14.

Eliminating the gapped amplitude at long wavelengths gives a calculable **warped Goldstone sound speed**

\[
\boxed{c_{ph}^2=\left[1+4\omega^2
\frac{F^TMK_r^{-1}MF}{F^TMF}\right]^{-1}\approx0.0128437405}.\]

The direct 5D eigenfrequency at p=.002 gives `(Ω/p)²≈0.01284456`, approaching the projected coefficient as p tends to zero. The speed is measured relative to the coordinate t and x of the fixed RS reference metric; it is not a microscopic experimental velocity of superfluid helium-3.

**The charge stress is physically significant:** in this same background, `T^t_t - T^i_i=-e^{2ky}ω² F(y)²<0`. Thus the charged condensate necessarily spoils exact 4D Lorentz invariance of the fixed warp when gravitationally backreacted. The stationary solution is a legitimate probe-sector background, not an exact self-gravitating 5D solution. It is a possible weak-stress starting guess for coupled numerical Einstein evolution.

## B. Gauge-invariant **four-dimensional** Einstein/radion/charged-matter calculation

Define the explicit Einstein-frame EFT

\[
S_4=\int d^4x\sqrt{-g}\left[\frac{M_4^2}{2}R
-\frac12\sum_{I=1}^3(\partial\phi_I)^2-V(r,x,v)\right],
\]

where `φ=(r,x,v)` includes a canonically normalized radion and Cartesian complex scalar components and

\[
\begin{aligned}
V={}&V_0+\tfrac12m_r^2r^2+\tfrac\beta4r^4
+\tfrac12m_B^2(x^2+v^2)+\tfrac\lambda4(x^2+v^2)^2\\
&+\tfrac g2r(x^2+v^2)+\tfrac\gamma4r^2(x^2+v^2).
\end{aligned}
\]

This has global U(1) symmetry. Its charged `m_B²` and leading quartic are matched to the 5D ground-mode overlap: `m_B²=ω₀²`, `λ₄=λ₅∫e^-4ky u₀(y)^4 dy` where the linear mode obeys `∫e^-2ky u₀²dy=1`. The 4D Planck mass is taken from the Z₂ orbifold RS metric convention `M₄²=M₅³(1-e^-2kL)/k=0.864664717`. The **uncharged** gauge-invariant radion mass `m_r²=0.00118065410539` is imported from WS-04. The interaction `g=.001`, `γ=.002` and radion quartic `β=.001` are chosen exploratory EFT parameters, not coefficients derived from a 5D junction action. The constant `V0=.40` is supplied by hand.

For homogeneous flat FRW, integrate `φ¨+3H φ˙+V_,φ=0`, `H²=(φ˙²/2+V)/(3M₄²)` and `H˙=-φ˙²/(2M₄²)`. At first order, eliminate the 4D scalar Einstein lapse/shift constraints. The (linear) gauge-invariant Mukhanov–Sasaki variables are

\[
Q_I=\delta\phi_I+\frac{\dot\phi_I}{H}\psi
\]

with `g_ij=a²(1−2ψ)δij` in this sign convention. They transform identically under an infinitesimal scalar time reparametrization. Their spatial Fourier modes satisfy

\[
\ddot Q_I+3H\dot Q_I+\left[\frac{p^2}{a^2}\delta_{IJ}+\mathcal M_{IJ}\right]Q_J=0,
\quad
\mathcal M_{IJ}=V_{,IJ}-\frac1{a^3M_4^2}\frac{d}{dt}\left(\frac{a^3}{H}\dot\phi_I\dot\phi_J\right).
\]

This matrix includes the 4D scalar metric perturbations and exactly mixes the radion, charged amplitude and phase (in the Cartesian basis) at finite 3D momentum. For an expanding background, there is no globally time-translation-invariant normal-mode frequency. The numerical observable is the finite-time transfer matrix. Using canonical momenta `Π=a³ Q˙`, the transfer is symplectic, `U(t)^T J U(t)=J`. Its residual after t=.65 stays below 1e-15, and the background Friedmann and charge conservation errors are below ~4e-16 and ~2e-15, respectively. The initial and final background Hubble rates are approximately 0.44835 and 0.42497.

At p=.2, the model predicts a nonzero finite-time transfer `∂Q_rad(t=.65)/∂Π_phase(t=0)=-2.27e-6` with the chosen weak coupling. When `g=0`, the same transfer vanishes in this U(1)-symmetric reference. All three instantaneous scalar mass-matrix eigenvalues remain positive over the simulated interval. The kinetic and gradient matrix eigenvalues equal one by construction of the canonical EFT. Those positivity results apply to the displayed short-time EFT trajectory; a fully stable five-dimensional radion–phase spectrum remains unproven.

## Explicit validation scope

The **26 local regression gates** cover the two-brane Robin conditions, numerical refinements, charged profile stationarity, Goldstone and radial operator positivity, spectral dispersion, energy and U(1) relations, the controlled negative-stiffness instability, 5D stress anisotropy, KK overlap, potential/Hessian derivatives, 4D scalar gauge transformations, Einstein mass-matrix symmetry, FRW constraint and charge conservation, finite-p transfer dependence and symplectic preservation, and the vanishing radion/phase coupling control.

### Required WS-07 full-theory gate

1. Derive the second variation of the **five-dimensional** Einstein + Goldberger–Wise + complex-scalar + *both brane actions* about an **actual charged, evolving, self-gravitating solution**.
2. Decompose scalar perturbations at finite 3D momentum, including the graviscalar metric fluctuations, lapse and shift constraints, radion/branes' bending and perturbed Israel junctions, then construct complete gauge-invariant variables. In a time-dependent background, test time evolution/Floquet exponents rather than imposing stationary frequencies without justification.
3. Verify that the kinetic and spatial principal symbols on the constrained physical subspace have healthy eigenvalues, compute normalizable mode transfer and monitor constraint growth across grids and gauges. Run an unstable control.
4. Derive a physical dark-energy scale from the system with radion stabilization, instead of prescribing V0, and compare to observational constraints. Independently match the helium-3-A BdG order parameter and texture geometry before making laboratory predictions.

## Literature

- Randall and Sundrum (1999), *A Large Mass Hierarchy from a Small Extra Dimension*: https://doi.org/10.1103/PhysRevLett.83.3370
- Goldberger and Wise (1999), *Modulus Stabilization with Bulk Fields*: https://doi.org/10.1103/PhysRevLett.83.4922
- Lesgourgues and Sorbo (2004), *Goldberger-Wise variations*: https://arxiv.org/abs/hep-th/0310007
- Bridgman, Malik and Wands (2002), *Cosmological perturbations in the bulk and on the brane*: https://doi.org/10.1103/PhysRevD.65.043502
- Kodama, Ishibashi and Seto (2000), *Brane world cosmology: Gauge-invariant formalism for perturbation*: https://doi.org/10.1103/PhysRevD.62.064022
- Van de Bruck, Dorca, Brandenberger and Lukas (2000), *Cosmological perturbations in brane-world theories*: https://doi.org/10.1103/PhysRevD.62.123515
- Maartens and Koyama, *Brane-World Gravity* (review): https://pmc.ncbi.nlm.nih.gov/articles/PMC5255527/
