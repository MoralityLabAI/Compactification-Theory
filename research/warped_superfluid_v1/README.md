# WS-01: Warped Superfluid, Einstein–Scalar Backreaction and Dark Energy

**Status: runnable mathematical prototype, 2026-10-29.** This experiment does not establish the existence of an extra dimension or explain the measured cosmological constant. The helium-3 analogy is conceptual: no actual helium-3 quasiparticle spectrum or experimental fit is used.

## Reproduce

Run from this folder after installing Python 3.12+:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python run_experiment.py

The runner writes results/results.json and results/diagnostics.png. All eight local unit tests pass. This folder is separate from Pluto-Gym and IRC-02.

## 1. Warped eigenmodes

For a warped flat-sliced metric, ds² = exp(-2A(y)) eta_mu_nu dx^mu dx^nu + dy², and y in [0,L], transverse-traceless gravitational excitations (or a minimally coupled massless scalar) satisfy the weighted eigenvalue problem

    -d/dy [exp(-4A) f'] = m² exp(-2A) f
    f'(0) = f'(L) = 0.

The relevant Hilbert-space weight is w(y) = exp(-2A); the stiffness coefficient is p(y) = exp(-4A). With A(y)=ky, k>0, and x = m exp(kL)/k, the positive-mass spectrum is determined by

    J1(x) - J1(x exp(-kL))/Y1(x exp(-kL)) * Y1(x) = 0.

The mode m=0 is constant. At large kL, x_n tends to the nth positive zero of J1. For k=0, m_n = n*pi/L.

The finite-element solution uses a consistent positive-definite generalized mass matrix, two-point Gauss quadrature and Neumann natural boundary conditions. Mesh halvings give the expected approximately factor-of-four decrease in eigenvalue errors. At kL=4 and 257 nodes, the first three positive masses are approximately 0.0702586, 0.1287563, 0.1868897. At kL=35, the first four scaled positive modes are 3.8317059702, 7.0155866698, 10.1734681351 and 13.3236919363.

An analytical mode profile integrated with the correct warped inner product places approximately 99.9993% of the first massive mode's norm in the final ln(10)/k of the warped direction when kL=35.

## 2. The helium-3 analogue test

Helium-3 A quasiparticles have an emergent relativistic-like dispersion near Fermi points. To test the dangers of comparing just that dispersion, the benchmark has a second operator with

    p_speed = exp(-2A), w_speed = 1.

This has the same local ratio p/w = exp(-2A) as the Randall–Sundrum operator but a different derivative structure and measure. At kL=4 it predicts first three positive masses 0.0526014, 0.1120493 and 0.1713150. The first mass differs by -25.1%.

A laboratory helium-3 analogue would need to reproduce an appropriate weighted wave operator, boundary conditions, and quasiparticle branches. A locally matched velocity profile alone is insufficient. Helium-3 superfluid transition temperatures are in millikelvin, around 2.3 mK depending on pressure; the 2.3 meV fourth-root of cosmic dark-energy density is a distinct quantity.

## 3. Backreacting five-dimensional scalar wall

Use the bulk action

    S = integral d5x sqrt(-G) [ M5³ R/2 - (partial chi)²/2 - V(chi) ].

With a de Sitter-sliced metric

    ds² = exp(-2A(y))[-dt² + exp(2Ht) dvecx²] + dy²,

the Einstein–scalar bulk equations include

    A'' = H² exp(2A) + chi'²/(3 M5³)
    6 M5³ (A'² - H² exp(2A)) = chi'²/2 - V
    chi'' - 4A' chi' = V_chi.

An exactly soluble H=0 background follows from

    W(chi)=3 M5³ k + u chi²/2,
    V= W_chi²/2 - 2 W²/(3 M5³),
    chi(y)=chi0 exp(uy),
    A(y)=ky + chi0² (exp(2uy)-1)/(12 M5³).

For M5³=k=1, L=4, u=0.25, chi0=0.4, the first mode decreases to 0.0670740, a shift of -4.53%. This background solves all three bulk field equations with analytical residuals at floating-point precision.

A collocation boundary-value solver enforces A(0)=0, chi(0)=chi0, chi(L)=chi0 exp(uL) and the UV Einstein constraint, then solves for the derivatives. With H=0.005, its maximum sampled Einstein-constraint residual is about 3.4e-12. A naive initial-value shooting method was ill-conditioned because its growing scalar solution amplifies tiny UV mismatches.

The bulk solution requires boundary-localized actions consistent with the resulting Israel junction conditions. Those brane terms are not constructed here. A y-dependent scalar amplitude alone is also not a conventional finite-density superfluid, and the de Sitter background has not undergone a complete perturbative stability analysis.

## 4. Finite-density superfluid obstruction

Take a homogeneous phase theta=mu*t and X=-G^AB partial_A theta partial_B theta/2. For P(X),

    T^t_t - T^i_i = -2 X P_X.

An exactly four-dimensionally Lorentz-invariant static warp would require this anisotropy to vanish, implying X P_X=0 for a phase-only finite-density source. A conventional fluid with X>0 and P_X>0 cannot be inserted into that ansatz as the only additional matter source. This restricts the static ansatz and motivates genuinely time-dependent Einstein–fluid calculations; it does not rule out every superfluid cosmology.

## 5. Four-dimensional k-essence baseline

Take P(X)=-V0+alpha X², with alpha>0, X>0 and conserved shift charge. The benchmark computes

    P_X = 2 alpha X
    P_X + 2X P_XX = 6 alpha X > 0
    cs² = 1/3
    X(z) = X0 (1+z)²
    rho(z) = V0 + 3 alpha X0² (1+z)^4
    p(z) = -V0 + alpha X0² (1+z)^4.

These positive fluctuation coefficients are necessary local stability checks. The evolving fluid energy scales like radiation; V0 is an input. For alpha X0²/V0 = 0.001, w(0)=-0.9960, w(1)=-0.9389 and w(3)=-0.4208. An observational analysis must account for the accompanying dark-radiation component and determine a mechanism for the vacuum term.

## Verified test gates

1. RS Bessel root asymptotics at kL=35.
2. Unwarped Neumann zero mode and second-order FE convergence.
3. Warped Neumann convergence to the exact Bessel spectrum.
4. IR-mode weighted localization.
5. Difference between equal-local-speed and full RS operators.
6. Exact scalar-wall and H=0 numerical Einstein constraints.
7. Expanding dS-slice scalar-wall constraint and H-to-zero limit.
8. Fluid stress mismatch, conserved charge scaling and cs² positivity.

## Open investigations

* Solve the helium-3 A-phase Bogoliubov–de Gennes/Andreev quasiparticle eigenproblem for a concrete experimental order-parameter texture. Compare its weighted spectral data with RS, including tetrad gradients, norm, density of states and boundary constraints.
* Add a complex five-dimensional condensate Psi=chi exp(i theta) with nonzero charge density and a time-dependent metric, solve the coupled Einstein–matter system, and check ghost/gradient modes.
* Specify both brane potentials and radion stabilization, then compute physical normalizations and scalar fifth-force constraints.
* Derive the observed dark-energy density dynamically and fit expansion, structure formation and gravitational-wave constraints with fully declared priors.

## Relevant literature

- Lisa Randall and Raman Sundrum, "A Large Mass Hierarchy from a Small Extra Dimension," Phys. Rev. Lett. 83 (1999), https://doi.org/10.1103/PhysRevLett.83.3370
- Walter Goldberger and Mark Wise, "Modulus Stabilization with Bulk Fields," Phys. Rev. Lett. 83 (1999), https://doi.org/10.1103/PhysRevLett.83.4922
- Grigory Volovik, "The Universe in a Helium Droplet," Oxford University Press, https://doi.org/10.1093/acprof:oso/9780199564842.001.0001
- Carlos Barceló and Antonio Campos, "Braneworld physics from the analog-gravity perspective," https://arxiv.org/abs/hep-th/0206217
