# WS-02: A charged brane, radion and helium-3 Weyl-texture benchmark

**Research status, 29 October 2026:** CPU-reproducible mathematical toy study, complementary to [WS-01](https://github.com/MoralityLabAI/Compactification-Theory/pull/6). The aim is to find where superfluid-motivated vacuum dynamics and warped five-dimensional geometry are mutually compatible, and where a superficially attractive analogy breaks mathematically. All numerical scales are **dimensionless** unless specified. No observation has been fitted; no claim of dark-energy self-tuning is made.

```bash
cd research/warped_superfluid_v2
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
```

The runner generates `results/summary.json`, `results/ws02_diagnostics.png`, and `results/gw_probe.png`. The eighteen reference tests check independent components of the 5D Einstein tensor, Israel boundary conditions, charge conservation, numerical evolution of the radion, Goldberger–Wise probe minima and the Hermiticity of the helium-3 Weyl analogue. The calculations use SciPy and NumPy on CPU.

## A. Exact 5D AdS single-brane cosmology with finite charge density

Take a pure anti-de Sitter five-dimensional bulk with curvature `k>0`, `M5³>0`, cosmological constant `Lambda5=-6k²` in the convention `G_AB+Lambda5 g_AB=0`, and a `Z₂`-symmetric brane of tuned positive tension `sigma=6kM5³`. A superfluid-inspired scalar phase `theta` lives **on the brane** with

```text
X = -(1/2) g^(mu nu) (∂mu theta)(∂nu theta)
P(X) = -V0 + alpha X²,  alpha>0, X>0.
Q = a0³ P_X theta_dot = alpha a0³ theta_dot³ = constant.
s = alpha X² = s0 a0^(-4)
rho = V0+3s;    p=-V0+s;
rho+p=4s;    cs² = P_X/(P_X+2 X P_XX) = 1/3.
```

Choose the `y=0` brane proper time (`a0=e^N`, `N=ln a0`). The exact `C=0`, spatially flat Randall–Sundrum cosmology satisfies

```text
H² = k²[(1+rho/sigma)²-1] = rho/(3M4²) + k²(rho/sigma)²,
M4² = M5³/k,
Hdot = -3k²(1+rho/sigma)(rho+p)/sigma,
rho_dot = -3H(rho+p).
```

The Gaussian-normal bulk metric is

```text
ds5² = -n(t,y)² dt² + A(t,y)² dvecx² + dy²,
D(t,y) = cosh(ky) - [1+rho(t)/sigma] sinh(k|y|),
A(t,y) = a0(t) D(t,y),
n(t,y) = cosh(ky) - [1-(2rho(t)+3p(t))/sigma] sinh(k|y|).
```

Use the patch where both `D>0` and `n>0`. The code returns all analytic partial derivatives with respect to the brane time and `y`. It checks all independent smooth-bulk equations `Gtt/n²=-6k²`, `Gyy=6k²`, `Gii/A²=6k²` and `Gty=0`, as well as the AdS first integral

```text
(A_y/A)² - (A_t/(n A))² = k².
```

For `k=M5³=1`, `V0=0.05`, `s0=0.02`, `sigma=6`, `0<=N<=3`, and `0<=y<=1.5`, the maximum absolute normalized Einstein-tensor residual is approximately `3.8e-14`. Charge conservation holds at ~`1.3e-15` relative precision. These are analytical-derivative arithmetic residuals, not numerical discretization convergence measures.

### The second-brane junction obstruction

At the ultraviolet brane, the Israel conditions are

```text
(A_y/A)|0 = -k(1+rho/sigma),
(n_y/n)|0 = -k[1-(2rho+3p)/sigma].
```

If we place another `Z₂` brane at **fixed** `y=L` with the usual opposite tuned tension `sigma_IR=-sigma`, the geometry demands additional matter on that brane. Its density and pressure are computed directly from the reversed-normal junction conditions:

```text
rho_IR = sigma[(A_y/(k A))|L + 1]
p_IR = [-sigma (n_y/(k n))|L - sigma - 2 rho_IR]/3.
```

The first equation simplifies exactly to

```text
rho_IR = -rho_UV exp(kL)/D(L).
```

With `L=1.5`, this predicts an initial `rho_IR=-2.677911743`, `p_IR=1.064026460`. The required IR fluid obeys its local conservation equation when evaluated in its own proper time. A fixed, empty IR brane fails the junctions in the presence of our UV charged fluid. The result identifies a source that stabilization, a moving brane, or additional bulk fields must provide. The **bulk here is pure AdS**; the charged condensate is restricted to the UV brane. A complete charged five-dimensional bulk-condensate solution is a future problem.

## B. Four-dimensional Einstein-frame stabilized-radion effective theory

To test the distinct stabilization mechanism without falsely asserting the exact 5D two-brane completion, use a canonical radion `r` and a Jordan-frame physical-brane metric `gJ_mu_nu = B(r)² gE_mu_nu`, with `B=exp(beta r/M4)`.

```text
S4 = ∫ d4x sqrt(-gE) [ M4² RE/2 - (∂r)²/2
     - mr² r²/2 + B(r)^4 P_J(XE/B(r)²) ],
P_J(XJ) = -V0 + alpha XJ².
```

The quartic phase kinetic term is conformal in four dimensions:

```text
B^4 P_J(XE/B²) = -V0 B^4 + alpha XE².
```

Consequently its positive-kinetic fluid has `cs²=1/3` and a separately conserved charge, with `alpha XE²=s0 aE^-4`; it **does not directly generate an evolving dark-energy density**. The radion is sourced by the vacuum part of the brane Lagrangian, giving

```text
Veff(r) = mr² r²/2 + V0 exp(4 beta r/M4),
rddot + 3H rdot + Veff_r = 0,
3M4² H² = rdot²/2 + Veff(r) + 3s0 aE^-4,
Hdot = -(rdot²+4s0 aE^-4)/(2M4²).
```

The stable equilibrium for `mr²>0` and `V0>0` is exactly

```text
r_min = -(M4/(4beta)) W0[16 beta² V0/(mr² M4²)]   (beta != 0),
Veff''(r_min) = mr² + (4beta/M4)² V0 exp(4beta r_min/M4) >0.
```

Here `W0` is the principal Lambert W function. For `M4=1`, `mr=0.8`, `beta=0.15`, `V0=0.05`, `s0=0.02`, the integrated radion tends to `r_min=-0.0456096224`; `Veff(r_min)=0.0493159399`, and the total cosmological equation of state tends to `-1`. An independent finite-difference check of Raychaudhuri and the radion evolution is included; the algebraic Friedmann constraint is enforced by construction.

**A small no-go in this model:** the envelope theorem gives

```text
d Veff(r_min; V0) / dV0 = exp(4 beta r_min/M4) > 0.
```

A positive bare vacuum term remains a positive effective vacuum energy at the minimum and varies with the bare input. Ordinary quadratic stabilization plus the conformal quartic phase cannot dynamically solve the cosmological-constant problem. This statement is specific to the declared action and assumptions.

### C. Goldberger–Wise local probe benchmark

For the dimensionless separation `z=kL`, the leading small-backreaction *schematic* Goldberger–Wise potential can be represented locally as

```text
U(z) = C exp(-4z) [vIR - vUV exp(-epsilon z)]²,
z* = ln(vUV/vIR)/epsilon.
```

The unit tests check positive local curvature at `z*=35` and `z*=69` with `epsilon=0.1`. The required ratios `vIR/vUV` are approximately `0.0301974` and `0.00100779` respectively. The potential has a local minimum at the designed separation, and the overall coefficient is not fixed. It becomes small in the infinite-radius direction, so **this probe potential by itself does not settle cosmological metastability or the vacuum-energy problem**. The full backreacted bulk scalar and radion-normalization calculation remains open. This probe is deliberately separate from the canonical effective radion potential in section B.

## D. Helium-3 A-phase quasiparticle ring analogue

Near a topologically protected Weyl node, quasiparticles in superfluid `3He-A` exhibit a low-energy Hamiltonian of the form

```text
H ~ tau_x c_perp(y) p_x + tau_y c_perp(y) p_y + tau_z c_parallel delta_p_z.
```

A spatially varying `c_perp` requires a symmetrized derivative to be Hermitian with respect to the flat `dy` inner product:

```text
H_y = -i tau_y [ v(y) ∂y + v'(y)/2 ]
    = -i tau_y sqrt(v) ∂y sqrt(v).
```

For a periodic ring, the finite difference uses the skew-adjoint derivative `D` and the exactly symmetric product `-i{diag(v),D}/2`. The chosen smooth periodic order-parameter speed profile is

```text
v(y) = exp[-depth (1-cos(2pi y/L))/2],  L=2pi.
```

At `depth=1`, the velocity minimum is approximately `0.36809`. For `delta_p_z=0.3`, `c_parallel=1.5`, `p_x=0`, and a 65-site grid, the first positive excitation above the constant `0.45` gap shifts from `0.67254` (uniform texture) to `0.53269` (spatial texture), a `-20.8%` change. The Hamiltonian has zero measured Hermiticity error, and its paired energy spectrum is symmetric to `2e-14` numerical accuracy for the stated parameter slice. The uniform-medium discrete cone is checked against an analytical lattice dispersion relation.

This is a **continuum Weyl/Andreev toy**, not a complete p-wave 3He BdG calculation. Its periodic ring is physically distinct from an RS interval with branes; a precise analogy would need matching spectral measures, boundaries, and microscopic order-parameter textures. The comparison helps identify false positives arising from matching propagation speeds alone.

## Verified acceptance gates

- Conserved superfluid phase charge and `a^-4` kinetic dilution.
- Modified brane Friedmann and acceleration identities, including the quadratic-density correction.
- Independent 5D Einstein `tt`, `yy`, `ii`, `ty` components and the first integral.
- UV spatial and temporal Israel junction conditions.
- IR second-brane mismatch and independent proper-time conservation.
- Empty-brane Randall–Sundrum limit and Gaussian-normal coordinate-patch guard.
- Conformal quartic identity, positivity of the radion effective mass, exact Lambert-W minimum and its sensitivity to bare vacuum energy.
- Numerical FRW/radion evolution with derivative-based error checks.
- Goldberger–Wise local probe minima at the two chosen warp scales.
- Weyl Hamiltonian Hermiticity, particle-hole paired spectrum on the stated slice, and the uniform lattice cone.

## Remaining hurdles

A physical two-brane completion must introduce an explicit bulk Goldberger–Wise field, impose both scalar boundary conditions and Israel junctions, allow a dynamic radion, and solve the joint Einstein–charged-matter PDE/BVP. A true bulk superfluid additionally needs a complex scalar `Psi=chi exp(i theta)` with a finite `U(1)` charge density, background solution and fluctuation analysis. Its spectra need ghost, gradient, tachyon, radion fifth-force and gravitational-wave checks. The helium-3 implementation next requires the experimentally meaningful p-wave BdG order-parameter matrix and geometry rather than the single-node Weyl approximation. Observationally, `H(z)`, supernova/BAO/CMB and growth of structure must be fitted before proposing a dark-energy model.

## Sources

1. Randall and Sundrum, *A Large Mass Hierarchy from a Small Extra Dimension*, PRL **83** (1999): https://doi.org/10.1103/PhysRevLett.83.3370
2. R. Maartens, *Brane-world gravity*, Living Reviews in Relativity (2004, subsequently updated): https://link.springer.com/article/10.12942/lrr-2004-7. See the exact gaussian-normal cosmology metric, equations 180–182.
3. Csáki, Graesser, Randall and Terning, *Cosmology of brane models with radion stabilization*, PRD **62** (2000): https://doi.org/10.1103/PhysRevD.62.045015
4. Cline and Firouzjahi, *Brane-world cosmology of modulus stabilization with a bulk scalar field*, PRD **64** (2001): https://doi.org/10.1103/PhysRevD.64.023505
5. Goldberger and Wise, *Modulus stabilization with bulk fields*, PRL **83** (1999): https://doi.org/10.1103/PhysRevLett.83.4922
6. G. E. Volovik, *Combined Lorentz Symmetry: Lessons from Superfluid 3He*, Journal of Low Temperature Physics (2021/2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC8770376/
7. Ishihara et al., *Torsional chiral magnetic effect due to skyrmion textures in a Weyl superfluid 3He-A*, PRB **99** (2019): https://doi.org/10.1103/PhysRevB.99.024513

## Repository coordination

WS-02 lives in its own directory and targets `main` in a separate PR. It changes no WS-01 files and depends on none of its unmerged modules. The two benchmarks may later share a common geometry layer once WS-01 has been accepted.