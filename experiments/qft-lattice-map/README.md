# QFT/Lattice exact-map CPU replication (Dirac sector)

Shao and Sopenko, *An Exact QFT/Lattice Map for Fermions*,
[arXiv:2611.05329v1](https://arxiv.org/abs/2611.05329), November 2026.

This is an independent, lightweight numerical reproduction of the **free,
non-chiral 1+1D Dirac one-particle sector**. The paper's exact map uses an
infinite flavor tower per site. Here every matrix is explicitly truncated.
No numerical result in this folder establishes exactness at finite M. The
Schwinger gauge theory, the many-body anomaly, the type-I/type-III split
property and an infinite-dimensional operator theorem are out of scope.

## Run

Requires Python 3.11+, `numpy`, `scipy` (no GPU, no network, no torch):

```bash
python -m pip install numpy scipy
python -m unittest discover -s experiments/qft-lattice-map/tests -v
python experiments/qft-lattice-map/run.py --output experiments/qft-lattice-map/results/reference.json
```

The committed JSON is the direct CPU output, using complex128 and the fixed
inputs in `run.py`. Default runtime is seconds on a laptop CPU. Reproducing
it requires no trained models or proprietary datasets.

## Reproduced definitions

- Eqs. (2)-(3): `hamiltonian(k,M)`, flavor range `-M/2,...,M/2-1`, and
  `W_d = sinc(d/4)^2`. Full Hermiticity is checked.
- Eqs. (10)-(11): independently integrate the *piecewise quadratic* rotation
  and its piecewise linear derivative. Split quadrature at `u=1/2` to avoid
  introducing a numerical kink error. The printed coupling passes the test.
- Sec. II.C: the exact analytic **M=2 finite-matrix** dispersion, the
  `M % 4` zero-mode distinction, and absence of extra zeros on sampled k.
- Eqs. (18)-(19): construct the **truncated axial charge**, testing both the
  full Frobenius commutator and the two-lowest-energy-state projection. The
  low-energy violation shrinks with M while boundary modes remain.
- Resolvent identity: compare `(PHP-z)^{-1}` with the projected resolvent of
  an explicitly **finite 50-flavor reference**. Check the Schur-complement
  identity and its operator-norm upper bound. Neither number is a certified
  bound against the infinite continuum resolvent.
- Eqs. (16)-(17): compute the **exact-vacuum restricted correlation matrix**
  using the continuum positive-momentum sum and Fresnel-evaluated `b(z)`.
  The entropy is *not* computed from the M-flavor Hamiltonian's ground state.
  Quantify the positive-momentum sum tail at 2048 vs 4096 terms.

## Fixed observations

For `k=0.7`, `z=0.3+0.7i`, `K=50` reference flavors:

| M | Error of two low bands vs ±k | `||P_low[H_M,Q_M]P_low||_F` | Finite-K resolvent error |
|--:|--:|--:|--:|
| 2 | 0.199580231 | 0.253779315 | 0.242488306 |
| 6 | 0.000000378 | 0.002983250 | 0.003067941 |
| 10 | 0.000217994 | 0.000185658 | 0.000790415 |
| 14 | 0.000005590 | 0.000033821 | 0.000138641 |

Spectral convergence is **not monotone** in M at this single momentum, even
though errors at larger cutoffs are small. The projected commutator tests the
low-energy two-band subspace; it is a diagnostic of approximate symmetry.

At L=64, interval ell=4, the exact-vacuum entropies at M=2,6,10,14 are
approximately `1.63061, 1.45049, 1.44771, 1.44728`. The paper's CFT
formula with its reported constant C=0.987 gives `1.44695`. At ell=1, the
constant-offset discrepancy is larger; data are in `results/reference.json`.
The largest 2048-to-4096-term entropy difference across the 12 registered
runs is `< 5.5e-9` with Fresnel arithmetic.

## Mathematical follow-up

For nested finite-dimensional projections P and a larger matrix pair (H,Q),
with R=I-P, the exact finite identity is

```
[PHP,PQP] = P[H,Q]P + P Q R H P - P H R Q P.
```

The boundary expression is a candidate compactification-error certificate,
provided the full commutator term and infinite-limit domain issues are
controlled. The 50-flavor Schur bound is a **finite comparator** only.
Next steps: analytic tail bounds for the Fourier kernel; interval enclosures
for low-band eigenvalues; and a Lean theorem with stated self-adjointness,
core/domain, and resolvent assumptions.

## Infinite-flavor resolvent extension (November 9, 2026)

[`THEORY.md`](THEORY.md) proves a uniform operator bound for the infinite
one-particle Hamiltonian of Eq. (3), including its self-adjoint domain.
For the centered M-flavor cutoff, the compressed-resolvent error is bounded
by `pi**2/(abs(Im(z))**2 * (pi*(M+1)/2 - abs(Re(z)) - pi))` whenever the
last parenthesis is positive. In particular, the infinite-flavor
**compressed-resolvent** approximation converges in operator norm as `O(1/M)`.

[`infinite_resolvent.py`](infinite_resolvent.py) adds more useful, quantitatively
tighter *exact-integer residual certificates* for the two central flavor
basis vectors at `k=pi/2, z=2i`. It uses rational enclosures of pi via the
Machin arctangent identity, fixed-point outward rounding, and an analytic
bound on the **infinite** uncomputed tail. The solver is used only to propose
candidate vectors. The certificate verifier accepts signed integer witness
entries and never calls the solver. The exact-scaled bounds and all witnesses
are committed as [`results/infinite_resolvent_certificate.json`](results/infinite_resolvent_certificate.json).

```bash
python experiments/qft-lattice-map/run_infinite_resolvent.py --verify \
  experiments/qft-lattice-map/results/infinite_resolvent_certificate.json
```

The proof of the mathematical inequality appears in `THEORY.md`; the numeric
checker is exact-integer Python, with a separate formal Lean bridge tracking
the abstract Schur and residual norm steps.

## Adaptive infinite-resolvent tolerance gate

The infinite-flavor certificate can drive a **finite-flavor selection** with an
exact rational threshold. The scan enumerates an explicit list of even mode
counts and validates each witness by integer interval arithmetic plus an
analytic infinite tail. It reports the **first passing mode count in this
candidate grid**, without assuming monotonic errors or global minimality.

```bash
python experiments/qft-lattice-map/adaptive_certificate.py \
  --target 1/1000 --cutoff 1024 \
  --output /tmp/qft-adaptive-receipt.json
python experiments/qft-lattice-map/adaptive_certificate.py \
  --verify /tmp/qft-adaptive-receipt.json
```

At $k=\pi/2$, $z=2i$, and source modes $n=-1,0$, the validated upper
bounds for this grid and cutoff include:

| Flavors M | Infinity-certified error upper |
|--:|--:|
| 30 | 0.003498405 |
| 38 | 0.002444086 |
| 54 | 0.001443057 |
| 70 | 0.000981997 |

Therefore 70 is the first grid entry that passes the exact threshold
$1/1000$, among the entries evaluated. Replay validates the embedded witness
bytes without numerical inversion; GitHub Actions saves the complete
run receipt as an artifact. The error applies to the two chosen source columns,
rather than to the full compressed-resolvent operator norm.
