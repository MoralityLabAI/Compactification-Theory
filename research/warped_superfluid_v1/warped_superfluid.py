"""Dimensionless warped-mode and 5D Einstein-scalar benchmark.

Research prototype.  In particular a static 5D scalar profile is a
condensate-*amplitude* analogue; it is NOT a finite-density superfluid phase.

Conventions: ds^2=exp(-2A(y))[-dt^2+exp(2Ht)dx^2]+dy^2,
S_bulk = integral sqrt(-G) [M5^3 R/2 - (partial chi)^2/2 - V(chi)].
RS spin-2 / massless-scalar transverse Sturm-Liouville operator (H=0):
  -(exp(-4A)f')' = m^2 exp(-2A) f,  f'(0)=f'(L)=0.
The H>0 background solver does not purport to compute a dS spin-2 spectrum.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import j1, y1, jv, yv, jn_zeros

Array = NDArray[np.float64]


def rs_warp(y: Array, k: float = 1.0) -> Array:
    return k * np.asarray(y, dtype=float)


def bessel_condition(x: float, kL: float) -> float:
    """Bessel root formula for positive RS masses, scaled x=m exp(kL)/k.

    UV Neumann condition determines B/A = -J1(x e^-kL)/Y1(x e^-kL).
    Values below the first UV Y1 zero avoid denominator poles.
    """
    a = x * np.exp(-kL)
    return float(j1(x) - j1(a) / y1(a) * y1(x))


def rs_exact_scaled_roots(kL: float, count: int = 4) -> Array:
    if kL <= 0:
        raise ValueError("kL must be positive; use Neumann cosine spectrum at k=0")
    # Sample sign brackets.  The first few modes must remain below the first
    # UV Y1 zero, or the ratio chart becomes singular.  For kL >= 3 this
    # comfortably covers the first four KK masses.
    max_x = max(22.0, (count + 3) * np.pi)
    if max_x * np.exp(-kL) >= 2.0:
        raise ValueError("This root scan expects kL >= log(max_x / 2)")
    grid = np.linspace(1.0e-5, max_x, 12000)
    vals = np.array([bessel_condition(float(x), kL) for x in grid])
    roots: list[float] = []
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            root = brentq(bessel_condition, grid[i], grid[i + 1], args=(kL,), xtol=2e-13)
            if not roots or abs(root - roots[-1]) > 1e-6:
                roots.append(root)
        if len(roots) == count:
            break
    if len(roots) < count:
        raise RuntimeError("Insufficient separated positive roots")
    return np.array(roots)


def fe_masses(y: Array, A: Array, count: int = 4, *, naive_speed_only: bool = False) -> Array:
    """P1 finite-element Neumann Sturm-Liouville masses.

    Consistent FE mass matrix, two-point Gauss quadrature, symmetric K/M.
    naive_speed_only uses p=exp(-2A), w=1, which matches p/w locally
    while omitting the RS measure and drift.  It is an intentional control.
    """
    y = np.asarray(y, dtype=float)
    A = np.asarray(A, dtype=float)
    if len(y) < count + 4 or len(y) != len(A) or np.any(np.diff(y) <= 0):
        raise ValueError("Need matching increasing arrays and enough nodes")
    n = len(y)
    K = np.zeros((n, n), dtype=float)
    M = np.zeros((n, n), dtype=float)
    # Interpolate A at Gauss nodes.  For A=ky this is exact.
    for i in range(n - 1):
        h = y[i + 1] - y[i]
        for xi in [0.5 - np.sqrt(3.0) / 6, 0.5 + np.sqrt(3.0) / 6]:
            aq = (1 - xi) * A[i] + xi * A[i + 1]
            p = np.exp(-2 * aq) if naive_speed_only else np.exp(-4 * aq)
            w = 1.0 if naive_speed_only else np.exp(-2 * aq)
            N = np.array([1.0 - xi, xi])
            dN = np.array([-1.0 / h, 1.0 / h])
            inds = np.ix_([i, i + 1], [i, i + 1])
            K[inds] += (0.5 * h) * p * np.outer(dN, dN)
            M[inds] += (0.5 * h) * w * np.outer(N, N)
    vals = eigh(K, M, eigvals_only=True, subset_by_index=[0, count],
                driver="gvx", check_finite=True)
    vals[0] = 0.0 if abs(vals[0]) < 2e-7 * vals[-1] else vals[0]
    if np.any(vals < -1e-10):
        raise ValueError("Unexpected negative eigenvalue")
    return np.sqrt(np.maximum(0.0, vals))


def rs_mode_ir_fraction(kL: float, root: float, t_threshold: float = 0.1) -> float:
    """Weighted mode norm in the last ln(1/t_threshold) curvature lengths.

    t=exp[k(y-L)], f=t^2 [J2(root*t)+b Y2(root*t)];
    measure exp(-2ky)dy is proportional to t^-3 dt.
    """
    t0 = np.exp(-kL)
    t = np.geomspace(t0, 1.0, 12000)
    beta = -j1(root * t0) / y1(root * t0)
    f = t * t * (jv(2, root * t) + beta * yv(2, root * t))
    density = t**(-3) * f**2
    total = np.trapezoid(density, t)
    last = np.trapezoid(density[t >= t_threshold], t[t >= t_threshold])
    return float(last / total)


@dataclass(frozen=True)
class DomainWall:
    """Exact H=0 superpotential wall; M3 denotes M5**3.

    W=3 M3 k + u chi^2/2;  chi'=W_chi=u chi; A'=W/(3 M3)
    V=W_chi**2/2 - (2/(3 M3))W**2.
    Brane boundary conditions must be chosen consistently with this profile.
    """
    k: float = 1.0
    u: float = 0.25
    chi0: float = 0.4
    M3: float = 1.0

    def chi(self, y: Array) -> Array:
        return self.chi0 * np.exp(self.u * np.asarray(y, dtype=float))

    def W(self, chi: Array) -> Array:
        return 3 * self.M3 * self.k + self.u * np.asarray(chi)**2 / 2

    def V(self, chi: Array) -> Array:
        chi = np.asarray(chi)
        return 0.5 * (self.u * chi)**2 - 2 * self.W(chi)**2 / (3 * self.M3)

    def dV(self, chi: Array) -> Array:
        chi = np.asarray(chi)
        return self.u**2 * chi - 4 * self.W(chi) * self.u * chi / (3 * self.M3)

    def warp(self, y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        if abs(self.u) < 1e-12:
            return self.k * y
        return self.k*y + self.chi0**2 * np.expm1(2*self.u*y) / (12*self.M3)

    def exact_residual(self, y: Array) -> tuple[float, float, float]:
        chi = self.chi(y)
        dchi = self.u * chi
        ddchi = self.u**2 * chi
        Ap = self.W(chi)/(3*self.M3)
        App = dchi**2/(3*self.M3)
        constraint = 6*self.M3*Ap**2 - 0.5*dchi**2 + self.V(chi)
        yy_minus_ii = App - dchi**2/(3*self.M3)
        scalar = ddchi - 4*Ap*dchi - self.dV(chi)
        return (float(np.max(np.abs(constraint))),
                float(np.max(np.abs(yy_minus_ii))),
                float(np.max(np.abs(scalar))))


def integrate_ds_wall(wall: DomainWall, L: float, H: float, n: int = 321):
    """Solve the 5D Einstein-scalar *boundary-value* problem with dS4 slices.

    Metric exp(-2A)[-dt^2+exp(2Ht)dx^2]+dy^2.  UV A(0)=0,
    chi(0)=chi0, IR chi(L)=chi0 exp(uL), and UV yy Einstein constraint.
    The resulting A'(0), A'(L), chi' values specify junction data;
    brane potentials/tensions realizing those data are a separate requirement.
    Shooting is unreliable because perturbations excite a growing solution.
    """
    ys = np.linspace(0, L, n)
    c0 = wall.chi(ys)
    A0 = wall.warp(ys)
    initial = np.vstack([A0, wall.W(c0)/(3*wall.M3), c0, wall.u*c0])
    chiL = float(wall.chi(np.array([L]))[0])

    def rhs(y, state):
        A, Ap, chi, cp = state
        curve = H*H * np.exp(2*A)
        return np.vstack([Ap, curve + cp**2/(3*wall.M3),
                          cp, 4*Ap*cp + wall.dV(chi)])

    def bc(left, right):
        A, Ap, chi, cp = left
        yy_constraint = (6*wall.M3*(Ap*Ap-H*H)
                         - 0.5*cp*cp + float(wall.V(chi)))
        return np.array([A, chi-wall.chi0, right[2]-chiL, yy_constraint])

    sol = solve_bvp(rhs, bc, ys, initial, tol=3e-10, max_nodes=12000)
    if not sol.success:
        raise RuntimeError(sol.message)
    state = sol.sol(ys)
    A, Ap, chi, cp = state
    residual = (6*wall.M3*(Ap*Ap - H*H*np.exp(2*A))
                - 0.5*cp*cp + wall.V(chi))
    return ys, state, float(np.max(np.abs(residual)))


def ads_ds_warp(y: Array, H: float, k: float = 1.0) -> Array:
    """Exact dS4 slices of pure AdS5; A(0)=0, H>0, y<y_horizon."""
    if H <= 0 or k <= 0:
        raise ValueError("H,k must be positive")
    horizon = np.arcsinh(k/H)/k
    y = np.asarray(y, dtype=float)
    if np.any(y >= horizon):
        raise ValueError("Horizon must lie beyond the compact interval")
    return -np.log((H/k)*np.sinh(k*(horizon-y)))


def finite_density_anisotropy(X: float, alpha: float = 1.0) -> float:
    """T^t_t-T^i_i=-2 X P_X for P=-V0+alpha X^2."""
    return -4 * alpha * X*X


def kessence_fluid(z: Array, r0: float = 1e-3, V0: float = 1.0):
    """Shift-symmetric P=-V0+alpha X^2; r0=alpha X0^2/V0.

    Shift charge conservation implies X~a^-2 and kinetic rho~a^-4.
    No effective source Q or radion dynamics is included.
    """
    z = np.asarray(z, dtype=float)
    q = r0 * (1 + z)**4
    rho = V0 * (1 + 3*q)
    p = V0 * (-1 + q)
    w = p/rho
    cs2 = np.full_like(z, 1/3)
    return rho, p, w, cs2
