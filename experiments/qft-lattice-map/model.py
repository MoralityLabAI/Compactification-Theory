"""CPU reference for Shao--Sopenko, arXiv:2611.05329v1, Dirac sector.

One-particle quantities are matrices in the flavor index n, not many-body
Fock-space Hamiltonians. All quantities use lattice spacing a=1.
"""
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre, fresnel


def flavors(m):
    if m <= 0 or m % 2:
        raise ValueError("m must be a positive even integer")
    return np.arange(-m // 2, m // 2, dtype=int)


def theta(u):
    u = np.asarray(u, dtype=float)
    return np.where(u <= .5, np.pi / 2 - np.pi*u*u,
                    np.pi*(1-u)*(1-u))


def theta_prime(u):
    u = np.asarray(u, dtype=float)
    return np.where(u <= .5, -2*np.pi*u, -2*np.pi*(1-u))


def w_kernel(d):
    """Eq. (2): W_d = sinc(d/4)^2 (numpy's sinc uses sin(pi*x))."""
    return np.sinc(np.asarray(d)/4.)**2


def hamiltonian(k, m):
    """Eq. (3), restricted to -M/2 <= n < M/2; Hermitian M x M."""
    n = flavors(m)
    row = n[:, None]
    d = n[None, :] - row
    parity = np.where((d % 2) == 0, np.cos(k), 1j*np.sin(k))
    h = np.diag(np.pi*(n+.5)).astype(complex)
    h -= np.pi/2 * np.where((row % 2) == 0, 1, -1)*w_kernel(d)*parity
    return h


def two_flavor_energy(k):
    """Sec. II.C analytic dispersion E_+(k) for M=2."""
    return float(np.hypot(np.pi/2*(1-np.cos(k)), 4/np.pi*np.sin(k)))


@lru_cache(maxsize=4)
def quadrature(order=256):
    nodes, weights = roots_legendre(order)
    return .5*(nodes+1), .5*weights


def w_integral(d, order=128):
    """Independent check of Eqs. (10)-(11); split at theta'' kink u=1/2."""
    u, wt = quadrature(order)
    total=0j
    for offset in (0., .5):
        x = .5*u+offset
        total += np.sum((wt/2)*theta_prime(x)*np.exp(1j*np.pi*d*(x-.5)))
    return -2/np.pi*total


@lru_cache(maxsize=8)
def axial_coefficients(max_difference, order=256):
    """Eqs. (18)-(19), C_d odd and S_d even under d -> -d."""
    u, wt = quadrature(order)
    d = np.arange(-max_difference, max_difference+1)
    phase = np.pi*d[:, None]*(u[None, :]-0.5)
    c = (np.sin(phase)*np.cos(2*theta(u))[None, :]) @ wt
    s = (np.cos(phase)*np.sin(2*theta(u))[None, :]) @ wt
    return c, s


def axial_charge(k, m):
    """Single-particle axial generator from Eq. (18), in momentum basis.

    In this convention c_j = L^-1/2 sum_k exp(i k j) c_k.
    """
    n = flavors(m)
    d = n[None, :] - n[:, None]
    cvals, svals = axial_coefficients(m-1)
    c = cvals[d + m-1] * ((d % 2) != 0)
    s = svals[d + m-1]
    sign_row = np.where((n[:, None] % 2) == 0, 1, -1)
    sign_col = np.where((n[None, :] % 2) == 0, 1, -1)
    return 1j*(c + s/2*(sign_col*np.exp(-1j*k)-sign_row*np.exp(1j*k)))


def commutator(a,b):
    return a@b-b@a


def central_indices(m_large, m_small):
    """Subset in a flavor-ordered larger matrix for centered small cutoff."""
    if m_small > m_large or m_small % 2:
        raise ValueError("invalid nested cutoffs")
    start = (m_large-m_small)//2
    return np.arange(start, start+m_small)


def projection_boundary_identity(k, m, large_m):
    """Finite-matrix exact identity with the P[H,Q]P remainder exposed.

    [PHP,PQP] = P[H,Q]P + P Q(1-P)H P - P H(1-P)Q P.
    """
    H = hamiltonian(k, large_m)
    Q = axial_charge(k, large_m)
    keep = central_indices(large_m, m)
    out = np.setdiff1d(np.arange(large_m), keep)
    A, B = H[np.ix_(keep, keep)], Q[np.ix_(keep, keep)]
    lhs = commutator(A, B)
    full_comm = commutator(H,Q)[np.ix_(keep, keep)]
    boundary = (Q[np.ix_(keep, out)]@H[np.ix_(out, keep)]
                - H[np.ix_(keep, out)]@Q[np.ix_(out, keep)])
    return float(np.linalg.norm(lhs - full_comm - boundary)), float(np.linalg.norm(boundary)), float(np.linalg.norm(full_comm))


def resolvent_comparison(k, m, large_m=50, z=.3+.7j):
    """Finite-reference compressed resolvent, plus rigorous Schur bound.

    Reference has K finite modes. These results are *not* certified limits
    to an infinite-dimensional operator.
    """
    H = hamiltonian(k, large_m)
    keep = central_indices(large_m, m)
    out = np.setdiff1d(np.arange(large_m), keep)
    A = H[np.ix_(keep, keep)]
    B = H[np.ix_(keep, out)]
    D = H[np.ix_(out, out)]
    RM = np.linalg.inv(A-z*np.eye(m))
    RD = np.linalg.inv(D-z*np.eye(len(out)))
    G = np.linalg.inv(H-z*np.eye(large_m))[np.ix_(keep, keep)]
    difference = G-RM
    # G = RM + RM B RD B* G (Schur complement)
    rhs = RM @ B @ RD @ B.conj().T @ G
    upper_bound = (np.linalg.norm(RM,2)*np.linalg.norm(B,2)**2
                   *np.linalg.norm(RD,2)*np.linalg.norm(G,2))
    return dict(error=float(np.linalg.norm(difference,2)),
                schur_residual=float(np.linalg.norm(difference-rhs,2)),
                bound=float(upper_bound))


def b_quadrature(z, order=256):
    """Eq. (16) integral, reference quadrature for moderate frequencies."""
    u, w = quadrature(order)
    half = .5*u
    hw = .5*w
    return 2*((np.cos(np.pi*half**2)*hw) @ np.cos(np.outer(half, z))
              + (np.sin(np.pi*half**2)*hw) @
              np.cos(np.outer(1-half, z)))


def b_function(z):
    """Eq. (16), evaluated using Fresnel integrals for oscillatory tails.

    With J_+/- = int_0^1/2 exp(i*pi*u^2 +/- i*z*u)du,
    b(z)/2 = Re((J_+ + J_-)/2) +
             Im(cos(z)*(J_+ + J_-)/2 + sin(z)*(J_+ - J_-)/(2i)).
    """
    z=np.asarray(z,dtype=float)
    def F(t):
        s,c=fresnel(np.sqrt(2)*t)
        return (c+1j*s)/np.sqrt(2)
    shift=z/(2*np.pi)
    phase=np.exp(-1j*z*z/(4*np.pi))
    plus=phase*(F(.5+shift)-F(shift))
    minus=phase*(F(.5-shift)-F(-shift))
    cosint=(plus+minus)/2
    sinint=(plus-minus)/(2j)
    return 2*(cosint.real+(np.cos(z)*cosint+np.sin(z)*sinint).imag)


def correlation_matrix(L, ell, m, qmax=2048):
    """Eq. (16) exact-vacuum restriction, with explicit k>0 cutoff.

    qmax is number of positive continuum momentum values, not max momentum.
    Check qmax convergence; b(z) is evaluated using Fresnel functions.
    """
    if not 0 < ell < L:
        raise ValueError("expected proper nonempty interval")
    n=flavors(m)
    k=2*np.pi/L*(np.arange(qmax)+.5)
    b=b_function((k[:,None]+np.pi*(n[None,:]+.5)).ravel()).reshape(qmax,m)
    # Hermitian C_{(j,n),(j',m)} = real weighted cosine sum.
    phase_flavor=np.pi/2*(n[:,None]-n[None,:])
    result=np.empty((ell*m,ell*m),float)
    for j in range(ell):
        for jp in range(ell):
            weights = np.cos(phase_flavor[None,:,:]+k[:,None,None]*(j-jp))
            result[j*m:(j+1)*m,jp*m:(jp+1)*m]=np.einsum('kn,km,knm->nm',b,b,weights,optimize=True)/L
    return (result+result.T)/2


def entropy_from_correlation(C):
    eigen=np.linalg.eigvalsh(C)
    if np.min(eigen)<-3e-5 or np.max(eigen)>1+3e-5:
        raise ValueError(f"correlation not physical: {eigen.min()}, {eigen.max()}; increase qmax/order")
    v=np.clip(eigen,1e-15,1-1e-15)
    return float(np.sum(-v*np.log(v)-(1-v)*np.log(1-v)))


def cft_entropy(L, ell, c0=.987):
    return float(np.log(L/np.pi*np.sin(np.pi*ell/L))/3+c0)
