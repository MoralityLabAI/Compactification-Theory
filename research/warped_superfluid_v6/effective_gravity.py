"""WS-06: 4D *specified* radion + U(1) scalar effective GR action.

The dimensional reduction is phenomenological: the radion mass is calibrated
from WS-04, while its normalization and cross coupling are parameters.  This
is NOT claimed to be a full 5D reduction.  For canonical Cartesian scalars
phi=(r,x,v), 3D Fourier mode p, spatially flat gauge defines the Mukhanov-
Sasaki variables Q^I=delta_phi^I + (dot_phi^I/H)*psi.

After scalar lapse and shift are eliminated via the Einstein constraints:
 D_t² Q + 3H D_t Q + (p²/a² I + M)Q=0,
 M=Hess(V) -(1/a³ Mpl²) d/dt[a³ dot(phi) dot(phi)^T / H].
The field-space metric here is exactly delta_ij; the kinetic/gradient forms
are positive.  Canonical momenta Pi=a³ Q_dot produce a symplectic flow.

This explicitly includes 4D metric perturbations (the gravitational mass
correction). It does NOT include 5D metric KK modes or perturbed 5D Israel
conditions; those remain the next necessary milestone for complete stability.
"""
from dataclasses import dataclass, replace
from scipy.integrate import solve_ivp
import numpy as np

@dataclass(frozen=True)
class EFTParams:
    Mpl2:float=(1-np.exp(-2.0))
    mass_rad2:float=0.0011806541053916162
    mass_bulk2:float=1.1625933720313903
    lam:float=0.3
    g:float=0.001
    gamma:float=0.002
    beta:float=0.001
    V0:float=0.40
    def __post_init__(self):
        if min(self.Mpl2,self.mass_rad2,self.mass_bulk2,self.lam,self.beta,self.V0)<=0:
            raise ValueError('positive gravity, masses and quartics required')


def potential(phi,par=EFTParams()):
    r,x,y=np.asarray(phi)
    F2=x*x+y*y
    return (par.V0+.5*par.mass_rad2*r*r+par.beta*r**4/4
            +.5*par.mass_bulk2*F2+par.lam*F2*F2/4
            +.5*par.g*r*F2+.25*par.gamma*r*r*F2)


def gradient_hessian(phi,par=EFTParams()):
    r,x,y=np.asarray(phi); f2=x*x+y*y
    radial=par.mass_bulk2+par.lam*f2+par.g*r+.5*par.gamma*r*r
    grad=np.array([par.mass_rad2*r+par.beta*r**3+.5*(par.g+par.gamma*r)*f2,
                   radial*x,radial*y])
    hess=np.array([[par.mass_rad2+3*par.beta*r*r+.5*par.gamma*f2,
                    (par.g+par.gamma*r)*x,(par.g+par.gamma*r)*y],
                   [(par.g+par.gamma*r)*x,radial+2*par.lam*x*x,2*par.lam*x*y],
                   [(par.g+par.gamma*r)*y,2*par.lam*x*y,radial+2*par.lam*y*y]])
    return grad,hess


def initial_conditions(par=EFTParams(),amplitude=.32,omega=1.092548):
    r=-par.g*amplitude*amplitude/(2*(par.mass_rad2+.5*par.gamma*amplitude**2))
    phi=np.array([r,amplitude,0.])
    velocity=np.array([0.,0.,omega*amplitude])
    rho=.5*(velocity@velocity)+potential(phi,par)
    H=np.sqrt(rho/(3*par.Mpl2))
    return np.r_[phi,velocity,1.,H]


def background_rhs(t,z,par):
    phi=z[:3];vel=z[3:6];a,H=z[6:8]
    grad,_=gradient_hessian(phi,par)
    return np.r_[vel,-3*H*vel-grad,a*H,-vel@vel/(2*par.Mpl2)]


def friedmann_constraint(z,par):
    phi=z[:3];v=z[3:6];H=z[7]
    return 3*par.Mpl2*H*H-.5*v@v-potential(phi,par)


def mass_matrix(z,par=EFTParams(),gravity=True):
    """Time-dependent symmetric, gauge-invariant 4D scalar mass matrix."""
    phi=z[:3];v=z[3:6];H=z[7]
    if H<=0: raise ValueError('expanding branch only')
    grad,hess=gradient_hessian(phi,par)
    if not gravity: return hess
    dv=-3*H*v-grad
    Hd=-v@v/(2*par.Mpl2)
    term=((3-Hd/(H*H))*np.outer(v,v)
          +(np.outer(dv,v)+np.outer(v,dv))/H)/par.Mpl2
    return hess-term


def gauge_invariant_Q(delta_phi,psi,phi_dot,H):
    return np.asarray(delta_phi)+np.asarray(phi_dot)*psi/H


def gauge_time_shift(delta_phi,psi,phi_dot,H,T):
    """Under t->t+T: delta_phi->delta_phi-phi_dot*T; psi->psi+H*T."""
    return np.asarray(delta_phi)-np.asarray(phi_dot)*T,psi+H*T


def transfer(p3,par=EFTParams(),tend=0.65,samples=18,numeric_tolerance=2e-10):
    """Coupled finite-momentum scalar metric + radion + charged field evolution.

    State (background phi,v,a,H; fundamental matrix U=(Q,Pi) where Pi=a³ Qdot).
    Hamiltonian propagator U obeys U.T J U = J.  This is a gauge-invariant
    4D EFT statement, not a 5D perturbation spectrum.
    """
    if p3<0:raise ValueError('nonnegative comoving p3 required')
    start=initial_conditions(par)
    T0=np.eye(6)
    def rhs(t,state):
        bg=state[:8]
        phi,v,a,H=bg[:3],bg[3:6],bg[6],bg[7]
        m=mass_matrix(bg,par)
        eye=np.eye(3)
        G=np.block([[np.zeros((3,3)),eye/a**3],
                    [-a**3*(m+(p3*p3/a**2)*eye),np.zeros((3,3))]])
        U=state[8:].reshape(6,6)
        return np.r_[background_rhs(t,bg,par), (G@U).ravel()]
    ts=np.linspace(0,tend,samples)
    sol=solve_ivp(rhs,(0,tend),np.r_[start,T0.ravel()],t_eval=ts,
                  method='DOP853',rtol=numeric_tolerance,atol=2e-12,
                  max_step=.015)
    if not sol.success: raise RuntimeError(sol.message)
    bg=sol.y[:8].T; U=sol.y[8:].T.reshape(-1,6,6)
    J=np.block([[np.zeros((3,3)),np.eye(3)],[-np.eye(3),np.zeros((3,3))]])
    symplectic=[np.linalg.norm(u.T@J@u-J,ord=np.inf) for u in U]
    constraints=[abs(friedmann_constraint(z,par)) for z in bg]
    rho_charge=[z[6]**3 * (z[1]*z[5]-z[2]*z[4]) for z in bg]
    coupling=[float(np.linalg.norm(mass_matrix(z,par)-np.diag(np.diag(mass_matrix(z,par))))) for z in bg]
    # Finite-time sensitivity from an initial phase velocity Pi_y to radion Q_r.
    mixing=[float(u[0,5]) for u in U]
    return {'time':ts,'background':bg,'fundamental':U,
            'max_symplectic_residual':float(max(symplectic)),
            'max_Friedmann_error':float(max(constraints)),
            'charge_drift':float(max(abs(np.asarray(rho_charge)/rho_charge[0]-1))),
            'mass_mixing':np.asarray(coupling),'response_radion_from_phase_pi':np.asarray(mixing),
            'growth_singular_values':np.linalg.svd(U[-1],compute_uv=False),
            'instantaneous_mass_eigen':np.array([np.linalg.eigvalsh(mass_matrix(z,par)) for z in bg])}


def eft_from_probe(params,solution,radion_m2=0.0011806541053916162,g=0.001):
    """Leading zero-mode KK truncation, radion cross-coupling phenomenological.

    The charged four-dimensional field uses the normalized *linear* lowest
    eigenfunction.  m_B²=lambda_0 and lam_4=lam_5*int e^-4A u_0^4.
    Z2 orbifold gives Mpl²=M5³*(1-exp(-2kL))/k, taking M5³=1.
    The radion mass comes from the WS-04 uncharged GI calculation; its
    normalization and r|Psi|² coupling are NOT derived from 5D.
    """
    from bulk_probe import assemble,weighted_nonlinear
    from scipy.linalg import eigh
    y,K0,M,cells=assemble(params,len(solution['F']))
    vals,vectors=eigh(K0,M,subset_by_index=[0,0])
    normalized=vectors[:,0]
    C,T=weighted_nonlinear(normalized,cells,len(normalized))
    quartic=params.lam * (normalized@C)
    m4sq=(1-np.exp(-2*params.k*params.L))/params.k
    return EFTParams(Mpl2=float(m4sq),mass_rad2=float(radion_m2),
                     mass_bulk2=float(vals[0]),lam=float(quartic),g=g)
