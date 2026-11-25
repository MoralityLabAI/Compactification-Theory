"""WS-07 static anisotropically warped 5D Einstein+GW+charged condensate.

Metric: ds^2=-exp(2N(y))dt^2+exp(2A(y)) dx_3^2+dy^2.
Matter: chi(y), Psi=F(y) exp(i omega t) with 1/2 real-field kinetic
and V=V_GW(chi)+m2 F²/2 + lam F⁴/4.

This is a static, *spatially anisotropic* 5D geometry, not a 4D
Lorentz-invariant RS metric; no claim is made that both branes have
ordinary NEC-satisfying matter.  The exact integrated anisotropy relation
exhibits the obstruction when both branes satisfy the null energy condition.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

@dataclass(frozen=True)
class Params:
    M3:float=1.;k:float=1.;u:float=.25;chi0:float=.4;L:float=1.
    F0:float=.08;kap_uv:float=1.;kap_ir:float=1.;mass2:float=.5;lam:float=.3
    def __post_init__(self):
        if min(self.M3,self.k,self.u,self.chi0,self.L,self.kap_uv,self.kap_ir)<=0:raise ValueError('positive gravity, geometry, and stiffness required')
        if self.F0<0 or self.lam<=0 or self.mass2<0:raise ValueError('canonical charged matter parameters required')


def superpotential(c,p):
    return 3*p.M3*p.k+0.5*p.u*c*c

def scalar_potential(c,f,p):
    W=superpotential(c,p)
    return .5*p.u*p.u*c*c - (2/(3*p.M3))*W*W + .5*p.mass2*f*f + .25*p.lam*f**4

def potential_derivatives(c,f,p):
    W=superpotential(c,p)
    return p.u*p.u*c -4*p.u*c*W/(3*p.M3),p.mass2*f+p.lam*f**3


def pressure_terms(state,omega,p):
    A,N,c,f,Ap,Np,cp,fp=np.asarray(state)
    R=cp*cp+fp*fp
    Q=np.exp(-2*N)*omega*omega*f*f
    V=scalar_potential(c,f,p)
    return R,Q,V


def odes(y,z,omega,p):
    A,N,c,f,Ap,Np,cp,fp=z
    R,Q,V=pressure_terms(z,omega,p)
    App=(-.5*(R+Q)-V)/(3*p.M3)-2*Ap*Ap
    D=Np-Ap
    Npp=App-(Np+3*Ap)*D+Q/p.M3
    Vc,Vf=potential_derivatives(c,f,p)
    cpp=Vc-(Np+3*Ap)*cp
    fpp=Vf-(Np+3*Ap)*fp-omega*omega*np.exp(-2*N)*f
    return np.array([Ap,Np,cp,fp,App,Npp,cpp,fpp])


def constraint(z,omega,p):
    A,N,c,f,Ap,Np,cp,fp=z
    R,Q,V=pressure_terms(z,omega,p)
    return 3*p.M3*Ap*(Ap+Np)-.5*(R+Q)+V


def initial_values(omega,p):
    c=p.chi0; f=p.F0;cp=p.u*c;fp=.5*p.kap_uv*f
    Q=omega*omega*f*f
    V=scalar_potential(c,f,p)
    rhs=(.5*(cp*cp+fp*fp+Q)-V)/(6*p.M3)
    if rhs<=0:raise ValueError('No real negative-warp UV constraint root')
    H=-np.sqrt(rhs)
    return np.array([0.,0.,c,f,H,H,cp,fp])


def integrate(omega,p,points=401,rtol=2e-11):
    initial=initial_values(omega,p)
    result=solve_ivp(lambda y,z:odes(y,z,omega,p),(0,p.L),initial,
                     dense_output=True,rtol=rtol,atol=rtol*.01,
                     method='DOP853',max_step=p.L/200)
    if not result.success: raise RuntimeError(result.message)
    yy=np.linspace(0,p.L,points)
    z=result.sol(yy)
    return yy,z


def ir_robin_mismatch(omega,p):
    _,z=integrate(omega,p,points=2,rtol=6e-10)
    return z[7,-1]+.5*p.kap_ir*z[3,-1]


def equilibrium(p=Params(),n=401):
    if p.F0<=0:raise ValueError('F0 must be positive for charged branch')
    # Positive mode nodeless root, robust bracket search on physical omega range.
    xs=np.linspace(.3,3.5,54)
    vals=[]
    for x in xs:
        try:val=ir_robin_mismatch(x,p)
        except (ValueError,RuntimeError,OverflowError,FloatingPointError): val=np.nan
        vals.append(val)
    candidates=[(xs[i],xs[i+1]) for i in range(len(xs)-1)
                if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1]<0]
    if not candidates:raise RuntimeError('No positive-frequency nodeless IR Robin root bracket found')
    omega=None;yy=z=None
    for lo,hi in candidates:
        root=root_scalar(ir_robin_mismatch,args=(p,),bracket=(lo,hi),xtol=1e-12)
        y,ztry=integrate(root.root,p,points=n)
        if np.min(ztry[3])>0:
            omega=float(root.root);yy=y;z=ztry;break
    if omega is None:raise RuntimeError('Could not find nodeless charged state')
    A,N,c,f,Ap,Np,cp,fp=z
    R,Q,V=pressure_terms(z,omega,p)
    D=Np-Ap
    vol=np.exp(N+3*A)
    source=(omega*omega/p.M3)*np.exp(-N+3*A)*f*f
    sumrule_lhs=float(vol[-1]*D[-1]-vol[0]*D[0])
    sumrule_rhs=float(np.trapezoid(source,yy))
    # With Z2 Israel signs: D(0)=(rho_uv+p_uv)/(2M3),
    # D(L)=-(rho_ir+p_ir)/(2M3).  UV NEC is saturated here.
    ir_nec=float(-2*p.M3*D[-1]);uv_nec=float(2*p.M3*D[0])
    all_constraints=np.array([constraint(z[:,i],omega,p) for i in range(n)])
    Dprime_from_eq=(omega*omega/p.M3)*np.exp(-2*N)*f*f -(Np+3*Ap)*D
    identities={
      'max_hamiltonian_constraint':float(np.max(abs(all_constraints))),
      'sumrule_lhs':sumrule_lhs,'sumrule_integral':sumrule_rhs,
      'sumrule_quad_error':float(abs(sumrule_lhs-sumrule_rhs)),
      'uv_D':float(D[0]),'ir_D':float(D[-1]),
      'uv_rho_plus_p':uv_nec,'ir_rho_plus_p':ir_nec,
      'bulk_charge':float(omega*np.trapezoid(np.exp(3*A-N)*f*f,yy)),
      'bulk_anisotropy_positive_min':float(np.min(Q)),
      'UV_F_robin':float(fp[0]-.5*p.kap_uv*f[0]),
      'IR_F_robin':float(fp[-1]+.5*p.kap_ir*f[-1]),
      'uv_chi_prime':float(cp[0]),'ir_chi_prime':float(cp[-1]),
      'uv_A_prime':float(Ap[0]),'uv_N_prime':float(Np[0]),
      'ir_A_prime':float(Ap[-1]),'ir_N_prime':float(Np[-1]),
      'D_rhs_min':float(np.min(Dprime_from_eq)),
    }
    # Local proper fifth-coordinate brane surface values inferred from junctions.
    # UV: rho=-6M3 A'; pressure=2M3(N'+2A')
    # IR: rho=+6M3 A'; pressure=-2M3(N'+2A')
    identities['uv_surface_rho']=float(-6*p.M3*Ap[0])
    identities['uv_surface_p']=float(2*p.M3*(Np[0]+2*Ap[0]))
    identities['ir_surface_rho']=float(6*p.M3*Ap[-1])
    identities['ir_surface_p']=float(-2*p.M3*(Np[-1]+2*Ap[-1]))
    return {'parameters':p,'omega':omega,'y':yy,'fields':z,
            'identities':identities,'source':source,'volume_factor':vol}



def brane_actions(sol):
    """Explicit local Z2 brane action values and their stress contributions.

    Brane potentials can realize field jumps with arbitrary constant and
    derivative at selected background values.  The IR anisotropy requires
    a timelike WRONG-SIGN brane scalar kinetic term. Its ghost is a failure
    of physical stability, not an endorsed matter source.
    """
    p=sol['parameters']; z=sol['fields']; d=sol['identities']
    A,N,c,f,Ap,Np,cp,fp=z
    # UV: U(c,f)=lambda+linear_chi*(chi-chi0)+kchi*(chi-chi0)^2/2+kap_uv*f²/2.
    # IR: same shape centered at c(L), with opposite outward scalar jump.
    lam_uv=d['uv_surface_rho']-.5*p.kap_uv*f[0]**2
    u_ir=d['ir_surface_rho']-.5*d['ir_rho_plus_p']
    lam_ir=u_ir-.5*p.kap_ir*f[-1]**2
    ghost_gradient_proper_squared=-d['ir_rho_plus_p']
    ghost_time_derivative=np.sqrt(ghost_gradient_proper_squared)*np.exp(N[-1])
    # Ghost: L= +1/2 g4^ab d_a zeta d_b zeta -U_zeta0,
    # zeta=v t. Stress rho_g=p_g=-v² exp(-2N(L))/2.
    rho_ghost=-.5*ghost_time_derivative**2*np.exp(-2*N[-1])
    p_ghost=rho_ghost
    rho_ir=u_ir+rho_ghost
    p_ir=-u_ir+p_ghost
    # Actual coordinate derivatives and surface stress assembly.
    return {
      'uv_potential_constant':float(lam_uv),
      'uv_chi_linear':float(2*cp[0]),
      'uv_charged_quadratic':float(p.kap_uv),
      'ir_potential_constant':float(lam_ir),
      'ir_chi_linear':float(-2*cp[-1]),
      'ir_charged_quadratic':float(p.kap_ir),
      'ir_ghost_proper_phase_gradient_squared':float(ghost_gradient_proper_squared),
      'ir_ghost_coordinate_phase_rate':float(ghost_time_derivative),
      'ir_rho_from_action':float(rho_ir),
      'ir_p_from_action':float(p_ir),
      'uv_rho_from_action':float(lam_uv+.5*p.kap_uv*f[0]**2),
      'uv_p_from_action':float(-lam_uv-.5*p.kap_uv*f[0]**2),
      'ir_F_slope_action_residual':float(fp[-1]+.5*p.kap_ir*f[-1]),
      'uv_F_slope_action_residual':float(fp[0]-.5*p.kap_uv*f[0]),
      'ir_chi_slope_action_residual':float(cp[-1]+.5*(-2*cp[-1])),
      'uv_chi_slope_action_residual':float(cp[0]-.5*(2*cp[0])),
    }


def independent_einstein_residual(solution, finite_difference=False):
    """Audit G^t_t,G^x_x,G^y_y against T/M3, with independent y differences."""
    y=solution['y'];z=solution['fields'];p=solution['parameters'];omega=solution['omega']
    A,N,c,f,Ap,Np,cp,fp=z
    if finite_difference:
        App=np.gradient(Ap,y,edge_order=2)
        Npp=np.gradient(Np,y,edge_order=2)
    else:
        rhs=odes(y,z,omega,p)
        App=rhs[4];Npp=rhs[5]
    R,Q,V=pressure_terms(z,omega,p)
    Gt=3*App+6*Ap*Ap
    Gx=Npp+2*App+Np*Np+3*Ap*Ap+2*Np*Ap
    Gy=3*Ap*(Ap+Np)
    Tt=-.5*(R+Q)-V
    Tx=-.5*R+.5*Q-V
    Ty=.5*(R+Q)-V
    return np.stack((Gt-Tt/p.M3,Gx-Tx/p.M3,Gy-Ty/p.M3))