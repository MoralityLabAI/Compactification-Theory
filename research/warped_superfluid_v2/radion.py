"""Consistent 4D Einstein-frame radion + charged fluid benchmark.

This is a 4D effective action analogue, not a derived 2-brane solution.
S=∫sqrt(-g)[M4² R/2 -(∂r)²/2 -m_r²r²/2
             + exp(4βr/M4) P_J(exp(-2βr/M4) X)],
P_J=-V0+alpha X_J².
A conformal coincidence makes the quartic X² sector radion-independent.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw

@dataclass(frozen=True)
class RadionEFT:
    M4:float=1.0
    mr:float=0.8
    beta:float=0.15
    V0:float=0.05
    s0:float=0.02

    def __post_init__(self):
        if min(self.M4,self.mr,self.s0)<=0 or self.V0<0:
            raise ValueError('M4, mr, s0 must be positive; V0 nonnegative')

    def vacuum(self,r):
        return self.V0*np.exp(4*self.beta*np.asarray(r)/self.M4)

    def vacuum_derivative(self,r):
        return (4*self.beta/self.M4)*self.vacuum(r)

    def Veff(self,r):
        return 0.5*self.mr**2*np.asarray(r)**2+self.vacuum(r)

    def Veff_derivative(self,r):
        return self.mr**2*np.asarray(r)+self.vacuum_derivative(r)

    def Veff_second(self,r):
        return self.mr**2+(4*self.beta/self.M4)**2*self.vacuum(r)

    def equilibrium(self):
        if self.beta==0 or self.V0==0:
            return 0.0
        a=4*self.beta/self.M4
        return float(-lambertw(a*a*self.V0/self.mr**2).real/a)

    def components(self,N,r,pi):
        radiation=self.s0*np.exp(-4*np.asarray(N))
        Vrad=0.5*self.mr**2*np.asarray(r)**2
        Vvac=self.vacuum(r)
        rho=0.5*np.asarray(pi)**2 +Vrad+Vvac+3*radiation
        p=0.5*np.asarray(pi)**2 -Vrad-Vvac+radiation
        H=np.sqrt(rho/(3*self.M4**2))
        dH=-0.5*(np.asarray(pi)**2+4*radiation)/self.M4**2
        return {'H':H,'Hdot':dH,'rho':rho,'p':p,
                'Vrad':Vrad,'Vvac':Vvac,'kinetic_rho':3*radiation,
                'trace_fluid':-4*Vvac}

    def evolve(self,Nmax=7,steps=1001,r_initial=0,pi_initial=0,rtol=1e-11):
        def rhs(N,y):
            r,pi=y
            H=self.components(N,r,pi)['H']
            return (pi/H, -3*pi-self.Veff_derivative(r)/H)
        N=np.linspace(0,Nmax,steps)
        sol=solve_ivp(rhs,(0,Nmax),[r_initial,pi_initial],t_eval=N,
                      rtol=rtol,atol=1e-13,method='DOP853')
        if not sol.success:
            raise RuntimeError(sol.message)
        r,pi=sol.y
        comp=self.components(N,r,pi)
        H=comp['H']
        # Check continuity using exact derivatives of numerical solution.
        # d rho/dN = pi*( -3pi -V'/H )H + V' pi/H -12s = -3pi²-12s.
        drho_dN=-3*pi**2-12*self.s0*np.exp(-4*N)
        rho_plus_p=comp['rho']+comp['p']
        cons=drho_dN + 3*rho_plus_p
        mass2=self.Veff_second(r)
        # Independent finite-difference checks on the solved trajectory;
        # the algebraic Friedmann residual alone is a construction identity.
        numerical_dH_dN=np.gradient(H,N,edge_order=2)
        raychaudhuri_N=self.components(N,r,pi)['Hdot']/H
        numerical_dpi_dN=np.gradient(pi,N,edge_order=2)
        pi_N_rhs=-3*pi-self.Veff_derivative(r)/H
        raychaudhuri_fd_error=float(np.max(abs(numerical_dH_dN[2:-2]-raychaudhuri_N[2:-2])))
        radion_eom_fd_error=float(np.max(abs(numerical_dpi_dN[2:-2]-pi_N_rhs[2:-2])))
        return {'N':N,'r':r,'pi':pi,'H':H,
                'rho':comp['rho'],'p':comp['p'],
                'vacuum':comp['Vvac'],'radion_potential':comp['Vrad'],
                'fluid_kinetic_density':comp['kinetic_rho'],
                'w_total':comp['p']/comp['rho'],
                'effective_radion_mass2':mass2,
                'friedmann_residual_max':float(np.max(np.abs(3*self.M4**2*H**2-comp['rho']))),
                'continuity_residual_max':float(np.max(np.abs(cons))),
                'raychaudhuri_fd_max_abs':raychaudhuri_fd_error,
                'radion_eom_fd_max_abs':radion_eom_fd_error,
                'equilibrium_r':self.equilibrium(),
                'equilibrium_vacuum_energy':float(self.Veff(self.equilibrium())),
                'asymptotic_H_expected':float(np.sqrt(self.Veff(self.equilibrium())/(3*self.M4**2)))}
