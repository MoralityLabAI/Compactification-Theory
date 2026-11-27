"""WS-04: gauge-invariant gravity+Goldberger-Wise radion, flat 4D branes.

Lesgourgues & Sorbo (2004), arXiv:hep-th/0310007, Eqs. (6)-(10),
(16)-(17). Gauge-invariant scalar Psi mixes metric and GW scalar.

H=a_y/a, Pi=Psi_y+2H Psi:
 Psi_y = Pi-2H Psi
 Pi_y =(-H+2 chi_yy/chi_y) Pi + (2 chi_y**2/(3 M5**3)-m2) Psi.
Branes: g_i Pi_i +m2 Psi_i=0, where
 g_UV = H-chi_yy/chi_y + a U_UV''/2 = +a*kappa_uv/2,
 g_IR = H-chi_yy/chi_y - a U_IR''/2 = -a*kappa_ir/2.

Valid on the zero-charge, tuned, flat-sliced two-brane background with ONE
GW scalar. The charged two-scalar time-dependent metric requires a separate
gauge-invariant perturbation study.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass(frozen=True)
class RadionModel:
    k: float=1.0
    M3: float=1.0
    u: float=0.25
    chi_uv: float=0.4
    L: float=1.0
    stiffness_uv: float=8.0
    stiffness_ir: float=8.0

    def __post_init__(self):
        if min(self.k,self.M3,self.u,self.chi_uv,self.L)<=0:
            raise ValueError('geometric and scalar parameters must be strictly positive')
        if min(abs(self.stiffness_uv),abs(self.stiffness_ir)) < 1e-8:
            raise ValueError('stiffness must be nonzero for the shooting Robin formulation')

    def W(self,c):
        return 3*self.M3*self.k + 0.5*self.u*np.asarray(c)**2

    def chi(self,z):
        return self.chi_uv*np.exp(self.u*np.asarray(z))

    def w(self,z):
        z=np.asarray(z)
        return self.k*z+self.chi_uv**2*np.expm1(2*self.u*z)/(12*self.M3)

    def background(self,z):
        z=np.asarray(z)
        c=self.chi(z)
        a=np.exp(-self.w(z))
        wp=self.W(c)/(3*self.M3)
        H=-a*wp
        cy=a*self.u*c
        ratio=a*(self.u-wp) # chi_yy/chi_y
        Hp=H*H-cy*cy/(3*self.M3)
        return a,H,cy,ratio,Hp

    def gvalues(self):
        a0=self.background(0.)[0]
        aL=self.background(self.L)[0]
        return float(a0*self.stiffness_uv/2),float(-aL*self.stiffness_ir/2)

    def integrate(self,m2:float,points:int=401,rtol:float=8e-11):
        """Shoot in proper z coordinate; both tachyonic and positive m2."""
        g0,gL=self.gvalues()
        z=np.linspace(0,self.L,points)
        def ode(zz,state):
            psi,p=state
            a,H,cy,ratio,_=self.background(zz)
            return [(p-2*H*psi)/a, ((-H+2*ratio)*p+(2*cy*cy/(3*self.M3)-m2)*psi)/a]
        sol=solve_ivp(ode,(0,self.L),[1.0,-m2/g0],t_eval=z,
                      method='DOP853',atol=rtol*0.01,rtol=rtol,
                      max_step=self.L/120)
        if not sol.success:
            raise RuntimeError(sol.message)
        psi,p=sol.y
        residual=gL*p[-1]+m2*psi[-1]
        return {'z':z,'psi':psi,'pi':p,'residual':float(residual),
                'g_uv':g0,'g_ir':gL,'m2':float(m2)}

    def boundary_residual(self,m2):
        return self.integrate(m2,points=2)['residual']

    def modes(self,count:int=4,m2max:float=400.,samples:int=1001,scan_tachyonic:float=25.0):
        if count<1 or samples<10: raise ValueError('count and samples too small')
        def roots_between(lims):
            xs=np.asarray(lims,dtype=float)
            vals=np.array([self.boundary_residual(float(x)) for x in xs])
            roots=[]
            for i in range(len(xs)-1):
                if vals[i]*vals[i+1]<0:
                    root=brentq(self.boundary_residual,float(xs[i]),float(xs[i+1]),xtol=1e-11)
                    if not roots or abs(root-roots[-1])>1e-7: roots.append(root)
            return roots
        positive=roots_between(np.linspace(1e-8,m2max,samples))
        negative=roots_between(np.linspace(-scan_tachyonic,-1e-8,max(100,samples//3)))
        if len(positive)<count: raise RuntimeError('mode scan did not bracket enough roots')
        return {'positive_m2':positive[:count],'negative_m2_scan':negative,
                'negative_scan_floor':-scan_tachyonic,'positive_scan_ceiling':m2max,
                'zero_m2_residual':self.boundary_residual(0.)}

    def gauge_invariant_reconstruction(self,m2:float,points:int=401):
        """Bardeen Phi=-2Psi, Delta=-3(Psi_y+2H Psi)/chi_y."""
        row=self.integrate(m2,points)
        z=row['z'];cy=self.background(z)[2]
        return {'z':z,'psi':row['psi'],'Phi':-2*row['psi'],
                'Delta':-3*row['pi']/cy,'boundary_residual':row['residual']}
