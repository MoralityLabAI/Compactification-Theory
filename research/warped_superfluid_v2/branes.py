"""WS-02: exact AdS5/FRW *single-brane* embedding of charged brane k-essence.

G_NN=1, Lambda5=-6k^2; Z2 brane at y=0 with tension sigma=6 k M5^3.
Metric: ds² = -n(t,y)²dt² + a(t,y)² dvecx² + dy².
The y=L boundary is a hypothetical SECOND brane: junction mismatch is measured,
not repaired by asserting a stabilizing bulk scalar solution.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid

@dataclass(frozen=True)
class ChargedBrane:
    k: float = 1.0
    M5cubed: float = 1.0
    V0: float = 0.05
    s0: float = 0.02   # alpha X0^2, positive
    alpha: float = 1.0

    def __post_init__(self):
        if min(self.k, self.M5cubed, self.alpha) <= 0 or self.V0 < 0 or self.s0 <= 0:
            raise ValueError('Choose positive k, M5cubed, alpha, s0, and nonnegative V0')

    @property
    def sigma(self):
        return 6 * self.k * self.M5cubed

    @property
    def M4squared(self):
        return self.M5cubed / self.k

    @property
    def Q(self):
        # P=alpha X^2-V0; Q=a^3 P_X theta_dot = alpha a^3 theta_dot^3.
        return self.alpha * (2 * np.sqrt(self.s0 / self.alpha))**1.5

    def fluid(self, N):
        """N=ln(a0/a_today).  rho,p,s, c_s², phase derivative, charge."""
        N = np.asarray(N, dtype=float)
        s = self.s0 * np.exp(-4*N)
        rho = self.V0 + 3*s
        p = -self.V0 + s
        theta_dot = np.sqrt(2 * np.sqrt(s / self.alpha))
        q = self.alpha * np.exp(3*N)*theta_dot**3
        return rho,p,s,theta_dot,q,np.broadcast_to(1/3, N.shape)

    def hubble(self, N):
        rho = self.fluid(N)[0]
        return self.k * np.sqrt((1+rho/self.sigma)**2-1)

    def hubble_dot(self, N):
        rho,p,*_ = self.fluid(N)
        return -3*self.k**2*(1+rho/self.sigma)*(rho+p)/self.sigma

    def bulk(self, N, y):
        """Exact 5D time-dependent solution sampled at arrays broadcastable to 2D.

        Inputs N shape (T,), y shape (Y,). All derivatives are analytical.
        The formula is valid in the Gaussian-normal patch where D,N_lapse>0.
        """
        N = np.atleast_1d(np.asarray(N,dtype=float))[:,None]
        y = np.atleast_1d(np.asarray(y,dtype=float))[None,:]
        rho,p,s,tdot,q,cs2=self.fluid(N)
        H=self.hubble(N)
        Hdot=self.hubble_dot(N)
        a0=np.exp(N)
        k=self.k
        c=np.cosh(k*y)
        sh=np.sinh(k*y)
        B=1+rho/self.sigma
        C=1-(2*rho+3*p)/self.sigma
        D=c-B*sh
        lapse=c-C*sh
        if np.any(D<=0) or np.any(lapse<=0):
            raise ValueError('Gaussian-normal patch reaches a zero of a or lapse: shorten y interval')
        D_y=k*(sh-B*c)
        lapse_y=k*(sh-C*c)
        a=a0*D
        a_t=a0*H*lapse
        a_tt=a0*((H*H+Hdot)*lapse -36*H*H*s/self.sigma*sh)
        n_t=-36*H*s/self.sigma*sh
        # (time derivative of warp factor) is tied to conservation of fluid charge
        a_y=a0*D_y
        a_yy=k*k*a
        a_ty=a0*H*lapse_y
        return {'Nlog':N, 'y':y, 'rho':rho, 'p':p, 's':s, 'H':H,
                'Hdot':Hdot, 'D':D, 'lapse':lapse, 'D_y':D_y,
                'lapse_y':lapse_y, 'a':a, 'a_t':a_t, 'a_tt':a_tt,
                'n_t':n_t, 'a_y':a_y, 'a_yy':a_yy, 'a_ty':a_ty}

    def einstein_bulk_residuals(self, N, y):
        """All independent Einstein equations on smooth bulk y>0.

        G_AB = +6k² g_AB.  Einstein tensor formulas for warped product
        with flat 3-space and base metric diag(-lapse²,1).
        Returns normalized max abs residuals for tt,yy,ii,ty, and first integral.
        """
        b=self.bulk(N,y)
        a,n,at,att,nt,ay,ayy,aty,ny,n_yy=(
            b['a'],b['lapse'],b['a_t'],b['a_tt'],b['n_t'],b['a_y'],
            b['a_yy'],b['a_ty'],b['lapse_y'],self.k**2*b['lapse'])
        grad=-at**2/n**2+ay**2
        lap=-att/n**2+(nt*at)/n**3+ayy+(ny/n)*ay
        cov_tt=att-(nt/n)*at-n*ny*ay
        cov_yy=ayy
        cov_ty=aty-(ny/n)*at
        Gtt_over_n2=-3*cov_tt/(a*n**2)-3*lap/a-3*grad/a**2
        Gyy=-3*cov_yy/a+3*lap/a+3*grad/a**2
        Gii_over_a2=n_yy/n+2*lap/a+grad/a**2
        Gty=-3*cov_ty/a
        identity=(ay/a)**2-(at/(n*a))**2-self.k**2
        return {'tt':float(np.max(np.abs(Gtt_over_n2+6*self.k**2))),
                'yy':float(np.max(np.abs(Gyy-6*self.k**2))),
                'ii':float(np.max(np.abs(Gii_over_a2-6*self.k**2))),
                'ty':float(np.max(np.abs(Gty))),
                'first_integral':float(np.max(np.abs(identity)))}

    def junction(self,N,L=1.5):
        """UV Israel conditions, and required IR fluid for second fixed-y brane.

        UV tuned tension +sigma; hypothetical IR tuned tension -sigma.
        Without a GW bulk scalar or evolving L, rho_IR,p_IR must take
        nontrivial generally exotic values. These values are diagnostics.
        """
        b=self.bulk(N,np.array([0.0,L]))
        ay_a=b['a_y']/b['a']
        ny_n=b['lapse_y']/b['lapse']
        rho,p,*_=self.fluid(np.atleast_1d(N))
        uv_a=ay_a[:,0]+self.k*(1+rho/self.sigma)
        uv_n=ny_n[:,0]+self.k*(1-(2*rho+3*p)/self.sigma)
        rho_IR=self.sigma*(ay_a[:,1]/self.k+1)
        p_IR=(-self.sigma*ny_n[:,1]/self.k -self.sigma-2*rho_IR)/3
        return {'uv_spatial_residual_max':float(np.max(abs(uv_a))),
                'uv_temporal_residual_max':float(np.max(abs(uv_n))),
                'ir_required_density':rho_IR,
                'ir_required_pressure':p_IR,
                'ir_tuned_vacuum_mismatch':float(max(np.max(abs(rho_IR)),np.max(abs(p_IR))))}

    def history(self,Nmax=4,n=501):
        N=np.linspace(0,Nmax,n)
        H=self.hubble(N)
        t=cumulative_trapezoid(1/H,N,initial=0)
        rho,p,s,tdot,q,cs=self.fluid(N)
        continuity=(-12*s+3*(rho+p))
        return {'N':N,'t':t,'H':H,'Hdot':self.hubble_dot(N),
                'rho':rho,'p':p,'charge':q,'charge_relative_drift':
                float(np.max(abs(q/self.Q-1))),
                'continuity_max_abs':float(np.max(abs(continuity)))}
