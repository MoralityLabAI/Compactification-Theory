"""WS-03: constraint-satisfying charged initial data and five-dimensional evolution.

Time-dependent conformal gauge:
 ds^2 = exp(2B(t,y))(-dt^2 + dy^2) + exp(2A(t,y)) dvecx^2,
 y in [0, yL], with Z2 branes at both endpoints.
Bulk: M5^3 R/2 - (d chi)^2/2 - V(chi).
Brane 0: +2W(chi) + stiffness*(chi-v0)^2/2 + P(X), P=alpha X^2.
Brane L: -2W(chi) + stiffness*(chi-vL)^2/2.

The derived evolution equations, 00 and 0y constraints, and Israel
junctions follow from this ONE stated action. No external matter
is inserted on the IR brane. The scalar chi is a real GW stabilizer,
while the finite-density phase theta lives on the UV brane.

Checks of constraints over finite evolution are reported separately from
exact background and exact initial-slice identities.  Coordinate-gauge
scalar/radion mode stability is NOT established by this experiment.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar


@dataclass(frozen=True)
class System:
    k: float = 1.0
    M3: float = 1.0
    u: float = 0.25
    chi_uv: float = 0.4
    proper_L: float = 1.0
    stiffness_uv: float = 8.0
    stiffness_ir: float = 8.0
    s0: float = 0.001  # alpha X0^2, alpha=1. UV fluid energy is 3*s0.
    vacuum_detune_uv: float = 0.0  # Independent positive UV tension detuning.

    def __post_init__(self):
        if min(self.k, self.M3, self.u, self.chi_uv, self.proper_L,
               self.stiffness_uv, self.stiffness_ir) <= 0 or self.s0 < 0 or self.vacuum_detune_uv < 0:
            raise ValueError('All scales/stiffness positive; s0 >= 0')

    def W(self,c):
        return 3*self.M3*self.k + 0.5*self.u*np.asarray(c)**2

    def V(self,c):
        c=np.asarray(c)
        return 0.5*(self.u*c)**2-2*self.W(c)**2/(3*self.M3)

    def dV(self,c):
        c=np.asarray(c)
        return self.u**2*c-4*self.W(c)*self.u*c/(3*self.M3)

    def d2V(self,c):
        c=np.asarray(c)
        return self.u**2 -4/(3*self.M3)*(self.u**2*c*c+self.u*self.W(c))

    def warp(self,z):
        z=np.asarray(z)
        return self.k*z+self.chi_uv**2*np.expm1(2*self.u*z)/(12*self.M3)

    def c_background(self,z):
        return self.chi_uv*np.exp(self.u*np.asarray(z))

    def uv_tension(self,c):
        return 2*self.W(c)+0.5*self.stiffness_uv*(np.asarray(c)-self.chi_uv)**2 + self.vacuum_detune_uv

    def ir_tension(self,c):
        v=self.c_background(self.proper_L)
        return -2*self.W(c)+0.5*self.stiffness_ir*(np.asarray(c)-v)**2

    def uv_tension_prime(self,c):
        return 2*self.u*np.asarray(c)+self.stiffness_uv*(np.asarray(c)-self.chi_uv)

    def ir_tension_prime(self,c):
        v=self.c_background(self.proper_L)
        return -2*self.u*np.asarray(c)+self.stiffness_ir*(np.asarray(c)-v)

    def background(self,N=3201):
        """Exactly BPS in proper coordinate z; convert to conformal y."""
        z=np.linspace(0,self.proper_L,N)
        w=self.warp(z)
        y=cumulative_trapezoid(np.exp(w),z,initial=0)
        yL=float(y[-1]); z_of_y=PchipInterpolator(y,z)
        def evaluate(yy):
            z=z_of_y(np.asarray(yy))
            A=-self.warp(z)
            c=self.c_background(z)
            Ay=-np.exp(A)*self.W(c)/(3*self.M3)
            cy=np.exp(A)*self.u*c
            return A,Ay,c,cy
        return yL,evaluate

    def fluid(self,Auv, Auv_init=0.0):
        """Conserved 4D phase charge on UV brane; physical a=exp(Auv)."""
        s=self.s0*np.exp(-4*(np.asarray(Auv)-Auv_init))
        return 3*s,s

    def initial_slice(self,n=129,scalar_bump=0.0):
        """Solve the Hamiltonian constraint and all six Israel/scalar jumps.

        Free initial data: B(y) polynomial perturbation of the exact
        GW background, chi(y) smooth polynomial perturbation, and
        vanishing Bdot and chidot. Momentum constraint fixes Adot(y)
        proportional to exp(B-A). The Hamiltonian constraint solves
        the profile A(y), and IR spatial Israel fixes H_UV^2.
        Exact constraints hold for the smooth profiles; grid derivatives
        introduce truncation errors measured separately.
        """
        yL,bps=self.background()
        b0,bp0,c0,cp0=bps(0.0)
        bL,bpL,cL,cpL=bps(yL)
        rho,p=self.fluid(0.)
        a=float((2*rho+3*p-self.vacuum_detune_uv)/(6*self.M3))
        def b_interpolation_equation(b):
            delta=a*yL+b*yL*yL
            return bpL+a+2*b*yL - bpL*np.exp(delta)
        # Isolate the small deformation branch near the BPS solution.
        b_lo=-max(0.3,4*a/yL)
        b_hi=max(0.3,4*a/yL)
        bracket=root_scalar(b_interpolation_equation,bracket=[b_lo,b_hi],xtol=1e-14)
        if not bracket.converged:
            raise RuntimeError('Unable to solve IR temporal junction')
        b=float(bracket.root)
        deltaL=a*yL+b*yL*yL
        coeff=float(cpL*np.expm1(deltaL)/yL**2)
        def profiles(y):
            y=np.asarray(y)
            base,basep,c,cp=bps(y)
            B=base+a*y+b*y*y
            Bp=basep+a+2*b*y
            # Perturb interior GW scalar, with vanishing value AND first
            # derivative at both branes; the boundary action stays fixed.
            t=y/yL
            bump=16*scalar_bump*t*t*(1-t)*(1-t)
            bump_y=(32*scalar_bump/yL)*t*(1-t)*(1-2*t)
            c2=c+coeff*y*y*(y-yL)+bump
            cp2=cp+coeff*(3*y*y-2*y*yL)+bump_y
            return B,Bp,c2,cp2
        initial_Ap=float(-self.uv_tension(self.chi_uv)/(6*self.M3)-rho/(6*self.M3))
        def integrate(q):
            def rhs(y,vec):
                A,Ap=vec
                B,Bp,c,cp=profiles(y)
                Adot2=q*np.exp(2*(B-A))
                App=Bp*Ap+Adot2-2*Ap*Ap - (0.5*cp**2+np.exp(2*B)*self.V(c))/(3*self.M3)
                return (Ap,App)
            sol=solve_ivp(rhs,[0,yL],[0,initial_Ap],dense_output=True,
                          rtol=2e-11,atol=1e-13)
            if not sol.success:
                raise RuntimeError(sol.message)
            return sol
        def boundary_mismatch(q):
            sol=integrate(q)
            _,Ap=sol.sol(yL)
            B,_,c,_=profiles(yL)
            return Ap-np.exp(B)*self.ir_tension(c)/(6*self.M3)
        if self.s0 == 0 and self.vacuum_detune_uv == 0 and scalar_bump == 0:
            q=0.0
        else:
            # Escalate until a positive UV expansion rate meets IR Israel.
            q_hi=max(float(rho)+self.vacuum_detune_uv,1e-6)
            if boundary_mismatch(0)*boundary_mismatch(q_hi)>0:
                for _ in range(30):
                    q_hi*=2
                    if boundary_mismatch(0)*boundary_mismatch(q_hi)<=0:
                        break
                else:
                    raise RuntimeError('No nonnegative H_UV^2 satisfies IR spatial Israel')
            root=root_scalar(boundary_mismatch,bracket=[0,q_hi],xtol=1e-14)
            if not root.converged:
                raise RuntimeError('Unable to find initial UV expansion rate')
            q=float(root.root)
        sol=integrate(q)
        y=np.linspace(0,yL,n)
        A,Ap=sol.sol(y)
        B,Bp,c,cp=profiles(y)
        Adot=np.sqrt(q)*np.exp(B-A)
        Bdot=np.zeros_like(y); cdot=np.zeros_like(y)
        # Analytic constraint: A_yy comes from the integrating ODE.
        App=Bp*Ap+Adot**2-2*Ap**2-(0.5*cp**2+np.exp(2*B)*self.V(c))/(3*self.M3)
        ham=3*(-App +Bp*Ap+Adot**2-2*Ap**2) - (0.5*cp**2+np.exp(2*B)*self.V(c))/self.M3
        # Analytic momentum: Adot_y = (B_y-A_y) Adot.
        mom=-3*((Bp-Ap)*Adot-Bp*Adot+Adot*Ap)
        bc=self.boundary_slopes(A,B,c)
        jumps={name:float(value) for name,value in zip(
            ('uv_A','uv_B','uv_chi','ir_A','ir_B','ir_chi'),
            (Ap[0]-bc['A'][0],Bp[0]-bc['B'][0],cp[0]-bc['chi'][0],
             Ap[-1]-bc['A'][1],Bp[-1]-bc['B'][1],cp[-1]-bc['chi'][1]))}
        return {'y':y,'A':A,'B':B,'chi':c,'Adot':Adot,'Bdot':Bdot,
                'chidot':cdot,'q':q,'yL':yL,'jump_errors':jumps,
                'hamiltonian_max_analytic':float(np.max(np.abs(ham))),
                'momentum_max_analytic':float(np.max(np.abs(mom))),
                'A_initial':0.0,'B_initial':0.0,
                'radion_length_initial':float(np.trapezoid(np.exp(B),y)),
                'scalar_bump':float(scalar_bump)}

    def boundary_slopes(self,A,B,c):
        rho,p=self.fluid(A[0])
        lam0=self.uv_tension(c[0]); lamL=self.ir_tension(c[-1])
        e0=np.exp(B[0]);eL=np.exp(B[-1]);den=6*self.M3
        return {'A':np.array([-e0*(lam0+rho)/den,+eL*lamL/den]),
                'B':np.array([-e0*(lam0-2*rho-3*p)/den,+eL*lamL/den]),
                'chi':np.array([+0.5*e0*self.uv_tension_prime(c[0]),
                                -0.5*eL*self.ir_tension_prime(c[-1])])}

    @staticmethod
    def differences(v,dy,left,right):
        v=np.asarray(v)
        derivative=np.empty_like(v)
        second=np.empty_like(v)
        derivative[1:-1]=(v[2:]-v[:-2])/(2*dy)
        derivative[0]=left;derivative[-1]=right
        second[1:-1]=(v[2:]-2*v[1:-1]+v[:-2])/(dy*dy)
        # Four nodes plus one prescribed first derivative: exact to degree 4.
        second[0]=(-85*v[0]/18+6*v[1]-1.5*v[2]+(2/9)*v[3])/dy**2-11*left/(3*dy)
        second[-1]=(-85*v[-1]/18+6*v[-2]-1.5*v[-3]+(2/9)*v[-4])/dy**2+11*right/(3*dy)
        return derivative,second

    def state_rhs(self,t,flat,n,dy):
        x=flat.reshape(6,n)
        A,B,c,At,Bt,ct=x
        bc=self.boundary_slopes(A,B,c)
        Ay,Ayy=self.differences(A,dy,*bc['A'])
        By,Byy=self.differences(B,dy,*bc['B'])
        cy,cyy=self.differences(c,dy,*bc['chi'])
        ev=np.exp(2*B)
        A2=Ayy-3*At**2+3*Ay**2+(2/(3*self.M3))*ev*self.V(c)
        B2=Byy+3*At**2-3*Ay**2-(ct**2-cy**2)/(2*self.M3)-ev*self.V(c)/(3*self.M3)
        c2=cyy-3*At*ct+3*Ay*cy-ev*self.dV(c)
        return np.stack((At,Bt,ct,A2,B2,c2)).ravel()

    @staticmethod
    def one_sided_junction_error(v,dy,left,right):
        """Independent fourth-order one-sided derivative vs Israel slope."""
        d0=(-25*v[0]+48*v[1]-36*v[2]+16*v[3]-3*v[4])/(12*dy)
        dL=(25*v[-1]-48*v[-2]+36*v[-3]-16*v[-4]+3*v[-5])/(12*dy)
        return max(abs(d0-left),abs(dL-right))

    def constraints(self,state,dy):
        A,B,c,At,Bt,ct=state.reshape(6,-1)
        bc=self.boundary_slopes(A,B,c)
        Ay,Ayy=self.differences(A,dy,*bc['A'])
        By,_=self.differences(B,dy,*bc['B'])
        cy,_=self.differences(c,dy,*bc['chi'])
        Aty=np.gradient(At,dy,edge_order=2)
        tt=3*(-Ayy +Bt*At+By*Ay+At**2-2*Ay**2) - (0.5*(ct**2+cy**2)+np.exp(2*B)*self.V(c))/self.M3
        ty=-3*(Aty-By*At-Bt*Ay+At*Ay)-ct*cy/self.M3
        return tt,ty

    def evolve(self,n=97,time_end=0.08,samples=21,rtol=3e-8,scalar_bump=0.0):
        initial=self.initial_slice(n=n,scalar_bump=scalar_bump)
        y=initial['y'];dy=float(y[1]-y[0])
        state0=np.stack([initial[key] for key in ('A','B','chi','Adot','Bdot','chidot')]).ravel()
        time=np.linspace(0,time_end,samples)
        # Boundary eigenvalues are stiff with dy^-2, use max_step ~ dx/4.
        sol=solve_ivp(lambda t,z:self.state_rhs(t,z,n,dy),(0,time_end),state0,
                      t_eval=time,method='DOP853',rtol=rtol,atol=max(1e-11,rtol*1e-3),
                      max_step=dy/5)
        if not sol.success:
            raise RuntimeError(sol.message)
        diagnostics=[]
        for timei,z in zip(sol.t,sol.y.T):
            state=z.reshape(6,n)
            A,B,c,At,Bt,ct=state
            tt,ty=self.constraints(z,dy)
            D=float(np.trapezoid(np.exp(B),y))
            rho,p=self.fluid(A[0])
            # Independent one-sided derivative estimate (the PDE stencil
            # imposes junctions using their exact slope rather than this test).
            slopes=self.boundary_slopes(A,B,c)
            mismatch={name:float(self.one_sided_junction_error(field,dy,*slopes[name]))
                      for name,field in zip(('A','B','chi'),(A,B,c))}
            # Local brane cosmic Hubble with induced lapse e^B.
            Huv=float(np.exp(-B[0])*At[0])
            charge_ratio=float(np.exp(3*A[0])*(2*np.sqrt(self.s0*np.exp(-4*A[0])))**1.5 / (2*np.sqrt(self.s0))**1.5) if self.s0>0 else 1.
            diagnostics.append({'t':float(timei),'constraint_00_max_interior':float(np.max(np.abs(tt[3:-3]))),
                'constraint_0y_max_interior':float(np.max(np.abs(ty[3:-3]))),
                'radion_proper_length':D,'radion_rate':float(np.trapezoid(np.exp(B)*Bt,y)),
                'H_UV':Huv,'rho_UV':float(rho),'p_UV':float(p),
                'charge_relative_error':abs(charge_ratio-1),
                'junction_one_sided_max':float(max(mismatch.values())),
                'junction_one_sided_by_field':mismatch})
        return {'y':y,'time':time,'state':sol.y.T.reshape(samples,6,n),
                'diagnostics':diagnostics,'initial':initial,
                'max_00_constraint_interior':max(d['constraint_00_max_interior'] for d in diagnostics),
                'max_0y_constraint_interior':max(d['constraint_0y_max_interior'] for d in diagnostics)}
