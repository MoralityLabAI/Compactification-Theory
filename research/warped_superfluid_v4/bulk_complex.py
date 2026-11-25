"""WS-04: global-U(1) charged bulk complex scalar, full 5D Einstein backreaction.

One canonical complex scalar is represented by two real fields x+iy.
The globally conserved U(1) charge has zero normal flux on both branes
because both Robin conditions derive from U(1)-symmetric brane potentials.
This is a coherent classical field, not a confirmed thermodynamic superfluid.
Gauge-invariant stability of the *time-dependent charged two-scalar* solution
remains open; radion.py applies to the uncharged static one-scalar reference.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from ws03 import System


@dataclass(frozen=True)
class BulkChargedSystem(System):
    s0: float=0.0
    bulk_amp: float=0.02
    bulk_omega: float=0.4
    bulk_mass2: float=0.5
    bulk_lambda: float=0.3
    bulk_stiffness_uv: float=1.0
    bulk_stiffness_ir: float=1.0

    def __post_init__(self):
        super().__post_init__()
        if min(self.bulk_amp,self.bulk_mass2,self.bulk_lambda,
               self.bulk_stiffness_uv,self.bulk_stiffness_ir)<0:
            raise ValueError('Bulk amplitude, quartic, mass and stiffness must be nonnegative')

    def bulk_potential(self,x,y):
        f2=np.asarray(x)**2+np.asarray(y)**2
        return .5*self.bulk_mass2*f2+.25*self.bulk_lambda*f2*f2

    def bulk_deriv_factor(self,x,y):
        return self.bulk_mass2+self.bulk_lambda*(np.asarray(x)**2+np.asarray(y)**2)

    def boundary_slopes(self,A,B,c,x,y):
        rho,p=self.fluid(A[0])
        u0=self.uv_tension(c[0])+.5*self.bulk_stiffness_uv*(x[0]**2+y[0]**2)
        uL=self.ir_tension(c[-1])+.5*self.bulk_stiffness_ir*(x[-1]**2+y[-1]**2)
        e0=np.exp(B[0]);eL=np.exp(B[-1]);den=6*self.M3
        return {'A':np.array([-e0*(u0+rho)/den,+eL*uL/den]),
                'B':np.array([-e0*(u0-2*rho-3*p)/den,+eL*uL/den]),
                'chi':np.array([+.5*e0*self.uv_tension_prime(c[0]),
                                -.5*eL*self.ir_tension_prime(c[-1])]),
                'x':np.array([+.5*e0*self.bulk_stiffness_uv*x[0],
                              -.5*eL*self.bulk_stiffness_ir*x[-1]]),
                'y':np.array([+.5*e0*self.bulk_stiffness_uv*y[0],
                              -.5*eL*self.bulk_stiffness_ir*y[-1]])}

    def initial_slice(self,n=129):
        """Exact smooth C00/C0y initial data and ten brane jump conditions.

        A time rotating bulk field initially has x(y)=F(y), y_field=0,
        xdot=0, ydot=omega F. GW field and B conformal lapse are chosen
        with boundary-corrected slopes. Hamiltonian determines A(y);
        IR spatial junction determines initial H_uv squared.
        """
        yL,bps=self.background()
        bL,bpL,cL,cpL=bps(yL)
        rho,p=self.fluid(0.0)
        f0=self.bulk_amp;fL=self.bulk_amp
        u0extra=.5*self.bulk_stiffness_uv*f0*f0
        uLextra=.5*self.bulk_stiffness_ir*fL*fL
        a=(2*rho+3*p-self.vacuum_detune_uv-u0extra)/(6*self.M3)
        bps_ir_target=bpL+np.exp(bL)*uLextra/(6*self.M3)

        def b_mismatch(b):
            delta=a*yL+b*yL*yL
            return bpL+a+2*b*yL - np.exp(delta)*bps_ir_target
        half=max(.3,4*abs(a)/yL)
        br=root_scalar(b_mismatch,bracket=[-half,half],xtol=1e-14)
        if not br.converged: raise RuntimeError('IR temporal junction bracket failed')
        b=float(br.root)
        deltaL=a*yL+b*yL*yL
        coeff=float(cpL*np.expm1(deltaL)/yL**2)
        xleft=.5*self.bulk_stiffness_uv*f0
        xright=-.5*np.exp(bL+deltaL)*self.bulk_stiffness_ir*fL

        def profiles(yy):
            yy=np.asarray(yy)
            base,basep,c,cp=bps(yy)
            B=base+a*yy+b*yy*yy
            Bp=basep+a+2*b*yy
            c2=c+coeff*yy*yy*(yy-yL)
            cp2=cp+coeff*(3*yy*yy-2*yy*yL)
            t=yy/yL
            x=f0+yL*((t**3-2*t**2+t)*xleft+(t**3-t**2)*xright)
            xp=(3*t*t-4*t+1)*xleft+(3*t*t-2*t)*xright
            return B,Bp,c2,cp2,x,xp

        initial_Ap=(-self.uv_tension(self.chi_uv)-u0extra-rho)/(6*self.M3)

        def integrate(q):
            def rhs(y,vec):
                A,Ap=vec
                B,Bp,c,cp,x,xp=profiles(y)
                At2=q*np.exp(2*(B-A))
                Vtot=self.V(c)+self.bulk_potential(x,0.)
                scalar_energy=.5*(cp*cp+xp*xp+self.bulk_omega**2*x*x)
                App=Bp*Ap+At2-2*Ap**2-(scalar_energy+np.exp(2*B)*Vtot)/(3*self.M3)
                return [Ap,App]
            sol=solve_ivp(rhs,(0,yL),[0.,initial_Ap],dense_output=True,
                          method='DOP853',rtol=3e-11,atol=2e-13)
            if not sol.success: raise RuntimeError(sol.message)
            return sol

        def mismatch(q):
            A,Ap=integrate(q).sol(yL)
            B,_,c,_,x,_=profiles(yL)
            lam=self.ir_tension(c)+.5*self.bulk_stiffness_ir*x*x
            return Ap-np.exp(B)*lam/(6*self.M3)
        if f0==0 and self.s0==0 and self.vacuum_detune_uv==0:
            q=0.
        else:
            lo=0.
            hi=max(1e-6,self.bulk_omega**2*f0*f0+self.bulk_mass2*f0*f0+rho+self.vacuum_detune_uv)
            flo=mismatch(lo);fhi=mismatch(hi)
            for _ in range(35):
                if flo*fhi<=0: break
                hi*=2;fhi=mismatch(hi)
            else: raise RuntimeError('Unable to bracket nonnegative H^2 with charged bulk')
            q=float(root_scalar(mismatch,bracket=[lo,hi],xtol=5e-14).root)

        sol=integrate(q)
        yy=np.linspace(0,yL,n)
        A,Ap=sol.sol(yy)
        B,Bp,c,cp,x,xp=profiles(yy)
        phase_y=np.zeros_like(yy)
        At=np.sqrt(q)*np.exp(B-A)
        Bt=np.zeros_like(yy);ct=np.zeros_like(yy)
        xt=np.zeros_like(yy);yt=self.bulk_omega*x
        energy=.5*(cp**2+xp**2+yt**2)+np.exp(2*B)*(self.V(c)+self.bulk_potential(x,phase_y))
        App=Bp*Ap+At**2-2*Ap**2-energy/(3*self.M3)
        c00=3*(-App+Bp*Ap+At**2-2*Ap**2)-energy/self.M3
        c0y=-3*((Bp-Ap)*At-Bp*At+At*Ap)
        jumps=self.boundary_slopes(A,B,c,x,phase_y)
        errors={name:float(max(abs(np.array([deriv[0],deriv[-1]])-jumps[name])))
                for name,deriv in [('A',Ap),('B',Bp),('chi',cp),('x',xp),
                                   ('y',np.zeros_like(yy))]}
        return {'grid':yy,'A':A,'B':B,'chi':c,'x':x,'y':phase_y,
                'At':At,'Bt':Bt,'ct':ct,'xt':xt,'yt':yt,
                'H2_uv':float(q),'constraint_00_exact':float(np.max(np.abs(c00))),
                'constraint_0y_exact':float(np.max(np.abs(c0y))),
                'junction_exact':errors}

    def rhs(self,t,flat,n,dy):
        A,B,c,x,y,At,Bt,ct,xt,yt=flat.reshape(10,n)
        bc=self.boundary_slopes(A,B,c,x,y)
        Ay,Ayy=self.differences(A,dy,*bc['A'])
        By,Byy=self.differences(B,dy,*bc['B'])
        cy,cyy=self.differences(c,dy,*bc['chi'])
        xy,xyy=self.differences(x,dy,*bc['x'])
        yy,yyy=self.differences(y,dy,*bc['y'])
        v=self.V(c)+self.bulk_potential(x,y)
        kin=(ct*ct-cy*cy)+(xt*xt-xy*xy)+(yt*yt-yy*yy)
        e2=np.exp(2*B)
        A2=Ayy-3*At*At+3*Ay*Ay+(2/(3*self.M3))*e2*v
        B2=Byy+3*At*At-3*Ay*Ay-kin/(2*self.M3)-e2*v/(3*self.M3)
        c2=cyy-3*At*ct+3*Ay*cy-e2*self.dV(c)
        coeff=e2*self.bulk_deriv_factor(x,y)
        x2=xyy-3*At*xt+3*Ay*xy-coeff*x
        y2=yyy-3*At*yt+3*Ay*yy-coeff*y
        return np.stack((At,Bt,ct,xt,yt,A2,B2,c2,x2,y2)).ravel()

    def constraints(self,state,dy):
        A,B,c,x,y,At,Bt,ct,xt,yt=state.reshape(10,-1)
        bc=self.boundary_slopes(A,B,c,x,y)
        Ay,Ayy=self.differences(A,dy,*bc['A'])
        By,_=self.differences(B,dy,*bc['B'])
        cy,_=self.differences(c,dy,*bc['chi'])
        xy,_=self.differences(x,dy,*bc['x'])
        yy,_=self.differences(y,dy,*bc['y'])
        Aty=np.gradient(At,dy,edge_order=2)
        tot=ct**2+cy**2+xt**2+xy**2+yt**2+yy**2
        potential=self.V(c)+self.bulk_potential(x,y)
        c00=3*(-Ayy+Bt*At+By*Ay+At**2-2*Ay**2)-(0.5*tot+np.exp(2*B)*potential)/self.M3
        c0y=-3*(Aty-By*At-Bt*Ay+At*Ay)-(ct*cy+xt*xy+yt*yy)/self.M3
        return c00,c0y

    @staticmethod
    def global_charge(state,grid):
        A,B,c,x,y,At,Bt,ct,xt,yt=state
        return float(np.trapezoid(np.exp(3*A)*(x*yt-y*xt),grid))

    def boundary_flux(self,state):
        A,B,c,x,y,At,Bt,ct,xt,yt=state
        bc=self.boundary_slopes(A,B,c,x,y)
        fluxes=[]
        for j in [0,1]:
            idx=0 if j==0 else -1
            fluxes.append(float(np.exp(3*A[idx])*(x[idx]*bc['y'][j]-y[idx]*bc['x'][j])))
        return fluxes

    def evolve(self,n=65,time_end=.12,samples=9,rtol=4e-8):
        initial=self.initial_slice(n)
        grid=initial['grid'];dy=float(grid[1]-grid[0])
        base=np.stack([initial[key] for key in ('A','B','chi','x','y','At','Bt','ct','xt','yt')])
        q0=self.global_charge(base,grid)
        ts=np.linspace(0,time_end,samples)
        sol=solve_ivp(lambda t,v:self.rhs(t,v,n,dy),(0,time_end),base.ravel(),
                      method='DOP853',t_eval=ts,rtol=rtol,
                      atol=max(1e-11,rtol*1e-3),max_step=dy/5)
        if not sol.success: raise RuntimeError(sol.message)
        diagnostics=[]
        for time,row in zip(ts,sol.y.T):
            state=row.reshape(10,n)
            C0,Cy=self.constraints(state,dy)
            charge=self.global_charge(state,grid)
            flux=self.boundary_flux(state)
            A,B,c,x,y,At,Bt,ct,xt,yt=state
            jc=self.boundary_slopes(A,B,c,x,y)
            mismatch=max(self.one_sided_junction_error(field,dy,*jc[name])
                         for name,field in [('A',A),('B',B),('chi',c),('x',x),('y',y)])
            diagnostics.append({'time':float(time),'Huv':float(At[0]*np.exp(-B[0])),
                'proper_length':float(np.trapezoid(np.exp(B),grid)),
                'charge':charge,'charge_drift':float(abs(charge/q0-1)) if abs(q0)>1e-14 else 0.,
                'normal_current_flux':max(abs(q) for q in flux),
                'constraint00':float(np.max(np.abs(C0[3:-3]))),
                'constraint0y':float(np.max(np.abs(Cy[3:-3]))),
                'junction_error':float(mismatch)})
        return {'t':ts,'grid':grid,'states':sol.y.T.reshape(samples,10,n),
                'diagnostics':diagnostics,'initial':initial,
                'max_constraint00':max(d['constraint00'] for d in diagnostics),
                'max_constraint0y':max(d['constraint0y'] for d in diagnostics),
                'max_charge_drift':max(d['charge_drift'] for d in diagnostics)}
