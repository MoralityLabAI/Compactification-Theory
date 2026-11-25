"""WS-06: 5D charged stationary *probe* on fixed RS spacetime.

The probe neglects its own 5D Einstein stress, and is therefore a controlled
low-amplitude/decoupling calculation, NOT a charged self-gravitating background.

For ds²=e^-2ky eta_mu_nu dx^mu dx^nu+dy², Psi=F(y)exp(i omega t),
 V=m²|Psi|²/2 + lam |Psi|⁴/4, there are Robin UV/IR brane conditions
 F'(0)=kap_uv*F(0)/2, F'(L)=-kap_ir*F(L)/2.

Weak-form equilibrium: (K0 + lam T(F²))F=omega² M F.
Linearized fields (r,s) are measured in the co-rotating basis:
 M*r_tt -2 omega M*s_t + (Kr+p²M)r=0,
 M*s_tt +2 omega M*r_t + (Ks+p²M)s=0.

Ks F=0, Kr=Ks+2lam T(F²).  For a nodeless ground F and lam>0,
 Ks is positive semidefinite and Kr is positive definite.  The gyroscopic
term conserves a positive quadratic energy.  Therefore the *probe* has no
exponential growing normal modes for finite p. These operators do NOT include
metric/brane-bending or GW perturbations and cannot establish 5D gravity stability.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, solve

@dataclass(frozen=True)
class ProbeParams:
    k: float = 1.0
    L: float = 1.0
    mass2: float = 0.5
    lam: float = 0.3
    kap_uv: float = 1.0
    kap_ir: float = 1.0
    charge_norm: float = 0.32
    def __post_init__(self):
        if min(self.k,self.L,self.mass2,self.lam,self.kap_uv,self.kap_ir,self.charge_norm)<=0:
            raise ValueError('positive parameters required for the stable-ground-state theorem')


def assemble(params,n):
    if n<8: raise ValueError('at least eight nodes')
    p=params; ys=np.linspace(0,p.L,n); K=np.zeros((n,n)); M=np.zeros_like(K)
    # Keep element quadrature data for nonlinear assembly.
    cells=[]
    for j in range(n-1):
        h=ys[j+1]-ys[j]; locK=np.zeros((2,2)); locM=np.zeros((2,2));locW=np.zeros((2,2)); qdata=[]
        for xi in (0.5-1/np.sqrt(12),0.5+1/np.sqrt(12)):
            N=np.array([1.-xi,xi]); D=np.array([-1./h,1./h])
            yy=ys[j]+xi*h; P=np.exp(-4*p.k*yy); W=np.exp(-2*p.k*yy); weight=h/2.
            locK+=weight*P*np.outer(D,D)
            locM+=weight*W*np.outer(N,N)
            locW+=weight*P*np.outer(N,N)
            qdata.append((N,P*weight))
        inds=np.ix_([j,j+1],[j,j+1]); K[inds]+=locK+p.mass2*locW;M[inds]+=locM
        cells.append((j,qdata))
    K[0,0]+=p.kap_uv/2
    K[-1,-1]+=np.exp(-4*p.k*p.L)*p.kap_ir/2
    return ys,K,M,cells


def weighted_nonlinear(F,cells,n):
    """Return cubic residual vector and the symmetric quadratic weighted matrix.

    residual_i = int P F³ N_i,  matrix_ij=int P F² N_i N_j.
    """
    C=np.zeros(n); T=np.zeros((n,n))
    for j,data in cells:
        for N,wp in data:
            value=np.dot(F[j:j+2],N)
            C[j:j+2]+=wp*value**3*N
            T[j:j+2,j:j+2]+=wp*value**2*np.outer(N,N)
    return C,T


def equilibrium(params=ProbeParams(),n=65,tol=1e-11):
    y,K0,M,cells=assemble(params,n)
    eig,vec=eigh(K0,M,subset_by_index=[0,0]); F=vec[:,0]
    if np.dot(F,np.ones(n))<0: F=-F
    F=params.charge_norm*F/np.sqrt(F@M@F)
    w2=float(eig[0]); q2=params.charge_norm**2
    history=[]
    for iteration in range(24):
        C,T=weighted_nonlinear(F,cells,n)
        R=K0@F+params.lam*C-w2*M@F
        RN=float(F@M@F-q2)
        scale=max(np.linalg.norm(K0@F),np.linalg.norm(w2*M@F),1)
        error=max(np.linalg.norm(R)/scale,abs(RN))
        history.append(error)
        if error<tol: break
        J=K0+3*params.lam*T-w2*M
        augmented=np.empty((n+1,n+1))
        augmented[:n,:n]=J
        augmented[:n,n]=-M@F
        augmented[n,:n]=2*F@M
        augmented[n,n]=0
        delta=solve(augmented,np.r_[-R,-RN],assume_a='sym' if False else 'gen')
        step=1.0
        while step>1e-5:
            nf=F+step*delta[:n];nw2=w2+step*delta[n]
            nc,_=weighted_nonlinear(nf,cells,n)
            resid=K0@nf+params.lam*nc-nw2*M@nf
            normerr=max(np.linalg.norm(resid)/scale,abs(nf@M@nf-q2))
            if normerr<error:
                F,w2=nf,nw2;break
            step*=0.5
        else: raise RuntimeError('damped Newton step failed')
    else: raise RuntimeError('Newton did not converge')
    C,T=weighted_nonlinear(F,cells,n)
    Ks=K0+params.lam*T-w2*M
    Kr=Ks+2*params.lam*T
    if np.min(F)<=0: raise RuntimeError('Ground eigenfunction has a node')
    return {'y':y,'F':F,'omega':float(np.sqrt(w2)),'omega2':w2,
            'K0':K0,'M':M,'Ks':Ks,'Kr':Kr,'T':T,'history':history,
            'nonlinear_residual':float(np.linalg.norm(Ks@F)),
            'phase_null_relative':float(np.linalg.norm(Ks@F)/np.linalg.norm(K0@F)),
            'min_eig_Ks':float(eigh(Ks,M,eigvals_only=True,subset_by_index=[0,0])[0]),
            'min_eig_Kr':float(eigh(Kr,M,eigvals_only=True,subset_by_index=[0,0])[0]),
            'linear_omega2':float(eigh(K0,M,eigvals_only=True,subset_by_index=[0,0])[0])}


def dynamical_matrix(solution,p3=0):
    M=solution['M']; Ks=solution['Ks'];Kr=solution['Kr'];n=len(M);omega=solution['omega']
    diag=np.eye(n);Z=np.zeros_like(M)
    ur=solve(M,Kr,assume_a='pos')+p3*p3*diag
    us=solve(M,Ks,assume_a='pos')+p3*p3*diag
    return np.block([[Z,Z,diag,Z],[Z,Z,Z,diag],[-ur,Z,Z,2*omega*diag],
                     [Z,-us,-2*omega*diag,Z]])


def normal_frequencies(solution,p3,imag_tolerance=1e-6):
    rates=np.linalg.eigvals(dynamical_matrix(solution,p3))
    max_real=float(np.max(np.abs(rates.real)))
    # Imaginary-conjugate pairs are twice-counted, take positive frequencies.
    positives=np.sort(rates.imag[rates.imag>imag_tolerance])
    return {'p3':float(p3),'min_frequency':float(positives[0]),
            'positive_frequencies':positives[:8].tolist(),
            'max_abs_growth_rate':max_real,
            'all_eigenvalues':rates}


def energy(solution, p3, q, qdot):
    n=len(solution['F']);r,s=np.split(q,2);v,w=np.split(qdot,2)
    M=solution['M'];Kr=solution['Kr']+p3*p3*M;Ks=solution['Ks']+p3*p3*M
    return .5*(v@M@v+w@M@w+r@Kr@r+s@Ks@s)


def energy_derivative(solution,p3,q,qdot):
    n=len(solution['F']);r,s=np.split(q,2);v,w=np.split(qdot,2)
    M=solution['M'];Kr=solution['Kr']+p3*p3*M;Ks=solution['Ks']+p3*p3*M
    acc_r=2*solution['omega']*w -solve(M,Kr@r,assume_a='pos')
    acc_s=-2*solution['omega']*v-solve(M,Ks@s,assume_a='pos')
    return float(v@M@acc_r+w@M@acc_s+v@Kr@r+w@Ks@s)


def goldstone_sound_speed_squared(solution):
    """Schur-complement effective dispersion of the phase zero mode, p->0.

    Project onto F, eliminate the gapped radial field Kr^-1 M F:
      c_ph² = [1 + 4omega²(F^T M Kr^-1 M F)/(F^T M F)]^-1.
    Gauge-fixed *probe* result, excluding gravity.
    """
    F=solution['F'];M=solution['M'];Kr=solution['Kr']; w=solution['omega']
    g=M@F
    susceptibility=4*w*w*(g@solve(Kr,g,assume_a='pos'))/(F@M@F)
    return float(1/(1+susceptibility))
