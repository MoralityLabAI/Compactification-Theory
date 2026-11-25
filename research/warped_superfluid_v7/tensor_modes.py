"""5D transverse-traceless gravitational modes on static charged background.

Metric ds²=-e^(2N) dt² + e^(2A) d x_3² + dy².
No tensor anisotropic stress on homogeneous isotropic branes => natural
Neumann h'_TT=0 boundaries. At p3 Fourier momentum, Sturm-Liouville:
 -(e^(N+3A)h')' + p3² e^(N+A) h = Omega² e^(-N+3A) h.

These TT modes are genuinely gauge-invariant in 3D rotational decomposition.
They DO NOT determine the mixed radion/Goldstone/ghost scalar stability.
The low-energy speed is measured with UV brane coordinates N(0)=A(0)=0.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import simpson


def tensor_matrices(solution, nodes=129):
    if nodes<5:raise ValueError('Need >=5 nodes')
    yy=solution['y']; state=solution['fields'];L=yy[-1]
    y=np.linspace(0,L,nodes)
    A=np.interp(y,yy,state[0]);N=np.interp(y,yy,state[1]);
    K=np.zeros((nodes,nodes));M=np.zeros_like(K);P=np.zeros_like(K)
    gauss=(.5-1/np.sqrt(12),.5+1/np.sqrt(12))
    for i in range(nodes-1):
        h=y[i+1]-y[i]
        for q in gauss:
            Ni=np.array([1-q,q]);dNi=np.array([-1/h,1/h]);
            n=N[i]*(1-q)+N[i+1]*q;a=A[i]*(1-q)+A[i+1]*q
            si=np.ix_([i,i+1],[i,i+1]);fac=.5*h
            K[si]+=fac*np.exp(n+3*a)*np.outer(dNi,dNi)
            M[si]+=fac*np.exp(-n+3*a)*np.outer(Ni,Ni)
            P[si]+=fac*np.exp(n+a)*np.outer(Ni,Ni)
    return y,K,M,P


def tensor_modes(solution,nodes=129,p3=0.0,count=4):
    if p3<0:raise ValueError('nonnegative p')
    _,K,M,P=tensor_matrices(solution,nodes)
    masses,eigen=eigh(K+p3*p3*P,M,subset_by_index=[0,count-1],driver='gvx')
    if masses[0]<-1e-9:raise ValueError('tensor energy has negative squared frequency')
    return np.maximum(masses,0.),eigen


def tensor_speed_squared(solution):
    y=solution['y'];A,N=solution['fields'][:2]
    return float(simpson(np.exp(N+A),x=y)/simpson(np.exp(-N+3*A),x=y))


def tensor_energy(solution,p3,profile,profile_dot,nodes=None):
    n=len(profile)
    if nodes is not None and nodes!=n:raise ValueError('grid mismatch')
    _,K,M,P=tensor_matrices(solution,n)
    return float(.5*(profile_dot@M@profile_dot+profile@(K+p3*p3*P)@profile))


def tensor_energetic_positive(solution,p3=0.2,nodes=65):
    _,K,M,P=tensor_matrices(solution,nodes)
    v=eigh(K+p3*p3*P,M,eigvals_only=True,subset_by_index=[0,0])[0]
    return float(v)
