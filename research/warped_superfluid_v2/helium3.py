"""Hermitian Weyl/Andreev TEXTURE toy for 3He-A quasiparticles.

Linearized 3He-A near a Weyl node:
 H~ tau_x c_perp p_x + tau_y c_perp p_y + tau_z c_parallel q_z.
For y-dependent c_perp, the continuum self-adjoint transverse term is
 -i tau_y sqrt(v) d_y sqrt(v) = -i tau_y(v d_y+v'/2).
We discretize using the exactly Hermitian anticommutator {V,D}/2,
where D is the skew-adjoint periodic central difference.
A ring with periodic smooth texture is an intentionally simplified comparison,
not a full microscopic helium-3 BdG treatment and NOT an RS extra dimension.
"""
import numpy as np
from scipy.linalg import eigvalsh

def periodic_velocity(n=65,L=2*np.pi,depth=1.0,v0=1.0):
    if n<7 or L<=0 or depth<0 or v0<=0:
        raise ValueError('Invalid grid or velocity parameters')
    y=L*np.arange(n)/n
    v=v0*np.exp(-0.5*depth*(1-np.cos(2*np.pi*y/L)))
    return y,v

def periodic_derivative(n,L):
    h=L/n
    D=np.zeros((n,n))
    for i in range(n):
        D[i,(i+1)%n] += 1/(2*h)
        D[i,(i-1)%n] -= 1/(2*h)
    return D

def hamiltonian(v,L=2*np.pi,p_x=0,p_z=0.3,c_parallel=1.5):
    """Matrix in Nambu pseudospin, with all quantities dimensionless."""
    v=np.asarray(v,dtype=float)
    n=v.size
    D=periodic_derivative(n,L)
    vd=0.5*(v[:,None]*D+D*v[None,:])
    moment=-1j*vd
    tx=np.array([[0,1],[1,0]],complex)
    ty=np.array([[0,-1j],[1j,0]],complex)
    tz=np.diag([1.0,-1.0]).astype(complex)
    H=np.kron(ty,moment)+p_x*np.kron(tx,np.diag(v))+p_z*c_parallel*np.kron(tz,np.eye(n))
    return H

def summary(depth=1.0,n=65,L=2*np.pi,pz=0.3):
    y,v=periodic_velocity(n,L,depth)
    H=hamiltonian(v,L,p_z=pz)
    Hu=hamiltonian(np.ones(n),L,p_z=pz)
    eig=np.sort(eigvalsh(H))
    eu=np.sort(eigvalsh(Hu))
    pos=eig[eig>=0]
    posu=eu[eu>=0]
    return {'y':y,'velocity':v,'hermiticity_residual':float(np.max(np.abs(H-H.conj().T))),
            'particle_hole_spectrum_residual':float(np.max(np.abs(eig+eig[::-1]))),
            'lowest_positive_eigenvalues':pos[:8],
            'uniform_lowest_positive_eigenvalues':posu[:8],
            'gap_min':float(pos[0]),'gap_uniform':float(posu[0]),
            'local_velocity_min':float(v.min()),'local_velocity_max':float(v.max()),
            'texture_strength':depth}
