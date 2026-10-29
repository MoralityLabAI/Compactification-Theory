"""Fixed-metric scalar-sector probe spectrum: NOT the coupled radion spectrum.

A perturbation of the GW bulk scalar around the exact BPS metric has a
quadratic form with p=e^-4warp(z), w=e^-2warp(z), q=p V''(chi).
Physical radion perturbations mix chi with 5D scalar metric fluctuations;
probe positivity cannot establish physical radion stability.
"""
import numpy as np
from scipy.linalg import eigh


def probe_spectrum(system, nodes=129, count=4):
    if nodes < count+3:
        raise ValueError('Need sufficient resolution')
    z=np.linspace(0,system.proper_L,nodes)
    dz=z[1]-z[0]
    K=np.zeros((nodes,nodes)); M=np.zeros_like(K)
    xis=(0.5-0.5/np.sqrt(3),0.5+0.5/np.sqrt(3))
    for i in range(nodes-1):
        ix=np.ix_([i,i+1],[i,i+1])
        for xi in xis:
            zq=z[i]+xi*dz
            warp=system.warp(zq)
            ch=system.c_background(zq)
            p=np.exp(-4*warp)
            w=np.exp(-2*warp)
            v2=system.d2V(ch)
            shape=np.array([1-xi,xi]); grad=np.array([-1,1])/dz
            K[ix] += .5*dz*(p*np.outer(grad,grad)+p*v2*np.outer(shape,shape))
            M[ix] += .5*dz*w*np.outer(shape,shape)
    pL=np.exp(-4*system.warp(system.proper_L))
    K[0,0] += system.u+.5*system.stiffness_uv
    K[-1,-1] += pL*(.5*system.stiffness_ir-system.u)
    vals=eigh(K,M,eigvals_only=True,subset_by_index=[0,count-1])
    return vals
