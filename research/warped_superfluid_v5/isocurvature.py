"""One genuinely diffeomorphism-invariant entropy projection at linear order.

Three real bulk scalars q^I=(chi,x,v), all with canonical flat target metric.
The background q(t,y) depends on t and the fifth coordinate y only.
At a rank-two point, the field-space unit normal
  n_I = epsilon_IJK (partial_t q)^J (partial_y q)^K /
        |partial_t q cross partial_y q|
annihilates both background gradients. Therefore
  S = n_I delta q^I
is unchanged by arbitrary infinitesimal scalar coordinate shifts
  delta q^I -> delta q^I + xi^t partial_t q^I + xi^y partial_y q^I.
It is also invariant under global SO(2) rotations of (x,v).

IMPORTANT: This proves linear diffeomorphism invariance of ONE scalar
projection. It does not by itself eliminate scalar metric constraints,
construct an entire gauge-invariant mode basis, or determine stability.
It is undefined at field-gradient-rank-deficient points.
"""
from __future__ import annotations
import numpy as np


def field_gradients(model, state, spacing):
    """Return q_t and q_y arrays with shape (3,n), including brane slopes."""
    s=np.asarray(state)
    if s.ndim!=2 or s.shape[0]!=10:
        raise ValueError('Expected state shape (10,n)')
    A,B,chi,x,v,At,Bt,ct,xt,vt=s
    boundary=model.boundary_slopes(A,B,chi,x,v)
    qy=np.stack([model.differences(f,spacing,*boundary[name])[0]
                 for f,name in ((chi,'chi'),(x,'x'),(v,'y'))])
    qt=np.stack((ct,xt,vt))
    return qt,qy


def unit_entropy_normal(model,state,spacing,threshold=1e-12):
    """Oriented normal to span{partial_t q,partial_y q} in field space."""
    qt,qy=field_gradients(model,state,spacing)
    cross=np.cross(qt.T,qy.T).T
    mag=np.linalg.norm(cross,axis=0)
    if np.any(mag<=threshold):
        raise ValueError('Background field gradients lack rank two')
    return cross/mag,mag


def entropy_projection(model,state,tangent,spacing):
    """The unique (up to orientation) instantaneous field-only GI scalar."""
    v=np.asarray(tangent)
    if v.shape!=np.asarray(state).shape:
        raise ValueError('Tangent must match background shape')
    n,mag=unit_entropy_normal(model,state,spacing)
    dphi=np.stack((v[2],v[3],v[4]))
    return np.sum(n*dphi,axis=0)


def shifted_scalar_tangent(model,state,tangent,spacing,xi_t,xi_y):
    """Apply the scalar part of an infinitesimal (t,y) coordinate shift."""
    qt,qy=field_gradients(model,state,spacing)
    out=np.array(tangent,copy=True)
    out[2:5]+=qt*np.asarray(xi_t)+qy*np.asarray(xi_y)
    # Metric components would also transform under a true diffeomorphism;
    # only the scalar part is needed to demonstrate S invariance.
    return out


def u1_rotate_state(state,angle):
    """Rotate scalar Cartesian pair and their velocities, preserving action."""
    out=np.array(state,copy=True)
    c,s=np.cos(angle),np.sin(angle)
    for i,j in ((3,4),(8,9)):
        x,v=out[i].copy(),out[j].copy()
        out[i]=c*x-s*v
        out[j]=s*x+c*v
    return out
