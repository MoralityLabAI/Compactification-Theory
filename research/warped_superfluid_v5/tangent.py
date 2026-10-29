"""WS-05: constraint-aware tangent evolution about nonlinear charged WS-04.

All 10 WS-04 dynamical fields are perturbed in ONE tangent system, including
metric, GW scalar, charged amplitude and phase. Differentiation is performed
by complex step on the full method-of-lines RHS, INCLUDING both brane Robin
and Israel boundary terms. It is exact for the discretized holomorphic RHS
up to roundoff and independent of amplitude of the tangent perturbation.

This is a conformal-*gauge-fixed* evolution. Physical U(1) charge is
invariant, and brane-proper-time Hubble is meaningful on a specified brane,
but no claim is made of a fully gauge-invariant charged cosmological spectrum.
WS-04 temporal and spatial truncation errors still apply.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.integrate import solve_ivp

ROOT_FIELDS=('A','B','chi','x','y','At','Bt','ct','xt','yt')


def base_initial_state(model,n):
    initial=model.initial_slice(n)
    grid=initial['grid']
    state=np.stack([initial[name] for name in ROOT_FIELDS])
    return grid, state


def initial_tangent(model,n=65, parameter='bulk_amp', step=1e-4):
    """Derivative along the constraint-satisfying WS-04 initial-data family.

    bulk_amp and bulk_omega are INITIAL DATA parameters only: the nonlinear
    equations and boundary potentials are identical across family members.
    Initial derivative uses a symmetric finite difference because the
    nonlinear shooting/bracketing routine is not holomorphic.
    """
    if parameter not in ('bulk_amp','bulk_omega'):
        raise ValueError('Only initial-data-only parameters may be varied')
    value=getattr(model,parameter)
    if parameter=='bulk_amp' and value-step<=0:
        raise ValueError('Amplitude variation would cross F=0')
    grid,s=base_initial_state(model,n)
    gp,plus=base_initial_state(replace(model,**{parameter:value+step}),n)
    gm,minus=base_initial_state(replace(model,**{parameter:value-step}),n)
    if not (np.allclose(grid,gp,atol=0,rtol=0) and
            np.allclose(grid,gm,atol=0,rtol=0)):
        raise RuntimeError('Initial-family coordinates differ')
    return grid,s,(plus-minus)/(2*step)


def jvp(model,t,state,tangent,dy,h=1e-24):
    """J(U)*dU via complex step, including state-dependent junction slopes.

    Asserts only method-of-lines derivative consistency, not continuum
    Einstein constraint preservation or gauge-invariant mode stability.
    """
    state=np.asarray(state)
    tangent=np.asarray(tangent)
    if state.shape!=tangent.shape or state.shape[0]!=10:
        raise ValueError('state and tangent must have shape (10,n)')
    n=state.shape[1]
    perturbed=(state.astype(complex)+1j*h*tangent).ravel()
    result=model.rhs(t,perturbed,n,dy).reshape(10,n)
    return np.imag(result)/h


def constraint_jvp(model,state,tangent,dy,h=1e-24):
    n=state.shape[1]
    complex_state=(state.astype(complex)+1j*h*tangent).reshape(10*n)
    c00,c0y=model.constraints(complex_state,dy)
    return np.imag(c00)/h,np.imag(c0y)/h


def junction_jvp(model,state,tangent,h=1e-24):
    state=np.asarray(state)
    tangent=np.asarray(tangent)
    if state.shape!=tangent.shape:
        raise ValueError('shape mismatch')
    u=state.astype(complex)+1j*h*tangent
    bc=model.boundary_slopes(*u[:5])
    return {key:np.imag(val)/h for key,val in bc.items()}


def charge_derivative(state,tangent,grid):
    A,B,chi,x,y,At,Bt,ct,xt,yt=state
    dA,dB,dc,dx,dy,dAt,dBt,dct,dxt,dyt=tangent
    w=np.exp(3*A)
    integrand=w*(3*dA*(x*yt-y*xt)+dx*yt+x*dyt-dy*xt-y*dxt)
    return float(np.trapezoid(integrand,grid))


def geometry_observables(state,grid):
    A,B,chi,x,y,At,Bt,ct,xt,yt=state
    H=float(At[0]*np.exp(-B[0]))
    # Constant-conformal-time proper length: slicing-dependent, NOT fully GI.
    length=float(np.trapezoid(np.exp(B),grid))
    charge=float(np.trapezoid(np.exp(3*A)*(x*yt-y*xt),grid))
    return np.array([H,length,charge])


def geometry_derivative(state,tangent,grid):
    A,B,chi,x,y,At,Bt,ct,xt,yt=state
    dA,dB,dc,dx,dy,dAt,dBt,dct,dxt,dyt=tangent
    dH=float(np.exp(-B[0])*(dAt[0]-At[0]*dB[0]))
    dlength=float(np.trapezoid(np.exp(B)*dB,grid))
    return np.array([dH,dlength,charge_derivative(state,tangent,grid)])


def phase_rotation_tangent(state):
    """Exact U(1) symmetry tangent for all states and all brane junctions."""
    p=np.zeros_like(state)
    p[3]=-state[4]
    p[4]=state[3]
    p[8]=-state[9]
    p[9]=state[8]
    return p


def evolve_tangent(model,n=65,time_end=.12,samples=7,initial_parameter='bulk_amp',
                   family_step=1e-4,rtol=3e-9):
    """Evolve a constrained 1-parameter initial family through its full JVP.

    The returned response is useful for finite-time, gauge-fixed diagnostics,
    not eigenvalue stability of a nonlinear time-dependent 5D geometry.
    """
    grid,s0,p0=initial_tangent(model,n,initial_parameter,family_step)
    dy=float(grid[1]-grid[0]);ts=np.linspace(0,time_end,samples)
    count=s0.size
    def ode(t,v):
        base=v[:count].reshape(10,n)
        pert=v[count:].reshape(10,n)
        f=model.rhs(t,base.ravel(),n,dy)
        fp=jvp(model,t,base,pert,dy).ravel()
        return np.concatenate((f,fp))
    sol=solve_ivp(ode,(0,time_end),np.concatenate((s0.ravel(),p0.ravel())),
                  t_eval=ts,method='DOP853',rtol=rtol,atol=max(1e-12,rtol*1e-3),
                  max_step=dy/5)
    if not sol.success: raise RuntimeError(sol.message)
    bases=sol.y[:count,:].T.reshape(samples,10,n)
    derivs=sol.y[count:,:].T.reshape(samples,10,n)
    observables=np.array([geometry_observables(s,grid) for s in bases])
    tangents=np.array([geometry_derivative(s,d,grid) for s,d in zip(bases,derivs)])
    linear_c=[]
    for s,d in zip(bases,derivs):
        dc00,dc0y=constraint_jvp(model,s,d,dy)
        linear_c.append([float(np.max(np.abs(dc00[3:-3]))),
                         float(np.max(np.abs(dc0y[3:-3])))])
    return {'t':ts,'grid':grid,'states':bases,'tangent_states':derivs,
            'observables':observables,'linear_observables':tangents,
            'linear_constraint_errors':np.array(linear_c)}


def finite_difference_response(model,tangent_output,parameter='bulk_amp',epsilon=5e-4,rtol=4e-9):
    """Independent FULL nonlinear centered response; expected O(epsilon²)."""
    if parameter not in ('bulk_amp','bulk_omega'):
        raise ValueError('Only initial data parameters supported')
    grid=tangent_output['grid'];n=len(grid)
    t=tangent_output['t']
    t_end=float(t[-1]);samples=len(t)
    value=getattr(model,parameter)
    responses=[]
    for delta in (-epsilon,epsilon):
        m=replace(model,**{parameter:value+delta})
        run=m.evolve(n=n,time_end=t_end,samples=samples,rtol=rtol)
        if not np.allclose(run['grid'],grid,rtol=0,atol=0):
            raise RuntimeError('Grid mismatch')
        responses.append(np.array([geometry_observables(s,grid)
                                  for s in run['states']]))
    finite_difference=(responses[1]-responses[0])/(2*epsilon)
    tangent=tangent_output['linear_observables']
    absolute_err=np.abs(finite_difference-tangent)
    relative_err=absolute_err/np.maximum(np.abs(tangent),1e-6)
    return {'finite_difference':finite_difference,'analytic_tangent':tangent,
            'absolute_error':absolute_err,'max_relative_error':np.max(relative_err,axis=0)}
