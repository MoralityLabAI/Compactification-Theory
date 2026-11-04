"""LSV-04: exact deployment surrogate for a frozen learned recurrent model.

Local hard-tanh is 1-Lipschitz. All runtime updates and the certificate use
fractions.Fraction. The Torch-trained source checkpoint is quantized at export;
no assurance about compiled floating-point execution is implied.
"""
from __future__ import annotations
import hashlib, json
from fractions import Fraction as Q
from pathlib import Path

from small_gain import Certificate
from delayed import audit_delayed_trace, delayed_superblock

TARGET = ((Q(0), Q(9,5)), (Q(1,5), Q(0)))
ACTIVATION_LIMIT = Q(1)
CLEAN_LIMIT = Q(9,10)
DAMAGE_LIMIT = Q(3,2)
WEIGHTS = (Q(3),Q(1))


def clip(v: Q) -> Q:
    return max(-ACTIVATION_LIMIT, min(ACTIVATION_LIMIT, v))


def model_matrix(ck: dict) -> tuple[tuple[Q,...],...]:
    if ck.get('schema') != 'LSV-04/rational-checkpoint-v1':
        raise ValueError('unrecognized checkpoint schema')
    if ck.get('class') != 'cpu_trained_two_node_nonlinear_recurrent_proxy':
        raise ValueError('unrecognized model class')
    if ck.get('target_gains') != [["0", "9/5"], ["1/5", "0"]]:
        raise ValueError('task contract changed')
    pair=ck.get('weights',{})
    if set(pair) != {'a01','a10'}:
        raise ValueError('expected precisely two trainable weights')
    encoded=json.dumps(pair,sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(encoded).hexdigest()!=ck.get('weights_sha256'):
        raise ValueError('trained weight fingerprint mismatch')
    a,b = Q(pair['a01']), Q(pair['a10'])
    if min(a,b)<0:
        raise ValueError('this certificate expects nonnegative coupling')
    return ((Q(0), a),(b, Q(0)))


def load_checkpoint(path: str|Path) -> dict:
    data=json.loads(Path(path).read_text())
    model_matrix(data)
    return data


def make_certificate(W: tuple[tuple[Q,...],...]):
    gains=tuple(tuple(abs(v) for v in row) for row in W)
    cert=Certificate.from_weights(gains,WEIGHTS)
    # The declared task has true target x*: |x*_j| <= 0.9.
    # b_i = x*_i - sum_j TARGET[i,j]*clip(x*_j), so
    # e_i(new) = sum_j W[i,j]*(clip(stale x_j)-clip(x*_j))
    #           + sum_j (W[i,j]-TARGET[i,j])*clip(x*_j).
    # Clip is globally 1-Lipschitz; the second term is bounded by eta.
    eta=tuple(CLEAN_LIMIT*sum(abs(w-a) for w,a in zip(row,rrow))
              for row,rrow in zip(W,TARGET))
    return cert,eta


def forcing(target: tuple[Q,Q]) -> tuple[Q,Q]:
    if any(abs(v)>CLEAN_LIMIT for v in target):
        raise ValueError('target outside declared task domain')
    return tuple(target[i]-sum((a*clip(x) for a,x in zip(TARGET[i],target)),Q(0))
                 for i in range(2))


def rollout(W, target, damage, cycles=24, delay=0):
    if cycles<1 or delay<0 or not isinstance(cycles,int) or not isinstance(delay,int):
        raise ValueError('invalid rollout length/delay')
    if any(abs(v)>DAMAGE_LIMIT for v in damage):
        raise ValueError('damage exceeds declared budget')
    b=forcing(target)
    initial=tuple(t+d for t,d in zip(target,damage))
    history=[initial]*(delay+1)
    states=[initial]
    schedule=[]
    ages=[]
    for t in range(2*cycles):
        i=t%2
        lag=(0 if delay==0 else delay)
        selected=[states[t-lag] if t-lag>=0 else history[t-lag+delay]
                  for j in range(2)]
        # Full-vector stale read for a named age. Index j extracted below.
        next_i=b[i]+sum((W[i][j]*clip(selected[j][j]) for j in range(2)),Q(0))
        state=list(states[-1]);state[i]=next_i
        states.append(tuple(state))
        schedule.append(i)
        ages.append((lag,lag))
    return history,states,schedule,ages


def audited_rollout(ck,target,damage,cycles=24,delay=0):
    W=model_matrix(ck)
    cert,eta=make_certificate(W)
    history, states, schedule, ages=rollout(W,target,damage,cycles,delay)
    herror=[tuple(x-y for x,y in zip(state,target)) for state in history]
    errors=[tuple(x-y for x,y in zip(state,target)) for state in states]
    audit=audit_delayed_trace(cert,herror,errors,schedule,ages,eta,2,delay)
    # This radius is certified on all declared tasks, not just the sampled trace.
    assert cert.error_norm(errors[-1]) <= cert.bound(cert.error_norm(errors[0]),eta,
                      (2*cycles)//delayed_superblock(2,delay))
    return states,audit,cert,eta


def ex_ante_bound(cert,eta,cycles,delay):
    # Checkable without target labels: a declared maximum injected perturbation.
    H=max(DAMAGE_LIMIT/w for w in cert.weights)
    return cert.bound(H,eta,(2*cycles)//delayed_superblock(2,delay))


def guaranteed_absolute_goal(cert,eta,cycles,delay,tolerance=Q(1,5)):
    # Convert weighted infinity bound to a bound on max_i |e_i|.
    return max(cert.weights)*ex_ante_bound(cert,eta,cycles,delay)<=tolerance
