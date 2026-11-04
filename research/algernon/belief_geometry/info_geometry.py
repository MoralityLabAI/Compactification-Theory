"""Finite categorical information geometry in natural-parameter coordinates.

Results are ordinary analytic identities. Numerical functions run on CPU using
math standard library; they are diagnostic references, not model activations.
"""
from __future__ import annotations
from math import exp, log
from typing import Sequence


def log_partition(theta: Sequence[float]) -> float:
    if not theta:
        raise ValueError("empty logit vector")
    m=max(theta)
    return m+log(sum(exp(x-m) for x in theta))


def softmax(theta: Sequence[float]) -> list[float]:
    a=log_partition(theta)
    return [exp(x-a) for x in theta]


def fisher(theta: Sequence[float]) -> list[list[float]]:
    """Fisher information = Hessian of log partition in full logits.

    The all-ones direction is its kernel because softmax is translation-invariant.
    """
    p=softmax(theta)
    return [[p[i]*((1.0 if i==j else 0.0)-p[j])
             for j in range(len(p))] for i in range(len(p))]


def kl_categorical(theta, eta) -> float:
    if len(theta)!=len(eta):
        raise ValueError("logit dimensions differ")
    p=softmax(theta)
    at,ae=log_partition(theta),log_partition(eta)
    return sum(p_i*(t-e-at+ae) for p_i,t,e in zip(p,theta,eta))


def partition_bregman(theta,eta) -> float:
    """KL(P_theta||P_eta) = A(eta)-A(theta)-p_theta·(eta-theta)."""
    if len(theta)!=len(eta):
        raise ValueError("logit dimensions differ")
    p=softmax(theta)
    return log_partition(eta)-log_partition(theta)-sum(
        x*(e-t) for x,e,t in zip(p,eta,theta))


def alpha_tie(alpha,beta,theta):
    """Expected reward difference between two fixed alpha plans."""
    if len(alpha)!=len(beta) or len(alpha)!=len(theta):
        raise ValueError("plan dimensions differ")
    return sum((a-b)*p for a,b,p in zip(alpha,beta,softmax(theta)))


def curved_tie_y(x):
    """Two natural coordinates (x,y) and third zero: e^x+e^y=1."""
    if x >= 0:
        raise ValueError("x must be negative for this boundary")
    from math import log1p
    return log1p(-exp(x))


def curved_tie_second_derivative(x):
    """d²/dx² log(1-e^x) = -e^x/(1-e^x)^2 < 0."""
    if x >= 0:
        raise ValueError("x must be negative for this boundary")
    u=exp(x)
    return -u / (1-u)**2
