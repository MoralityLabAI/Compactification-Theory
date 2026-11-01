"""Exact finite-state dynamical closure and empirical conditional-variance risk."""
from fractions import Fraction

def closure_witness(transition, partition):
    """Return an aliasing pair violating closure, or None."""
    n = len(transition)
    if len(partition) != n or any(not isinstance(t,int) or t < 0 or t >= n for t in transition):
        raise ValueError("invalid finite dynamical system")
    seen = {}
    for x,z in enumerate(partition):
        nxt = partition[transition[x]]
        if z in seen and seen[z][1] != nxt:
            return seen[z][0], x
        seen[z] = (x,nxt)
    return None

def induced_dynamics(transition, partition):
    witness = closure_witness(transition, partition)
    if witness is not None:
        raise ValueError(f"non-closed partition: witness={witness}")
    return {partition[x]:partition[transition[x]] for x in range(len(transition))}

def closure_risk(transition, observable, counts=None):
    """Exact min_g E[(observable(F(X))-g(observable(X)))**2]."""
    n = len(transition)
    if len(observable) != n:
        raise ValueError("length mismatch")
    if counts is None:
        counts = [1]*n
    if len(counts) != n or any(w < 0 for w in counts) or not sum(counts):
        raise ValueError("invalid empirical measure")
    groups = {}
    for x,w in enumerate(counts):
        if w:
            groups.setdefault(observable[x],[]).append((Fraction(observable[transition[x]]),w))
    loss = Fraction(0)
    for values in groups.values():
        mass = sum(w for v,w in values)
        avg = sum((v*w for v,w in values),Fraction(0))/mass
        loss += sum((w*(v-avg)**2 for v,w in values),Fraction(0))
    return loss/sum(counts)

def partitions(n):
    """All partitions as restricted-growth strings."""
    if n < 1: return
    def grow(p,m):
        if len(p) == n:
            yield tuple(p)
        else:
            for j in range(m+2):
                yield from grow(p+[j],max(m,j))
    yield from grow([0],0)
