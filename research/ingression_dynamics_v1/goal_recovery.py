"""Goal-aware quotient diagnostics and finite Markov reachability, exact arithmetic."""
from fractions import Fraction as Q

def goal_profiles(states, goals):
    """Immutable Boolean goal signatures, indexed by microscopic state."""
    return {s: tuple(bool(g[s]) for g in goals) for s in states}

def goal_visible(states, projection, goals):
    profiles = goal_profiles(states, goals)
    by_fibre = {}
    for s in states:
        z = projection[s]
        if z in by_fibre and by_fibre[z] != profiles[s]:
            return False
        by_fibre[z] = profiles[s]
    return True

def min_goal_classes(states, goals):
    """Necessary number of distinguishable abstract labels for all goals."""
    return len(set(goal_profiles(states, goals).values()))

def goal_bits_lower_bound(states, goals):
    """Fixed-length code bit lower bound to preserve all goal signatures."""
    return (min_goal_classes(states, goals) - 1).bit_length()

def goal_radius_absolute(states, projection, goal):
    """Exact optimal worst-case |g(s)-q(pi(s))| with scores q in [0,1]."""
    by_fibre = {}
    for s in states:
        by_fibre.setdefault(projection[s], set()).add(int(bool(goal[s])))
    return Q(1,2) if any(len(v)==2 for v in by_fibre.values()) else Q(0)

def hit_probability(kernel, initial, target, horizon):
    """Probability of visiting the target at any t<=horizon transitions."""
    assert horizon >= 0 and initial in kernel
    assert all(sum(row.values(), Q(0)) == 1 and all(p >= 0 for p in row.values())
               for row in kernel.values())
    values = {s: Q(int(s in target)) for s in kernel}
    for _ in range(horizon):
        values = {s: Q(1) if s in target else
                  sum((p*values[t] for t,p in row.items()), Q(0))
                  for s,row in kernel.items()}
    return values[initial]
