"""Finite controlled quotient audits, with exact rational stochastic arithmetic."""
from fractions import Fraction

def commutes(micro_maps, macro_maps, projection):
    """True iff every supplied action respects the proposed quotient."""
    return all(
        projection[micro_map[s]] == macro_maps[action][projection[s]]
        for action, micro_map in micro_maps.items()
        for s in micro_map
    )

def projected_path(start, actions, micro_maps, projection):
    states = [projection[start]]
    state = start
    for action in actions:
        state = micro_maps[action][state]
        states.append(projection[state])
    return states

def macro_path(start, actions, macro_maps):
    states = [start]
    state = start
    for action in actions:
        state = macro_maps[action][state]
        states.append(state)
    return states

def pushforward(row, projection):
    result = {}
    for state, p in row.items():
        z = projection[state]
        result[z] = result.get(z, Fraction(0)) + p
    return result

def total_variation(p, q):
    return sum((abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q)), Fraction(0)) / 2

def one_action_error(micro_kernel, macro_kernel, projection):
    return max(
        total_variation(pushforward(row, projection), macro_kernel[projection[s]])
        for s, row in micro_kernel.items()
    )

def expected_return(kernel, initial, rewards, horizon):
    """One fixed action; stochastic row dictionaries must sum to 1."""
    dist = {initial: Fraction(1)}
    result = Fraction(0)
    for _ in range(horizon):
        result += sum((weight * rewards[s] for s, weight in dist.items()), Fraction(0))
        next_dist = {}
        for s, weight in dist.items():
            for target, p in kernel[s].items():
                next_dist[target] = next_dist.get(target, Fraction(0)) + weight * p
        dist = next_dist
    return result
