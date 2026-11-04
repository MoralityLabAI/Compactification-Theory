"""QO-02: exact polynomial-time minimax state aggregation for hit/hold chains.

The assumptions are special: one absorbing goal, per-state hit probabilities,
and all non-hit mass stays at its original transient state. Full arbitrary
controlled Markov state aggregation remains a different optimization problem.
"""
from fractions import Fraction as Q


def optimal_rate_groups(rates, max_groups):
    """Minimize maximum within-group hit-probability range / 2.

    Returns (optimal TV radius, original-order group labels, sorted contiguous
    groups). Exact O(n² k) dynamic programming for k=min(n,max_groups).
    """
    p = tuple(Q(x) for x in rates)
    n = len(p)
    if n == 0 or max_groups < 1 or any(x < 0 or x > 1 for x in p):
        raise ValueError('nonempty probabilities in [0,1] and at least one group required')
    k = min(n, max_groups)
    order = sorted(range(n), key=lambda i: (p[i], i))
    values = [p[i] for i in order]
    dp = [[None] * (n + 1) for _ in range(k + 1)]
    back = [[None] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = Q(0)
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            for h in range(j - 1, i):
                if dp[j - 1][h] is None:
                    continue
                cost = max(dp[j - 1][h], (values[i - 1] - values[h]) / 2)
                if dp[j][i] is None or cost < dp[j][i]:
                    dp[j][i], back[j][i] = cost, h
    assert dp[k][n] is not None
    blocks = []
    i, j = n, k
    while j > 0:
        h = back[j][i]
        assert h is not None
        blocks.append(tuple(order[h:i]))
        i, j = h, j - 1
    assert i == 0
    blocks.reverse()
    assignment = [None] * n
    for label, block in enumerate(blocks):
        for s in block:
            assignment[s] = label
    assert set(assignment) == set(range(k))
    return dp[k][n], tuple(assignment), tuple(blocks)


def optimal_with_state_bits(rates, bits):
    """Goal must have its own class, leaving 2**bits-1 transient classes."""
    if not isinstance(bits, int) or bits < 1:
        raise ValueError('the goal-visible one-goal family requires >=1 bit')
    return optimal_rate_groups(rates, (1 << bits) - 1)


def uniform_hit_closed_form(n, bits):
    """For p_i=(i+1)/(n+1), exact minimum uniform one-step TV error."""
    if n < 1 or bits < 1:
        raise ValueError('n>=1 and bits>=1 required')
    k = min(n, (1 << bits) - 1)
    max_group_size = (n + k - 1) // k
    return Q(max_group_size - 1, 2 * (n + 1))


def min_bits_for_uniform_error(n, error):
    """Exact minimum goal-visible bits for uniform rates and tolerated TV error."""
    eps = Q(error)
    if n < 1 or eps < 0:
        raise ValueError('n>=1 and nonnegative error required')
    max_block_size = min(n, (2 * eps * (n + 1)).numerator // (2 * eps * (n + 1)).denominator + 1)
    required_groups = (n + max_block_size - 1) // max_block_size
    total_states = required_groups + 1  # one independent absorbing goal state
    return (total_states - 1).bit_length()
