"""LSV-02: exact-rational audit for boundedly stale asynchronous repair."""
from __future__ import annotations
from fractions import Fraction as Q
from small_gain import Certificate, _q


def delayed_superblock(block_size: int, max_delay: int) -> int:
    """Smallest multiple of B at least B+D, for aligned B-fair schedules."""
    if not isinstance(block_size, int) or not isinstance(max_delay, int) or block_size < 1 or max_delay < 0:
        raise ValueError("B must be positive and D nonnegative")
    return block_size * (1 + (max_delay + block_size - 1)//block_size)


def audit_delayed_trace(cert: Certificate, history, states, schedule,
                        delays, limits, block_size: int, max_delay: int):
    """Check every stale local read, the fair schedule, and the lagged bound.

    history rows are e(-D),...,e(0); states rows are e(0),...,e(T).
    delays[t][j] specifies how many steps old coordinate j was when
    component schedule[t] updated from e(t) to e(t+1).
    """
    cert.validate()
    L = delayed_superblock(block_size, max_delay)
    n = len(cert.weights)
    if block_size < n or len(history) != max_delay + 1:
        raise ValueError("invalid history length or fair-block size")
    if not schedule or len(states) != len(schedule) + 1 or len(delays) != len(schedule):
        raise ValueError("trace, schedule, and per-step delays must align")
    if len(schedule) % block_size or any(set(schedule[k:k+block_size]) != set(range(n))
            for k in range(0, len(schedule), block_size)):
        raise ValueError("each aligned block must update all coordinates")
    if any(not isinstance(i, int) or not 0 <= i < n for i in schedule):
        raise ValueError("invalid update index")
    historic = [tuple(_q(v) for v in x) for x in history]
    trace = [tuple(_q(v) for v in x) for x in states]
    if any(len(x) != n for x in historic + trace) or historic[-1] != trace[0]:
        raise ValueError("history/state mismatch")
    d = tuple(_q(x) for x in limits)
    cert.scaled_noise(d)
    initial_bound = max(cert.error_norm(x) for x in historic)
    for t, i in enumerate(schedule):
        read = delays[t]
        if len(read) != n or any(not isinstance(k,int) or not 0 <= k <= max_delay for k in read):
            raise ValueError(f"invalid stale-read delay at step {t}")
        before, after = trace[t], trace[t+1]
        if any(after[j] != before[j] for j in range(n) if j != i):
            raise ValueError(f"nonlocal update at step {t}")
        source = [((trace[t-k] if t-k >= 0 else historic[t-k+max_delay])[j])
                  for j,k in enumerate(read)]
        ceiling = sum((a*abs(v) for a,v in zip(cert.gains[i], source)), Q(0)) + d[i]
        if abs(after[i]) > ceiling:
            raise ValueError(f"stale gain contract violated at step {t}")
        if cert.error_norm(after) > cert.bound(initial_bound,d,(t+1)//L):
            raise AssertionError(f"delayed bound violated at step {t}")
    return {"steps":len(schedule),"B":block_size,"D":max_delay,"superblock":L,
            "completed_superblocks":len(schedule)//L,
            "initial_history_max":str(initial_bound),
            "final_error_norm":str(cert.error_norm(trace[-1])),
            "certified_bound":str(cert.bound(initial_bound,d,len(schedule)//L))}