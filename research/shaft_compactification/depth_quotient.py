"""Exact finite objective quotients and substrate-dependent observation coverage.

Independent from model weights, device operations and communication channels.
A substrate specifies which of four latent objective bits is visible at each
time step. The monitor only sees the designated substrate/depth observations.
"""

import json
from math import ceil, log2

GOALS = range(16)


def observed_signature(goal: int, horizon: int, substrates: tuple[int, ...]) -> tuple[int, ...]:
    if goal not in GOALS or horizon < 0 or any(offset not in range(4) for offset in substrates):
        raise ValueError('invalid finite goal/horizon/substrate')
    return tuple((goal >> ((depth + offset) % 4)) & 1
                 for offset in substrates for depth in range(horizon))


def exact_quotient(horizon: int, substrates: tuple[int, ...]) -> dict:
    signatures = {observed_signature(g, horizon, substrates) for g in GOALS}
    count = len(signatures)
    return {'behavior_classes': count, 'minimal_fixed_length_code_bits': ceil(log2(count))}


def scalar_deviation(lipschitz: float, step_defect: float, depth: int) -> float:
    """Sharp cumulative error for scalar F(h)=L*h, G(h)=L*h+delta."""
    if lipschitz < 0 or step_defect < 0 or depth < 0:
        raise ValueError('nonnegative coefficients and depth required')
    err = 0.0
    for _ in range(depth):
        err = lipschitz * err + step_defect
    return err


def report() -> dict:
    horizons = range(5)
    return {
        'status': 'constructed_exact_finite_reference_only',
        'goals': 16,
        'horizons': list(horizons),
        'single_substrate_offset_0': [exact_quotient(h, (0,)) for h in horizons],
        'joint_substrates_offsets_0_2': [exact_quotient(h, (0, 2)) for h in horizons],
        'complete_goal_erasure_classes': [1 for _ in horizons],
        'scalar_depth_defects': {
            f'L={l},delta=1,depth=5': scalar_deviation(float(l), 1.0, 5)
            for l in (0, 1, 2)
        },
        'note': 'Behavior classes depend on observer coverage. The scalar recurrence is a sharp algebraic bound, not a neural certificate.'
    }


if __name__ == '__main__':
    print(json.dumps(report(), indent=2, sort_keys=True))
