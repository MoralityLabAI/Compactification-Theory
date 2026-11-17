"""SHAFT-COMP-14: exact two-dimensional directional-observability reference.

A monitor can have unchanged rank while responding differently to different
state contrasts. All costs use nonnegative integer arithmetic. The neural
matrix/Gramian results are independently measured in the trained benchmark.
"""
from itertools import product


def dot(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("dimension mismatch")
    return sum(x*y for x,y in zip(xs,ys))


def weighted_energy(rows, contrast, weights):
    if len(rows) != len(weights):
        raise ValueError("one weight required per observed row")
    if any(w < 0 for w in weights):
        raise ValueError("nonnegative weights required")
    return sum((w*dot(row,contrast))**2 for row,w in zip(rows,weights))


def cumulative_energies(rows, contrast, weights):
    out=[]
    for n in range(len(rows)+1):
        out.append(weighted_energy(rows[:n],contrast,weights[:n]))
    return out


def reference():
    fixed_rank_one_monitor=[(1,0)]
    aligned=(1,0)
    unobserved=(0,1)
    aligned_energy=weighted_energy(fixed_rank_one_monitor,aligned,[1])
    blind_energy=weighted_energy(fixed_rank_one_monitor,unobserved,[1])
    assert (aligned_energy,blind_energy)==(1,0)
    # A second, independent observation makes the previously blind axis visible.
    expanded=[(1,0),(0,1)]
    later=cumulative_energies(expanded,unobserved,[1,1])
    assert later==[0,0,1]
    # Any nonnegative, bounded weighting preserves PSD and cumulative monotonicity.
    for contrast in product(range(3),repeat=2):
        for weights in product(range(3),repeat=2):
            seq=cumulative_energies(expanded,contrast,weights)
            assert seq==sorted(seq)
            assert seq[-1]>=0
    return {
        "evidence":"exhaustive_nonnegative_integer_2d_cost_reference",
        "monitor_row":[1,0],
        "fixed_monitor_rank":1,
        "aligned_delta":[1,0],
        "hidden_delta":[0,1],
        "aligned_directional_energy":aligned_energy,
        "hidden_directional_energy":blind_energy,
        "hidden_delta_cumulative_energy_two_rows":later,
        "exact_weighted_monotonicity_cases":9*9,
        "insight":"same observation rank allows distinct contrast-specific energies",
    }


if __name__=="__main__":
    import json
    print(json.dumps(reference(),sort_keys=True,indent=2))
