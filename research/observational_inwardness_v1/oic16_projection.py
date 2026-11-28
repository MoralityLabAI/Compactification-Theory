"""OIC-16 post-hoc projection of learned cyclic kernels onto finite shifts.

The projection is an explicit *extra structural prior*: each family moves by
one deterministic modular offset, and emissions fire at one discrete phase.
It is evaluated after the continuous-kernel experiment. The evaluator sees only
public probe and raw state histories; no family/phase labels enter inference.
"""
from __future__ import annotations

import math
from typing import Mapping

SHIFTS = (-2, -1, 0, 1, 2)


def projected_steps(weights: Mapping[str, object]) -> tuple[int, int]:
    logits = weights['kernel_logits']
    if len(logits) != 2 or any(len(row) != len(SHIFTS) for row in logits):
        raise ValueError('expected two five-offset kernels')
    return tuple(SHIFTS[max(range(len(row)), key=lambda k: row[k])] for row in logits)


def projected_marker(weights: Mapping[str, object], n: int) -> int:
    if n < 2:
        raise ValueError('n >= 2 required')
    a, b, c = weights['emission_coeff']
    logits = [a+b*math.cos(2*math.pi*z/n)+c*math.cos(4*math.pi*z/n) for z in range(n)]
    return max(range(n), key=lambda z: logits[z])


def trace(n: int, family: int, initial_phase: int, word: tuple[int, ...],
          shifts: tuple[int, int], marker: int, x0: int = 0) -> tuple[int, ...]:
    if n < 2 or family not in (0, 1) or not 0 <= initial_phase < n or x0 not in (0, 1):
        raise ValueError('invalid state or model family')
    z = initial_phase
    x = x0
    out = [x]
    for u in word:
        if u not in (0, 1):
            raise ValueError('nonbinary action')
        x ^= 1 ^ int(bool(u) and z == marker)
        z = (z + shifts[family]) % n
        out.append(x)
    return tuple(out)


def projected_sets(n: int, word: tuple[int, ...], weights: Mapping[str, object]):
    shifts = projected_steps(weights)
    marker = projected_marker(weights, n)
    return tuple(
        frozenset(trace(n, family, z, word, shifts, marker) for z in range(n))
        for family in (0, 1)
    )


def oracle_sets(n: int, word: tuple[int, ...]):
    return tuple(
        frozenset(trace(n, family, z, word, (0, 1), 0) for z in range(n))
        for family in (0, 1)
    )


def distance(received: tuple[int, ...], codeword: tuple[int, ...]) -> int:
    if len(received) != len(codeword):
        raise ValueError('unequal lengths')
    if any(v not in (0, 1, -1) for v in received):
        raise ValueError('readout must be 0, 1 or -1 for erasure')
    return sum(a != -1 and a != b for a, b in zip(received, codeword))


def decode(n: int, word: tuple[int, ...], weights: Mapping[str, object], received: tuple[int, ...], codebooks=None):
    if codebooks is None:
        codebooks = projected_sets(n, word, weights)
    distances = tuple(min(distance(received, candidate) for candidate in fam) for fam in codebooks)
    if distances[0] == distances[1]:
        p = 0.5
    else:
        p = float(distances[1] < distances[0])
    return p, distances


def panel(n: int, word: tuple[int, ...], weights: Mapping[str, object],
          flip_position: int | None = None, erase_position: int | None = None):
    results=[]
    codebooks=projected_sets(n,word,weights)
    for family in (0, 1):
        for z in range(n):
            clean = trace(n, family, z, word, (0, 1), 0)
            received = list(clean)
            if flip_position is not None:
                received[flip_position] ^= 1
            if erase_position is not None:
                received[erase_position] = -1
            probability, (d0, d1) = decode(n, word, weights, tuple(received), codebooks=codebooks)
            results.append({'family': family, 'phase': z, 'probability': probability,
                            'distance_to_fixed': d0, 'distance_to_rotating': d1})
    return {
        'accuracy':sum(int((entry['probability'] > 0.5) == (entry['family'] == 1))
                       for entry in results) / len(results),
        'brier':sum((entry['probability']-entry['family'])**2 for entry in results)/len(results),
        'abstention_fraction':sum(entry['probability'] == 0.5 for entry in results)/len(results),
    }


def evaluate(weights: Mapping[str, object], ns=(4, 5, 6, 7, 8, 9, 10, 12)) -> dict:
    records={}
    for n in ns:
        shifts=projected_steps(weights)
        marker=projected_marker(weights,n)
        word=(1,)*n
        nstep_clean=panel(n,word,weights)
        passive=panel(n,(0,)*(2*n),weights)
        latch=(1,)*n+(0,)*n
        oracle_match=projected_sets(n,latch,weights)==oracle_sets(n,latch)
        v={'n':n,'projected_shifts':shifts,'emission_marker':marker,
           'codebook_equality_on_latch':oracle_match,
           'full_active':nstep_clean,'passive':passive,'latch':panel(n,latch,weights)}
        if n%2==0:
            word=(1,)*n+(0,)*2
            v['raw_fault_horizon']=n+2
            v['single_flip_minimum_accuracy']=min(panel(n,word,weights,flip_position=i)['accuracy']
                                                for i in range(n+3))
            v['single_erase_minimum_accuracy']=min(panel(n,word,weights,erase_position=i)['accuracy']
                                                 for i in range(n+3))
        records[str(n)] = v
    return records
