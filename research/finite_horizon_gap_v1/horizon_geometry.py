"""FHC-01B: exact future-observation rank for an existing RC-01 affine looper.

Uses the RC-01 GF(2) reference directly. A primitive degree-six
linear feedback shift register has 63 distinct nonzero output masks.
The 64 full states are a *finite*, uniformly sampled certificate.

No claim about infinite doubling witnesses or learned neural models.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.recurrent_circuits_v1.reference import (
    Affine, coordinates, row_basis, row_times_matrix,
)

WIDTH = 6
PERIOD = (1 << WIDTH) - 1
HORIZONS = (1, 2, 4, 6, 8, 16, 32, 63)


def primitive_shift6() -> Affine:
    """Companion recurrence h_i'=h_{i+1}, h_5'=h_0 xor h_1."""
    return Affine(tuple(1 << (i+1) for i in range(WIDTH-1)) + (3,))


def observation_masks(horizon: int) -> tuple[int, ...]:
    if type(horizon) is not int or not 0 <= horizon <= PERIOD:
        raise ValueError("require horizon in 0..63")
    step = primitive_shift6()
    mask = 1
    masks = []
    for _ in range(horizon):
        masks.append(mask)
        mask = row_times_matrix(mask, step)
    return tuple(masks)


def trace_from_state(state: int, masks: tuple[int, ...]) -> tuple[int, ...]:
    if type(state) is not int or not 0 <= state < (1 << WIDTH):
        raise ValueError("state outside 6-bit register")
    return tuple((mask & state).bit_count() & 1 for mask in masks)


def symbolic_quotient(horizon: int) -> dict:
    """Minimal number of BINARY code bits for exact horizon prediction."""
    masks = observation_masks(horizon)
    basis = row_basis(masks, WIDTH)
    decoder_masks = tuple(coordinates(mask, basis, WIDTH) for mask in masks)
    codes = {
        state: tuple((b & state).bit_count() & 1 for b in basis)
        for state in range(1 << WIDTH)
    }
    for state in range(1 << WIDTH):
        packed = sum(bit << i for i, bit in enumerate(codes[state]))
        reconstructed = tuple((packed & decoder).bit_count() & 1
                              for decoder in decoder_masks)
        assert reconstructed == trace_from_state(state, masks), (
            "RC-01 quotient failed exact future reconstruction"
        )
    return {
        "horizon": horizon,
        "rank_gf2": len(basis),
        "distinct_traces": len(set(trace_from_state(x, masks)
                                   for x in range(1 << WIDTH))),
        "basis_masks": list(basis),
        "output_masks": list(masks),
        "decoder_masks": list(decoder_masks),
    }


def walsh_orthogonality(horizon: int) -> dict:
    """Exact real covariance of full 64-state future traces, no numpy."""
    masks = observation_masks(horizon)
    states = list(range(1 << WIDTH))
    sums = [
        sum((-1 if (x & mask).bit_count() & 1 else 1)
            for x in states)
        for mask in masks
    ]
    if any(value != 0 for value in sums):
        raise AssertionError("nonzero observable is unexpectedly unbalanced")
    gram_diag = True
    for i, a in enumerate(masks):
        for j, b in enumerate(masks):
            correlation = sum(
                (1 if (x & (a ^ b)).bit_count() % 2 == 0 else -1)
                for x in states
            )
            if correlation != (64 if i == j else 0):
                gram_diag = False
    if not gram_diag:
        raise AssertionError("Walsh characters fail exact orthogonality")
    return {
        "horizon": horizon,
        "distinct_masks": len(set(masks)),
        "all_nonzero_masks": all(masks),
        "real_centered_feature_rank": horizon,
        "centered_covariance_diagonal": "1/4",
        "centered_covariance_off_diagonal": "0",
    }


def action_refinement(horizon: int = 1) -> dict:
    """One-shot interface: observer may rotate before reading bit0.

    Rotation by k produces the mask e_k. Adding action access to all
    six rotations makes the single-step predictive quotient 6 bits.
    """
    if horizon != 1:
        raise ValueError("this interface fixture has exactly one readout")
    plain = row_basis((1,), WIDTH)
    allowed = tuple(1 << i for i in range(WIDTH))
    enlarged = row_basis(allowed, WIDTH)
    return {
        "horizon": horizon,
        "plain_observation_bits": len(plain),
        "all_rotations_observation_bits": len(enlarged),
        "action_masks": list(allowed),
        "counterexample_pair": [0, 2],
        "plain_pair_equal": trace_from_state(0, (1,)) ==
                            trace_from_state(2, (1,)),
        "post_rotation_pair_differs": trace_from_state(0, (2,)) !=
                                     trace_from_state(2, (2,)),
    }


def full_certificate() -> dict:
    step = primitive_shift6()
    masks = observation_masks(PERIOD)
    if not (len(set(masks)) == PERIOD and
            set(masks) == set(range(1, 1 << WIDTH)) and
            row_times_matrix(masks[-1], step) == 1):
        raise AssertionError("expected maximal LFSR period 63")
    records = []
    for T in HORIZONS:
        quotient = symbolic_quotient(T)
        real = walsh_orthogonality(T)
        assert quotient["rank_gf2"] == min(T, WIDTH)
        assert quotient["distinct_traces"] == 1 << quotient["rank_gf2"]
        records.append({
            "horizon": T,
            "minimum_binary_semantic_memory_bits":
                quotient["rank_gf2"],
            "minimum_exact_euclidean_isometry_dimension":
                real["real_centered_feature_rank"],
            "distinct_semantic_traces": quotient["distinct_traces"],
        })
    return {
        "status": "EXACT_FINITE_CONTROL",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rc01_dependency": "research.recurrent_circuits_v1.reference",
        "state_width": WIDTH,
        "full_state_count": 1 << WIDTH,
        "lfsr_period": PERIOD,
        "results": records,
        "action_refinement": action_refinement(),
        "source_family_098_witness_constructed": False,
        "trained_checkpoint_evaluated": False,
    }


if __name__ == "__main__":
    print(json.dumps(full_certificate(), indent=2))
