"""FHC-01B: measured finite future-metric compression of a frozen RC-01 looper.

PCA is fitted on 48 calibration states; random projection has a fixed seed;
the exact GF(2) quotient is a certified symbolic encoder/decoder. The 16
held-out states and all their pair distances are used only for evaluation.
All 64 source states are enumerated by the independent symbolic oracle.

This uses NumPy. It does not train/evaluate a language model checkpoint or
verify the infinite-dimensional #098 witness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.finite_horizon_gap_v1.horizon_geometry import (
    HORIZONS, WIDTH, full_certificate, observation_masks, symbolic_quotient,
    trace_from_state,
)

SCHEMA_VERSION = "fhc01b.future_metric.v1"
SEED = 940198
SPLIT = 48
D_THRESHOLDS = (1.5, 2.0)
WIDTHS = (2, 4, 6, 8, 16, 32)
DEFAULT_HORIZONS = HORIZONS


def features(horizon: int) -> np.ndarray:
    masks = observation_masks(horizon)
    return np.array([
        trace_from_state(x, masks) for x in range(1 << WIDTH)
    ], dtype=np.float64)


def split_indices(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(1 << WIDTH)
    train, test = np.sort(perm[:SPLIT]), np.sort(perm[SPLIT:])
    if set(train) & set(test):
        raise AssertionError("calibration/heldout state leak")
    return train, test


def _positive_pairs(original: np.ndarray, encoded: np.ndarray):
    if (original.ndim != 2 or encoded.ndim != 2 or
            original.shape[0] != encoded.shape[0]):
        raise ValueError("mismatched 2D source and code arrays")
    if not (np.isfinite(original).all() and np.isfinite(encoded).all()):
        raise ValueError("nonfinite code or future features")
    ii, jj = np.triu_indices(len(original), k=1)
    d_source = np.linalg.norm(original[ii] - original[jj], axis=1)
    d_code = np.linalg.norm(encoded[ii] - encoded[jj], axis=1)
    valid = d_source > 1e-12
    if np.any((~valid) & (d_code > 1e-10)):
        raise AssertionError("same semantic trace maps to distinct encoder codes")
    if not np.any(valid):
        raise ValueError("no distinct semantic pairs")
    return ii[valid], jj[valid], d_source[valid], d_code[valid]


def calibration_scale(source, code) -> float:
    _, _, a, b = _positive_pairs(source, code)
    values = b / a
    nonzero = values[values > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    return float(1 / np.median(nonzero))


def distortion_metrics(source: np.ndarray, code: np.ndarray,
                       scale: float, state_ids: np.ndarray) -> dict:
    """Heldout pair metric with a calibration-only fixed code scaling."""
    if scale <= 0 or len(state_ids) != len(source):
        raise ValueError("invalid scale or ids")
    ii, jj, a, b = _positive_pairs(source, code)
    ratios = scale * b / a
    min_ratio = float(ratios.min())
    max_ratio = float(ratios.max())
    optimized_distortion = (None if min_ratio <= 1e-12
                            else float(max_ratio / min_ratio))
    residual = np.abs(np.log(np.maximum(ratios, 1e-300)))
    worst = int(np.argmax(residual))
    count = {}
    witnesses = {}
    for D in D_THRESHOLDS:
        cutoff = math.sqrt(D)
        fails = np.flatnonzero((ratios < 1/cutoff - 1e-12) |
                               (ratios > cutoff + 1e-12))
        key = str(D)
        count[key] = int(len(fails))
        if len(fails):
            ix = int(fails[np.argmax(residual[fails])])
            witnesses[key] = {
                "state_a": int(state_ids[int(ii[ix])]),
                "state_b": int(state_ids[int(jj[ix])]),
                "ratio_scaled": float(ratios[ix]),
                "source_distance": float(a[ix]),
                "code_distance": float(b[ix]),
            }
    return {
        "pairs_distinct": len(ratios),
        "min_ratio_scaled": min_ratio,
        "max_ratio_scaled": max_ratio,
        "optimal_rescaling_distortion_heldout": optimized_distortion,
        "max_log_distortion_magnitude": float(residual[worst]),
        "worst_pair": [int(state_ids[int(ii[worst])]),
                       int(state_ids[int(jj[worst])])],
        "violation_counts_calibration_scale": count,
        "violation_witnesses": witnesses,
    }


def exact_binary_code(horizon: int) -> np.ndarray:
    quotient = symbolic_quotient(horizon)
    basis = quotient["basis_masks"]
    return np.array([
        [(mask & state).bit_count() % 2 for mask in basis]
        for state in range(1 << WIDTH)
    ], dtype=np.float64)


def _fit_pca(source_train: np.ndarray, d: int):
    center = source_train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(source_train - center, full_matrices=False)
    return center, vt[:d].T


def _random_embedding(T: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, d)) / math.sqrt(d)


def run_one(horizon: int, train: np.ndarray, test: np.ndarray,
            widths: tuple[int, ...] = WIDTHS) -> dict:
    x = features(horizon)
    exact = symbolic_quotient(horizon)
    rows = []
    methods = [("exact_binary_quotient", exact_binary_code(horizon), {
        "binary_live_bits": exact["rank_gf2"],
        "basis_description_bits_upper": WIDTH * exact["rank_gf2"],
        "decoder_description_bits_upper": horizon * exact["rank_gf2"],
        "exact_future_bit_reconstruction_all_64_states": True,
        "trained": False,
    })]
    methods.append(("full_trace_identity", x.copy(), {
        "binary_live_bits": horizon,
        "basis_description_bits_upper": 0,
        "decoder_description_bits_upper": 0,
        "exact_future_bit_reconstruction_all_64_states": True,
        "trained": False,
    }))
    center_all = x[train].mean(axis=0, keepdims=True)
    for d in widths:
        if d > horizon:
            continue
        if d <= min(len(train), horizon):
            center, matrix = _fit_pca(x[train], d)
            methods.append((f"pca_{d}", (x-center) @ matrix, {
                "coordinates": d, "trained": True,
                "fit_states": len(train), "oracle_future_steps_per_state": horizon,
                "matrix_parameters": horizon*d,
                "mean_parameters": horizon,
            }))
        mat = _random_embedding(horizon, d, seed=SEED + 100* horizon + d)
        methods.append((f"gaussian_{d}", (x-center_all) @ mat, {
            "coordinates": d, "trained": False,
            "matrix_parameters": horizon*d,
            "oracle_future_steps_per_state": horizon,
        }))
    for name, code, ledger in methods:
        scale = calibration_scale(x[train], code[train])
        metric = distortion_metrics(x[test], code[test], scale, test)
        rows.append({"method": name, "encoded_dimension": int(code.shape[1]),
                     "calibration_scale": scale, "resource_ledger": ledger,
                     "heldout": metric})
    return {
        "horizon": horizon,
        "source_dimension": horizon,
        "exact_binary_memory_minimum": exact["rank_gf2"],
        "exact_real_isometric_dimension_minimum": horizon,
        "train_size": len(train),
        "heldout_size": len(test),
        "methods": rows,
    }


def run(seed: int = SEED, horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        widths: tuple[int, ...] = WIDTHS) -> dict:
    if not horizons or any(T < 1 or T > 63 for T in horizons):
        raise ValueError("horizon must be in 1..63")
    if not widths or any(d < 1 for d in widths):
        raise ValueError("widths must be positive")
    train, test = split_indices(seed)
    records = [run_one(T, train, test, widths) for T in horizons]
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    exact_digest = hashlib.sha256(
        (Path(__file__).parent / "horizon_geometry.py").read_bytes()
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_SYNTHETIC_RECURRENT_PILOT",
        "model": "RC-01 GF2 primitive linear feedback shift register, width 6",
        "source_python_sha256": source_digest,
        "exact_reference_sha256": exact_digest,
        "numpy_version": np.__version__,
        "seed": seed,
        "calibration_state_ids": list(map(int, train)),
        "heldout_state_ids": list(map(int, test)),
        "horizons": list(horizons),
        "widths": list(widths),
        "scale_fit": "median positive calibration code/source pair ratio",
        "metric": "Euclidean distance of finite future output bit traces",
        "random_projector": "Gaussian standard_normal/sqrt(d), seed=940198+100*T+d",
        "finite_exact_certificate": full_certificate(),
        "records": records,
        "checkpoint_evaluated": False,
        "proof_of_infinite_doubling_obstruction": False,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    p.add_argument("--widths", type=int, nargs="+", default=list(WIDTHS))
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    result = run(args.seed, tuple(args.horizons), tuple(args.widths))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                         allow_nan=False) + "\n")
    brief = [
        {"T": row["horizon"],
         "binary_bits": row["exact_binary_memory_minimum"],
         "real_isometry_dimension": row["exact_real_isometric_dimension_minimum"],
         "quotient_test_optimal_D": next(x for x in row["methods"]
                                         if x["method"] == "exact_binary_quotient")
                                        ["heldout"]["optimal_rescaling_distortion_heldout"]}
        for row in result["records"]
    ]
    print(json.dumps({"status": result["status"], "sha256": result["source_python_sha256"],
                      "cases": brief}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
