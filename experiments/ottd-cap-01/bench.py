"""Run offline OTTD-CAP-01 experiments, live seat probes, or ATIF inventory."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from bridge import SeatBridge, UPSTREAM, UPSTREAM_PIN, atif_summary
from model import Greedy, Scout, run_episode, seeded_world
from oracle import exact_bayesian_value, clairvoyant_mean, policy_mean


def rational(x: Fraction) -> dict:
    return {"fraction": str(x), "decimal": float(x)}


def exact_fixture(seed: int, routes: int, horizon: int, cash: int, bits: int) -> dict:
    s, _ = seeded_world(seed, routes, horizon, cash)
    value = exact_bayesian_value(s)
    greedy = policy_mean(s, Greedy)
    scout = policy_mean(s, lambda: Scout(bits))
    upper = clairvoyant_mean(s)
    if not (greedy <= value <= upper and scout <= value):
        raise AssertionError("exact finite oracle bound failed")
    return {"mode": "finite_exact_synthetic", "scenario": asdict(s),
            "bayesian_optimum": rational(value), "clairvoyant_upper": rational(upper),
            "greedy_expected": rational(greedy), "scout_expected": rational(scout),
            "greedy_regret": rational(value - greedy),
            "scout_regret": rational(value - scout), "scout_memory_budget_bits": bits}


def paired_sweep(seeds: list[int], routes: int, horizon: int, cash: int,
                 budgets: list[int], include_traces: bool = False) -> dict:
    if not seeds or not budgets or min(budgets) < 0:
        raise ValueError("seeds and budgets required; budgets nonnegative")
    out = []
    for seed in seeds:
        s, mask = seeded_world(seed, routes, horizon, cash)
        arms = [("greedy", None, Greedy())] + [
            (f"scout_{b}_bits", b, Scout(b)) for b in budgets
        ]
        for name, b, agent in arms:
            receipt = run_episode(s, mask, agent)
            row = {"seed": seed, "policy": name, "memory_budget_bits": b,
                   "peak_memory_bits": receipt["peak_memory_bits"],
                   "quality_mask_evaluator_only": mask,
                   "net_cash": receipt["net_cash"],
                   "operating_profit": receipt["operating_profit"],
                   "terminal_cash": receipt["terminal"]["cash"],
                   "capital_spent": receipt["terminal"]["capital_spent"],
                   "trace_sha256": receipt["trace_sha256"], "steps": receipt["steps"]}
            if include_traces:
                row["receipt"] = receipt
            out.append(row)
    return {"mode": "paired_synthetic_sweep", "seeds": seeds,
            "route_count": routes, "horizon_months": horizon,
            "starting_cash": cash, "records": out,
            "native_sessions": 0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("sweep", "exact", "probe", "atif"), default="sweep")
    p.add_argument("--seeds", default="101,202,303")
    p.add_argument("--routes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--cash", type=int, default=80)
    p.add_argument("--memory-bits", type=int, default=9)
    p.add_argument("--budgets", default="0,3,6,9")
    p.add_argument("--include-traces", action="store_true")
    p.add_argument("--include-map", action="store_true",
                   help="LIVE: issuing map reads may start the real-time clock")
    p.add_argument("--atif", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.mode == "sweep":
        data = paired_sweep([int(v) for v in args.seeds.split(",")],
                            args.routes, args.horizon, args.cash,
                            [int(v) for v in args.budgets.split(",")], args.include_traces)
    elif args.mode == "exact":
        data = exact_fixture(int(args.seeds.split(",")[0]),
                             args.routes, args.horizon, args.cash, args.memory_bits)
    elif args.mode == "probe":
        data = {"mode": "real_seat_probe", "result": SeatBridge().probe(args.include_map)}
    else:
        if args.atif is None:
            p.error("--atif is required for ATIF mode")
        data = {"mode": "external_atif_inventory", "result": atif_summary(args.atif)}
    root = Path(__file__).resolve().parent
    data["provenance"] = {
        "upstream": UPSTREAM, "upstream_pin": UPSTREAM_PIN,
        "local_source_sha256": {name: sha256((root / name).read_bytes()).hexdigest()
                                for name in ("model.py", "oracle.py", "bridge.py", "bench.py")}}
    serialized = json.dumps(data, sort_keys=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
