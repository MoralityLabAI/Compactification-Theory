"""CLI: python -m research.pluto_gym --help (from the repository root)."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from .io import load_json, source_hashes, write_json


def parser():
    root = argparse.ArgumentParser(description="Pluto-Gym: bounded interface and adaptation experiments")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="train the synthetic CPU student and compare curricula")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--interface", choices=("camera", "assisted", "api"), default="camera")
    run.add_argument("--width", type=int, default=8)
    run.add_argument("--iterations", type=int, default=4)
    run.add_argument("--pretrain-updates", type=int, default=200)
    run.add_argument("--updates", type=int, default=120)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--train-episodes", type=int, default=16)
    run.add_argument("--probe-episodes", type=int, default=8)
    run.add_argument("--test-episodes", type=int, default=32)
    run.add_argument("--replay-capacity", type=int, default=512)
    run.add_argument("--max-replay-bytes", type=int, default=1048576)
    run.add_argument("--max-seconds", type=int, default=120)
    run.add_argument("--designer", choices=("offline", "webhook"), default="offline")
    run.add_argument("--designer-max-output-tokens", type=int, default=800)
    doctor = commands.add_parser("doctor", help="inspect local capabilities and pinned integration")
    doctor.add_argument("--tensorcraft", type=Path)
    calibration = commands.add_parser("calibrate", help="calculate exact interface/memory fixture ceilings")
    calibration.add_argument("--output", type=Path)
    surrogate = commands.add_parser("surrogate", help="run TensorCraft's pinned stochastic combat surrogate")
    surrogate.add_argument("--tensorcraft", type=Path, required=True)
    surrogate.add_argument("--scenario", default="zvz_equal_spend_clustered")
    surrogate.add_argument("--variant", default="control")
    surrogate.add_argument("--trials", type=int, default=16)
    surrogate.add_argument("--seed", type=int, default=0)
    surrogate.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("native-prepare", help="delegate hashed A/B planning to TensorCraft")
    prepare.add_argument("--tensorcraft", type=Path, required=True)
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--runtime", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    native = commands.add_parser("native-run", help="execute an explicitly prepared TensorCraft farm")
    native.add_argument("--tensorcraft", type=Path, required=True)
    native.add_argument("--workspace", type=Path, required=True)
    native.add_argument("--allow-native", action="store_true")
    trace = commands.add_parser("trace", help="analyze a complete TensorCraft trace; optional replay binding")
    trace.add_argument("--tensorcraft", type=Path, required=True)
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--output", type=Path, required=True)
    trace.add_argument("--manifest", type=Path)
    trace.add_argument("--replay-report", type=Path)
    trace.add_argument("--map", type=Path)
    trace.add_argument("--spawn-tiles", type=Path)
    return root


def _print(value):
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        result = {"python": sys.version.split()[0],
                  "numpy_available": importlib.util.find_spec("numpy") is not None,
                  "gymnasium_available": importlib.util.find_spec("gymnasium") is not None,
                  "native_pluto_weights_trainable": False,
                  "native_action_bridge_available": False,
                  "webhook_requires": ["PLUTO_GYM_DESIGN_URL", "PLUTO_GYM_DESIGN_MODEL"],
                  "source_sha256": source_hashes()}
        lock = Path(__file__).with_name("dependencies.lock.json")
        if lock.is_file():
            result["dependencies"] = load_json(lock)
        if args.tensorcraft is not None:
            from .tensorcraft import TensorCraftAdapter
            result["tensorcraft"] = TensorCraftAdapter(args.tensorcraft).inspect()
        _print(result)
        return 0
    if args.command == "calibrate":
        from .env import MultiFrontEnv, RuleSpec
        from .math_ref import deadline_capacity, query_memory_lower_bound, reset_required_bits
        from .run import interfaces
        result = {
            "evidence": "exact_synthetic_fixture_calibration",
            "interfaces": {name: {"specification": interface.to_dict(),
                                    "maximum_completed": deadline_capacity(4, interface.command_ticks,
                                                                           interface.switch_ticks, 12),
                                    "oracle_reward": _initial_oracle(MultiFrontEnv(RuleSpec(), interface))}
                           for name, interface in interfaces().items()},
            "reset_memory": [{"addressable": n, "required_bits": reset_required_bits(8, list(range(n)))}
                             for n in range(9)],
            "query_64_bits_at_5pct_error_lower_bound": query_memory_lower_bound(64, 0.05),
        }
        if args.output:
            write_json(args.output, result)
        _print(result)
        return 0
    if args.command == "run":
        from .design import DesignBudget, OfflineDesigner, WebhookDesigner
        from .run import RunConfig, run_experiment
        config = RunConfig(seed=args.seed, interface=args.interface, width=args.width,
                           pretrain_updates=args.pretrain_updates, iterations=args.iterations,
                           updates_per_iteration=args.updates, batch_size=args.batch_size,
                           train_episodes=args.train_episodes, probe_episodes=args.probe_episodes,
                           test_episodes=args.test_episodes, replay_capacity=args.replay_capacity,
                           max_replay_bytes=args.max_replay_bytes, max_seconds=args.max_seconds)
        budget = DesignBudget(max_calls=config.iterations,
                              max_output_tokens=args.designer_max_output_tokens)
        designer = (OfflineDesigner(budget) if args.designer == "offline"
                    else WebhookDesigner.from_env(budget=budget))
        receipt = run_experiment(config, args.output, designer=designer)
        _print({"status": receipt["status"], "run_id": receipt["run_id"],
                "evidence_scope": receipt["evidence_scope"],
                "receipt": str(args.output / "receipt.json"),
                "designer": receipt["designer"],
                "final": {arm: {"heldout": {name: {"reward": m["mean_reward"],
                                                       "oracle": m["interface_oracle_value"],
                                                       "regret": m["mean_interface_regret"]}
                                             for name, m in data["heldout"].items()},
                                 "original_rule_reward": data["retention"]["mean_reward"]}
                          for arm, data in receipt["arms"].items()}})
        return 0
    from .tensorcraft import TensorCraftAdapter
    adapter = TensorCraftAdapter(args.tensorcraft)
    if args.command == "surrogate":
        result = adapter.simulate(args.scenario, variant=args.variant, trials=args.trials, seed=args.seed)
        write_json(args.output, result)
    elif args.command == "native-prepare":
        result = adapter.prepare(args.spec, args.runtime, args.output)
    elif args.command == "native-run":
        result = adapter.run(args.workspace, allow_native=args.allow_native)
    elif args.command == "trace":
        result = adapter.analyze_trace(args.trace, require_complete=True,
                                       manifest_file=args.manifest,
                                       replay_report_file=args.replay_report,
                                       map_file=args.map, spawn_tiles_file=args.spawn_tiles)
        write_json(args.output, result)
    else:
        raise ValueError("unknown command")
    _print(result)
    return 0


def _initial_oracle(env):
    env.reset(seed=0)
    return env.oracle_value()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        raise SystemExit(f"Pluto-Gym stopped: {exc}") from None

