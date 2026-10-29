"""Bounded synthetic curriculum experiment with sealed evaluation partitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import platform
from pathlib import Path
import random
import time

from .design import DesignBudget, OfflineDesigner, make_request, validate_response
from .env import InterfaceSpec, RuleSpec
from .io import canonical, digest, source_hashes, write_json
from .learner import Student, collect_demonstrations, evaluate
from .replay import DRAMReplay


def interfaces() -> dict[str, InterfaceSpec]:
    return {
        "camera": InterfaceSpec(name="camera", observation="camera", command_ticks=3, switch_ticks=4),
        "assisted": InterfaceSpec(name="assisted", observation="camera", command_ticks=1, switch_ticks=1),
        "api": InterfaceSpec(name="api", observation="global", command_ticks=1, switch_ticks=0),
    }


def training_variants() -> dict[str, RuleSpec]:
    return {
        "target_reversal": RuleSpec(target_xor=1),
        "short_priority": RuleSpec(deadline_ticks=8, reward_weights=(1, 4, 2, 3)),
        "combined": RuleSpec(deadline_ticks=10, target_xor=1, reward_weights=(3, 1, 4, 2)),
        "long_window": RuleSpec(deadline_ticks=24),
    }


def holdout_variants() -> dict[str, RuleSpec]:
    # Combinations are absent from training, although each primitive is declared.
    return {
        "heldout_reversal_priority": RuleSpec(deadline_ticks=17, target_xor=1,
                                             reward_weights=(4, 1, 2, 3)),
        "heldout_original_priority": RuleSpec(deadline_ticks=19,
                                             reward_weights=(1, 3, 4, 2)),
    }


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    interface: str = "camera"
    width: int = 8
    pretrain_updates: int = 200
    iterations: int = 4
    updates_per_iteration: int = 120
    batch_size: int = 32
    train_episodes: int = 16
    probe_episodes: int = 8
    test_episodes: int = 32
    replay_capacity: int = 512
    max_replay_bytes: int = 1048576
    max_seconds: int = 120

    def __post_init__(self):
        limits = {
            "seed": (0, 2**32 - 1), "width": (1, 128),
            "pretrain_updates": (0, 5000), "iterations": (1, 100),
            "updates_per_iteration": (1, 5000), "batch_size": (1, 1024),
            "train_episodes": (4, 256), "probe_episodes": (2, 256),
            "test_episodes": (2, 1024), "replay_capacity": (16, 1_000_000),
            "max_replay_bytes": (1024, 64 * 1024 * 1024),
            "max_seconds": (1, 3600),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}: integer in {low}..{high} required")
        if self.interface not in interfaces():
            raise ValueError("unknown interface profile")
        if self.replay_capacity < 2 * self.train_episodes:
            raise ValueError("replay must retain at least two demonstration blocks")


def split_seeds(config: RunConfig) -> dict[str, list[int]]:
    groups = {"pretrain": config.train_episodes,
              "probe": config.probe_episodes,
              "heldout": config.test_episodes,
              "retention": config.test_episodes}
    groups.update({f"train_{i}": config.train_episodes for i in range(config.iterations)})
    result = {
        name: [int.from_bytes(hashlib.sha256(f"pluto-gym:{config.seed}:{name}:{j}".encode()).digest()[:4],
                              "little") for j in range(count)]
        for name, count in groups.items()
    }
    flattened = [seed for seeds in result.values() for seed in seeds]
    if len(set(flattened)) != len(flattened):
        raise ValueError("seed split collision; choose another run seed")
    return result


def _supervision(rules, interface, seeds):
    rows = collect_demonstrations(rules, interface, seeds)
    # Keep one attack per episode, preserving paired seed sampling across rules.
    # Collection cost is separate because deadlines change the actions generated.
    count = len(seeds)
    first_by_seed = {}
    for row in rows:
        first_by_seed.setdefault(row["episode_seed"], row)
    if set(first_by_seed) != set(seeds):
        raise ValueError("interface cannot collect the declared demonstration block")
    retained = [first_by_seed[seed] for seed in seeds]
    return retained, {"episodes": len(seeds), "oracle_rows_generated": len(rows),
                      "rows_retained": count, "retention_rule": "first attack per episode",
                      "unique_retained_episode_seeds": len(first_by_seed)}


def _probes(student, candidates, interface, seeds):
    scores = {}
    for name, rules in candidates.items():
        measured = evaluate(student, rules, interface, seeds)
        scores[name] = {"mean_reward": measured["mean_reward"], "episodes": len(seeds),
                        "interface_oracle_value": measured["interface_oracle_value"],
                        "mean_interface_regret": measured["mean_interface_regret"]}
    return scores


def _suite(student, tests, interface, seeds):
    return {name: evaluate(student, rules, interface, seeds) for name, rules in tests.items()}


def run_experiment(config: RunConfig, output: Path, *, designer=None) -> dict:
    """Train actual small policies, preserving raw receipts and checkpoints.

    This experiment learns a target-choice map. Its public scheduling solver is
    a declared component, and the native Pluto release is not loaded or updated.
    """
    import numpy as np

    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new output directory; previous runs are immutable")
    interface = interfaces()[config.interface]
    base_rules = RuleSpec()
    candidates, tests = training_variants(), holdout_variants()
    train_hashes = {digest(r.to_dict()) for r in candidates.values()} | {digest(base_rules.to_dict())}
    if any(digest(r.to_dict()) in train_hashes for r in tests.values()):
        raise ValueError("held-out rule combination overlaps training")
    seeds = split_seeds(config)
    designer = designer or OfflineDesigner(DesignBudget(max_calls=config.iterations))
    if designer.budget.max_calls < config.iterations:
        raise ValueError("designer call budget is smaller than iteration budget")
    sources = source_hashes()
    designer_configuration = designer.configuration()
    run_id = digest({"config": asdict(config), "sources": sources,
                     "designer_configuration": designer_configuration})[:20]
    manifest = {
        "schema": "pluto-gym.run.v1", "run_id": run_id, "configuration": asdict(config),
        "evidence_scope": "synthetic_multifront_target_imitation",
        "environment": {"interface": interface.to_dict(), "base_rules": base_rules.to_dict()},
        "candidate_rules": {k: v.to_dict() for k, v in candidates.items()},
        "heldout_rules": {k: v.to_dict() for k, v in tests.items()},
        "seeds": seeds, "source_sha256": sources,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "designer_profile": designer.profile,
        "designer_configuration": designer_configuration,
        "budget_scope": "equal retained labels, SGD updates, batch size, replay cap and training probes for adaptive arms",
        "collection_cost_scope": "raw oracle rows and reachable environment transitions can differ with rules",
        "native_pluto_training": False, "native_interface_enforcement": False,
        "gpu_used": False, "external_game_files": False,
        "test_policy": "held-out results enter receipts only; designer receives training probes only",
    }
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    start = time.monotonic()

    def time_check():
        if time.monotonic() - start > config.max_seconds:
            raise TimeoutError("cooperative experiment time budget reached between phases")

    try:
        base_rows, base_collection = _supervision(base_rules, interface, seeds["pretrain"])
        base_replay = DRAMReplay(config.replay_capacity, max_bytes=config.max_replay_bytes)
        base_replay.extend(base_rows)
        base = Student(seed=config.seed, width=config.width)
        pretrain = base.train(base_replay, updates=config.pretrain_updates,
                              batch_size=config.batch_size, seed=config.seed)
        base_state = base.state_dict()
        write_json(output / "base_checkpoint.json", base_state)
        frozen = base.clone()
        sealed_before = _suite(frozen, tests, interface, seeds["heldout"])
        retention_before = evaluate(frozen, base_rules, interface, seeds["retention"])
        result_arms = {"frozen": {"heldout": sealed_before, "retention": retention_before,
                                   "updates": 0, "checkpoint_sha256": digest(base_state),
                                   "ledger": frozen.ledger()}}
        candidates_wire = [{"variant_id": name, "rules": rules.to_dict()}
                           for name, rules in candidates.items()]
        design_traces = output / "designer_traces.jsonl"
        with design_traces.open("x", encoding="utf-8") as trace:
            for arm in ("random", "designer"):
                student = base.clone()
                replay = DRAMReplay(config.replay_capacity, max_bytes=config.max_replay_bytes)
                replay.extend(base_rows)
                rng = random.Random(config.seed)
                timeline = []
                for iteration in range(config.iterations):
                    time_check()
                    summary = _probes(student, candidates, interface, seeds["probe"])
                    request = response = None
                    if arm == "designer":
                        request = make_request(
                            iteration=iteration, backend="synthetic_multifront",
                            candidates=candidates_wire, training_summary=summary,
                            model=designer.model, reasoning=designer.reasoning,
                            max_output_tokens=designer.budget.max_output_tokens, run_id=run_id)
                        response = validate_response(designer.propose(request), request,
                                                     max_bytes=designer.budget.max_response_bytes)
                        choice = response["variant_id"]
                    else:
                        choice = rng.choice(list(candidates))
                    rows, collection = _supervision(candidates[choice], interface, seeds[f"train_{iteration}"])
                    replay.extend(rows)
                    fit = student.train(replay, updates=config.updates_per_iteration,
                                        batch_size=config.batch_size,
                                        seed=config.seed + iteration + 1)
                    post_summary = _probes(student, candidates, interface, seeds["probe"])
                    if request is not None:
                        trace.write(canonical({
                            "schema": "pluto-gym.designer-trace.v1", "request": request,
                            "response": response, "training_feedback": post_summary,
                            "fitness_scope": "training_probe_only", "evaluation_metrics_included": False,
                            "eligible_for_confirmed_game_design_benefit_label": False,
                        }).decode() + "\n")
                        trace.flush()
                    row = {"iteration": iteration, "variant_id": choice,
                           "cumulative_updates": (iteration + 1) * config.updates_per_iteration,
                           "collection": collection, "fit": fit,
                           "training_probe_before": summary, "training_probe_after": post_summary,
                           "heldout": _suite(student, tests, interface, seeds["heldout"]),
                           "retention": evaluate(student, base_rules, interface, seeds["retention"]),
                           "replay": replay.ledger()}
                    timeline.append(row)
                    write_json(output / f"{arm}_iteration_{iteration}.json", row)
                state = student.state_dict()
                write_json(output / f"{arm}_checkpoint.json", state)
                result_arms[arm] = {
                    "heldout": timeline[-1]["heldout"], "retention": timeline[-1]["retention"],
                    "updates": config.iterations * config.updates_per_iteration,
                    "timeline": timeline, "checkpoint_sha256": digest(state),
                    "ledger": student.ledger(), "replay": replay.ledger(),
                    "interface_sweep": {name: evaluate(student, base_rules, spec, seeds["retention"])
                                        for name, spec in interfaces().items()},
                }
        time_check()
        receipt = {
            "schema": "pluto-gym.receipt.v1", "run_id": run_id, "status": "completed",
            "manifest_sha256": digest(manifest), "source_sha256": sources,
            "evidence_scope": manifest["evidence_scope"],
            "pretrain": {"fit": pretrain, "collection": base_collection, "replay": base_replay.ledger()},
            "arms": result_arms, "designer": designer.ledger(),
            "wall_seconds": time.monotonic() - start,
            "boundaries": {
                "released_pluto_weights_updated": False,
                "native_starcraft_match_executed": False,
                "learned_scheduler": False,
                "dram_claim": "Host replay payload and optimizer/weight storage are accounted separately.",
                "learning_claim": "Small synthetic target-choice policy trained by imitation.",
                "meta_adaptation_claim": "Per-run fixed-budget adaptation observations; no statistical superiority gate.",
            },
        }
        write_json(output / "receipt.json", receipt)
        return receipt
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "run_id": run_id,
                   "error_type": type(exc).__name__, "designer": designer.ledger(),
                   "elapsed_seconds": time.monotonic() - start})
        raise

