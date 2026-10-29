"""Trainable target policy and transparent schedule controller.

Imitation uses synthetic labels. The NumPy MLP sees the visible binary cue and
public matchup rule. An exact public schedule solver chooses which front to
visit. Both belong in the controller's accounting. This module does not train
released Pluto weights and does not implement PPO or self-play.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import numbers
import random
import sys
import time
from typing import Iterable, Mapping

from .replay import DRAMReplay, _index


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("the optional compact student requires NumPy") from exc
    return np


def _binary(value, name: str) -> int:
    return _index(value, name, 0, 1)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _seed_list(seeds: Iterable[int]) -> list[int]:
    result = [_index(x, "episode seed", 0, 2**63 - 1) for x in seeds]
    if not result or len(result) > 100_000 or len(set(result)) != len(result):
        raise ValueError("provide 1..100000 distinct episode seeds")
    return result


def _process_peak_rss_bytes() -> int | None:
    """Whole-process lifetime peak, including allocations outside this trainer."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError):
        return None
    return int(peak if sys.platform == "darwin" else peak * 1024)


class Student:
    """Two-input tanh MLP, binary sigmoid output and CPU Adam updates.

    This baseline has no recurrent state. Each train call resets optimizer
    moments. Checkpoints contain the parameters and cumulative update count.
    Bitwise portability across NumPy/BLAS environments is not asserted.
    """

    def __init__(self, seed: int = 0, width: int = 8):
        np = _numpy()
        self.seed = _index(seed, "seed", 0, 2**63 - 1)
        self.width = _index(width, "width", 1, 1024)
        rng = np.random.default_rng(self.seed)
        self._weights = {
            "W1": rng.normal(0.0, 0.7, (2, self.width)),
            "b1": np.zeros(self.width, dtype=np.float64),
            "W2": rng.normal(0.0, 0.7 / math.sqrt(self.width), self.width),
            "b2": np.zeros(1, dtype=np.float64),
        }
        self.training_updates = 0

    def _forward(self, features):
        np = _numpy()
        x = 2.0 * np.asarray(features, dtype=np.float64) - 1.0
        hidden = np.tanh(x @ self._weights["W1"] + self._weights["b1"])
        logits = hidden @ self._weights["W2"] + self._weights["b2"][0]
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        return x, hidden, logits, probability

    def predict_probability(self, cue: int, target_xor: int) -> float:
        cue = _binary(cue, "cue")
        target_xor = _binary(target_xor, "target_xor")
        return float(self._forward([[cue, target_xor]])[3][0])

    def act(self, observation: Mapping) -> dict:
        from .env import optimal_front_from_observation
        front = optimal_front_from_observation(observation)
        if front is None:
            remaining = _index(observation["remaining_ticks"], "remaining_ticks", 1, 2**31 - 1)
            return {"kind": "wait", "ticks": remaining}
        if observation["interface"]["observation"] == "camera" and front != observation["current_front"]:
            return {"kind": "switch", "front": front}
        cue = observation["visible_cues"][front]
        if cue is None:
            raise ValueError("selected attack has no interface-visible cue")
        probability = self.predict_probability(cue, observation["rules"]["target_xor"])
        return {"kind": "attack", "front": front, "target": int(probability >= 0.5)}

    def state_dict(self) -> dict:
        return {"schema_version": 1, "model": "two_input_tanh_target_mlp",
                "seed": self.seed, "width": self.width, "dtype": "float64",
                "training_updates": self.training_updates,
                "weights": {k: v.tolist() for k, v in self._weights.items()}}

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "Student":
        np = _numpy()
        expected = {"schema_version", "model", "seed", "width", "dtype", "training_updates", "weights"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("invalid student checkpoint fields")
        if type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise ValueError("unsupported student checkpoint schema")
        if state["model"] != "two_input_tanh_target_mlp" or state["dtype"] != "float64":
            raise ValueError("unsupported student checkpoint model/dtype")
        result = cls(state["seed"], state["width"])
        updates = _index(state["training_updates"], "training_updates", 0, 2**63 - 1)
        if not isinstance(state["weights"], Mapping) or set(state["weights"]) != set(result._weights):
            raise ValueError("invalid student checkpoint weights")
        checked = {}
        for name, template in result._weights.items():
            raw = state["weights"][name]
            def validate_node(node):
                if isinstance(node, list):
                    for item in node:
                        validate_node(item)
                elif isinstance(node, bool) or not isinstance(node, numbers.Real) or not math.isfinite(float(node)):
                    raise ValueError("checkpoint weights must be finite numbers")
            validate_node(raw)
            try:
                weight = np.asarray(raw, dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid checkpoint tensor") from exc
            if weight.shape != template.shape or not np.isfinite(weight).all():
                raise ValueError("checkpoint tensor shape/value mismatch")
            checked[name] = weight.copy()
        result._weights = checked
        result.training_updates = updates
        return result

    def clone(self) -> "Student":
        return self.from_state_dict(self.state_dict())

    def checkpoint_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.state_dict())).hexdigest()

    def ledger(self) -> dict:
        from .env import optimal_front_from_observation
        scheduler = inspect.getsource(optimal_front_from_observation).encode("utf-8")
        params = sum(int(value.size) for value in self._weights.values())
        payload = sum(int(value.nbytes) for value in self._weights.values())
        return {
            "controller": "learned_target_mlp_plus_public_exact_schedule",
            "trainable_parameters": params, "weight_precision_bits": 64,
            "parameter_storage_bytes": payload, "parameter_storage_bits": payload * 8,
            "recurrent_state_bits": 0,
            "mutable_observation_state": "supplied_by_environment_each_decision",
            "inference_hidden_array_payload_bytes": self.width * 8,
            "inference_budget_scope": "one MLP forward per attack plus public exact schedule helper; search work unmetered",
            "scheduler_function_source_bytes": len(scheduler),
            "scheduler_function_sha256": hashlib.sha256(scheduler).hexdigest(),
            "scheduler_source_is_total_program_bound": False,
            "training_updates_total": self.training_updates,
            "numpy_version": _numpy().__version__,
        }

    def _corpus_loss(self, replay: DRAMReplay) -> float:
        np = _numpy()
        total = 0.0
        count = 0
        batch = []
        for row in replay.iter_rows():
            batch.append(row)
            if len(batch) == 256:
                logits = self._forward([x["features"] for x in batch])[2]
                labels = np.array([x["target"] for x in batch], dtype=np.float64)
                total += float((np.logaddexp(0, logits) - labels * logits).sum())
                count += len(batch)
                batch.clear()
        if batch:
            logits = self._forward([x["features"] for x in batch])[2]
            labels = np.array([x["target"] for x in batch], dtype=np.float64)
            total += float((np.logaddexp(0, logits) - labels * logits).sum())
            count += len(batch)
        return total / count

    def train(self, examples, updates: int = 200, batch_size: int = 32,
              seed: int = 0, learning_rate: float = 0.03,
              replay_capacity: int = 4096) -> dict:
        np = _numpy()
        updates = _index(updates, "updates", 0, 100_000)
        batch_size = _index(batch_size, "batch_size", 1, 65_536)
        seed = _index(seed, "seed", 0, 2**63 - 1)
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, numbers.Real):
            raise ValueError("learning_rate must be a finite positive number")
        learning_rate = float(learning_rate)
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if updates * batch_size * self.width > 100_000_000:
            raise ValueError("requested training work exceeds the CPU reference budget")
        if isinstance(examples, DRAMReplay):
            replay = examples
        else:
            replay = DRAMReplay(replay_capacity, feature_dim=2)
            replay.extend(examples)
        if replay.feature_dim != 2 or not len(replay):
            raise ValueError("student training requires nonempty two-feature replay")
        digest = hashlib.sha256()
        for row in replay.iter_rows():
            if any(x not in (0.0, 1.0) for x in row["features"]):
                raise ValueError("student features must contain binary cue and public rule")
            digest.update(_canonical(row) + b"\n")
        initial_hash = self.checkpoint_sha256()
        initial_loss = self._corpus_loss(replay)
        moments = {k: np.zeros_like(w) for k, w in self._weights.items()}
        variances = {k: np.zeros_like(w) for k, w in self._weights.items()}
        rng = random.Random(seed)
        start = time.perf_counter()
        for step in range(1, updates + 1):
            batch = replay.sample(batch_size, rng=rng)
            labels = np.array([row["target"] for row in batch], dtype=np.float64)
            x, hidden, _logits, probability = self._forward([row["features"] for row in batch])
            error = (probability - labels) / batch_size
            dh = error[:, None] * self._weights["W2"][None, :] * (1.0 - hidden**2)
            gradients = {"W1": x.T @ dh, "b1": dh.sum(axis=0),
                         "W2": hidden.T @ error,
                         "b2": np.array([error.sum()], dtype=np.float64)}
            for name, gradient in gradients.items():
                moments[name] = 0.9 * moments[name] + 0.1 * gradient
                variances[name] = 0.999 * variances[name] + 0.001 * gradient**2
                m_hat = moments[name] / (1.0 - 0.9**step)
                v_hat = variances[name] / (1.0 - 0.999**step)
                self._weights[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
                if not np.isfinite(self._weights[name]).all():
                    raise RuntimeError("training produced a non-finite parameter")
            self.training_updates += 1
        elapsed = time.perf_counter() - start
        parameter_bytes = sum(int(w.nbytes) for w in self._weights.values())
        return {
            "evidence": "synthetic_oracle_target_imitation_cpu",
            "optimizer": "adam", "optimizer_state_reset": True,
            "updates_requested": updates, "updates_completed": updates,
            "batch_size": batch_size, "examples_drawn": updates * batch_size,
            "sampler": "uniform_with_replacement_from_host_dram_replay",
            "learning_rate": learning_rate, "seed": seed,
            "initial_corpus_cross_entropy": initial_loss,
            "final_corpus_cross_entropy": self._corpus_loss(replay),
            "loss_scope": "entire_retained_replay",
            "initial_checkpoint_sha256": initial_hash,
            "checkpoint_sha256": self.checkpoint_sha256(),
            "replay_rows_sha256": digest.hexdigest(),
            "wall_seconds": elapsed, "replay": replay.ledger(),
            "optimizer_moments_payload_bytes": 2 * parameter_bytes,
            "gradient_payload_bytes": parameter_bytes,
            "batch_feature_label_payload_bytes": batch_size * 3 * 8,
            "hidden_and_backprop_payload_bytes": 2 * batch_size * self.width * 8,
            "memory_payloads_are_total_rss": False,
            "host_process_peak_rss_bytes": _process_peak_rss_bytes(),
            "host_process_peak_rss_scope": "whole process since launch; includes other workloads",
            "released_pluto_weights_updated": False,
        }


def collect_demonstrations(rules, interface, seeds: Iterable[int]) -> list[dict]:
    """Collect only actions reached by the explicitly labeled synthetic oracle."""
    from .env import MultiFrontEnv
    seeds = _seed_list(seeds)
    rule_digest = hashlib.sha256(_canonical(rules.to_dict())).hexdigest()
    result = []
    for seed in seeds:
        env = MultiFrontEnv(rules, interface)
        observation, _info = env.reset(seed=seed)
        done = False
        while not done:
            action = env.optimal_action()
            if action["kind"] == "attack":
                cue = observation["visible_cues"][action["front"]]
                if cue is None:
                    raise RuntimeError("oracle demonstration attempted an unobserved cue")
                result.append({"features": [cue, rules.target_xor],
                               "target": action["target"], "episode_seed": seed,
                               "front": action["front"], "rules_sha256": rule_digest,
                               "source": "synthetic_exact_oracle"})
            observation, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
    return result


def evaluate(student: Student, rules, interface, seeds: Iterable[int]) -> dict:
    """Score a frozen checkpoint; evaluation never calls the optimizer."""
    from .env import MultiFrontEnv
    seeds = _seed_list(seeds)
    before = student.checkpoint_sha256()
    records = []
    errors = attacks = 0
    brier_sum = 0.0
    bins = [{"count": 0, "probability_sum": 0.0, "target_sum": 0.0} for _ in range(10)]
    for seed in seeds:
        env = MultiFrontEnv(rules, interface)
        observation, _info = env.reset(seed=seed)
        oracle = env.oracle_value()
        total = 0.0
        steps = episode_errors = episode_attacks = deadline_misses = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = student.act(observation)
            if action["kind"] == "attack":
                cue = observation["visible_cues"][action["front"]]
                if cue is None:
                    raise RuntimeError("student attack used an unavailable cue")
                target = int(cue) ^ rules.target_xor
                probability = student.predict_probability(cue, rules.target_xor)
                episode_errors += int(action["target"] != target)
                episode_attacks += 1
                brier_sum += (probability - target)**2
                bucket = bins[min(9, int(probability * 10))]
                bucket["count"] += 1
                bucket["probability_sum"] += probability
                bucket["target_sum"] += target
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            deadline_misses += int(info.get("outcome") == "deadline_before_completion")
        if total > oracle + 1e-12:
            raise RuntimeError("student exceeded declared exact interface oracle")
        errors += episode_errors
        attacks += episode_attacks
        records.append({"seed": seed, "reward": total, "interface_oracle_value": oracle,
                        "interface_regret": max(0.0, oracle - total), "decisions": steps,
                        "attacks": episode_attacks, "target_errors": episode_errors,
                        "deadline_misses": deadline_misses,
                        "terminated": terminated, "truncated": truncated})
    if student.checkpoint_sha256() != before:
        raise RuntimeError("evaluation mutated the checkpoint")
    calibration = []
    ece = 0.0
    for index, bucket in enumerate(bins):
        count = bucket["count"]
        mean_probability = bucket["probability_sum"] / count if count else None
        target_frequency = bucket["target_sum"] / count if count else None
        if count:
            ece += count * abs(mean_probability - target_frequency)
        calibration.append({"bin": index, "count": count,
                            "mean_probability": mean_probability,
                            "positive_target_frequency": target_frequency})
    return {
        "evidence": "synthetic_exact_environment_evaluation", "checkpoint_sha256": before,
        "rules": rules.to_dict(), "interface": interface.to_dict(), "episodes": records,
        "mean_reward": sum(x["reward"] for x in records) / len(records),
        "mean_interface_regret": sum(x["interface_regret"] for x in records) / len(records),
        "interface_oracle_value": records[0]["interface_oracle_value"],
        "attacks": attacks, "target_errors": errors,
        "target_error_rate": errors / attacks if attacks else None,
        "brier_score": brier_sum / attacks if attacks else None,
        "expected_calibration_error_10_bins": ece / attacks if attacks else None,
        "calibration_bins": calibration,
        "deadline_misses": sum(x["deadline_misses"] for x in records),
        "frozen_during_evaluation": True,
    }
