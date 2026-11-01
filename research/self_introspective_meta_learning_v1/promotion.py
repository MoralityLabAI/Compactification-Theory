"""SIML-01 single-writer, finite-demo promotion broker.

The code enforces run ledgers and paired score arithmetic. File permissions alone
do NOT isolate the proposer from broker private seeds. A real deployment needs a
separate credentialed broker process and independent test-family generation.
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import os
import random
import secrets

from gate_math import endpoint_alpha, paired_interval
from microlearner import GRAPH_SHA, WEIGHTS, MicroLearner, canonical, digest, sample


class ProtocolError(ValueError):
    pass


def source_identity() -> str:
    folder = Path(__file__).resolve().parent
    names = ("microlearner.py", "promotion.py", "gate_math.py")
    return digest({name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                   for name in names})


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical(data) + b"\n")
    os.replace(tmp, path)


def _as_model(raw: dict) -> MicroLearner:
    if set(raw) != {"retained_bits", "joint_shear"}:
        raise ProtocolError("unexpected checkpoint representation")
    return MicroLearner(**raw)


class Broker:
    """Serial finite-world broker. Each attempt consumes a fresh alpha index."""

    def __init__(self, root: Path, *, max_blocks: int = 8192):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "state.json"
        self.events_file = self.root / "events.jsonl"
        self.keys_dir = self.root / "broker_private"
        self.keys_dir.mkdir(exist_ok=True)
        self.max_blocks = max_blocks
        if not self.state_file.exists():
            initial = MicroLearner()
            _atomic_json(self.state_file, {
                "schema": "siml01.broker.v1", "graph_sha": GRAPH_SHA,
                "scorer_source_sha": source_identity(), "total_alpha": "1/20",
                "attempt_next": 1, "incumbent": asdict(initial),
                "anchors": {"T0": asdict(initial)},
                "request_ids": [], "attempts": [],
                "event_head": "0" * 64,
            })
        self._load()

    def _load(self) -> dict:
        state = json.loads(self.state_file.read_text())
        if state["schema"] != "siml01.broker.v1":
            raise ProtocolError("unknown broker schema")
        if state["graph_sha"] != GRAPH_SHA or state["scorer_source_sha"] != source_identity():
            raise ProtocolError("graph or scorer source changed after broker initialization")
        self.state = state
        return state

    @property
    def incumbent(self) -> MicroLearner:
        return _as_model(self.state["incumbent"])

    def _record(self, kind: str, payload: dict) -> None:
        entry = {"sequence": len(self.state["attempts"]), "kind": kind,
                 "payload": payload, "previous": self.state["event_head"]}
        entry["hash"] = digest(entry)
        with self.events_file.open("ab") as fp:
            fp.write(canonical(entry) + b"\n")
        self.state["event_head"] = entry["hash"]

    def _record_attempt(self, receipt: dict) -> None:
        self.state["attempts"].append(receipt)
        self._record("promotion_attempt", receipt)
        _atomic_json(self.state_file, self.state)

    def evaluate(self, proposal: dict, candidate: MicroLearner, *,
                 blocks: int = 4096, gain_floor: float = 0.025,
                 target_floor: float = 0.05,
                 retention_tolerance: float = 0.08) -> dict:
        self._load()
        attempt = self.state["attempt_next"]
        self.state["attempt_next"] = attempt + 1
        request = {"attempt": attempt, "proposal": proposal, "candidate": asdict(candidate),
                   "parent": self.state["incumbent"], "graph_sha": GRAPH_SHA,
                   "scorer_source_sha": self.state["scorer_source_sha"],
                   "blocks": blocks, "floors": [gain_floor, target_floor, retention_tolerance]}
        request_id = digest(request)
        receipt = {"schema": "siml01.promotion.v1", "attempt": attempt,
                   "request_id": request_id, "parent_checkpoint": self.incumbent.checkpoint,
                   "candidate_checkpoint": candidate.checkpoint,
                   "graph_sha": GRAPH_SHA, "scorer_source_sha": self.state["scorer_source_sha"],
                   "blocks": blocks, "operator": proposal.get("operator"),
                   "target": proposal.get("target"),
                   "candidate_model": asdict(candidate),
                   "parent_model": self.state["incumbent"],
                   "anchor_models": {k: v for k,v in self.state["anchors"].items()
                                     if k != proposal.get("target")}}
        try:
            if request_id in self.state["request_ids"]:
                raise ProtocolError("duplicate frozen request")
            self.state["request_ids"].append(request_id)
            if type(blocks) is not int or blocks < 1 or blocks > self.max_blocks:
                raise ProtocolError("invalid evaluation budget")
            if proposal.get("target") not in WEIGHTS:
                raise ProtocolError("unknown target")
            if proposal.get("diagnostics", {}).get("checkpoint") != self.incumbent.checkpoint:
                raise ProtocolError("diagnostic checkpoint mismatch")
            if proposal.get("source") is None or proposal.get("competing_explanation") is None:
                raise ProtocolError("missing precommitted proposal fields")
            if self.incumbent.update(proposal.get("operator")) != candidate:
                raise ProtocolError("unregistered candidate mutation")
            if any(not isinstance(f, (int, float)) or not math.isfinite(f)
                   for f in (gain_floor, target_floor, retention_tolerance)):
                raise ProtocolError("invalid floors")
            if not 0 < gain_floor <= 1 or not 0 < target_floor <= 1 or not 0 <= retention_tolerance <= 1:
                raise ProtocolError("invalid threshold")
            # Broker-only entropy. Publish its digest but retain material privately
            # for trusted replay. This example is NOT an OS-isolated broker.
            seed = secrets.token_bytes(32)
            key_path = self.keys_dir / f"attempt_{attempt:04d}.key"
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fp:
                fp.write(seed)
            receipt["seed_sha256"] = hashlib.sha256(seed).hexdigest()
            scores, floors = self._score(candidate, self.incumbent,
                                          proposal["target"], seed, blocks,
                                          gain_floor, target_floor,
                                          retention_tolerance)
            total_alpha = Fraction(self.state["total_alpha"])
            alph = total_alpha / (attempt * (attempt + 1))
            endpoint_share = endpoint_alpha(total_alpha, attempt, len(floors))
            intervals = {key: paired_interval(vals, float(endpoint_share))
                         for key, vals in scores.items()}
            failures = [key for key, floor in floors.items() if intervals[key][0] < floor]
            decisive = [key for key in failures if intervals[key][1] < floors[key]]
            status = ("rejected" if decisive else "insufficient_evidence"
                      if failures else "promoted")
            receipt.update({
                "alpha_attempt": str(alph), "alpha_endpoint": str(endpoint_share),
                "comparators": {"aggregate": self.incumbent.checkpoint,
                                "target": self.incumbent.checkpoint,
                                **{"retain:" + k: _as_model(v).checkpoint
                                   for k, v in self.state["anchors"].items()
                                   if k != proposal["target"]}},
                "mean_differences": {k: math.fsum(v)/len(v) for k, v in scores.items()},
                "intervals": {k: [a, b] for k, (a, b) in intervals.items()},
                "floors": floors, "failed_endpoints": failures,
                "decisive_endpoints": decisive, "status": status,
                "predicted_potential_gain": proposal.get("predicted_potential_gain"),
            })
            if status == "promoted":
                self.state["incumbent"] = asdict(candidate)
                self.state["anchors"].setdefault(proposal["target"], asdict(candidate))
        except (ProtocolError, ValueError, KeyError, TypeError) as exc:
            receipt.update({"status": "invalid", "reason": str(exc)})
        self._record_attempt(receipt)
        return receipt

    def _score(self, candidate: MicroLearner, parent: MicroLearner,
               target: str, seed: bytes, blocks: int, gain_floor: float,
               target_floor: float, retention_tolerance: float,
               protected_override: dict | None = None) -> tuple[dict, dict]:
        raw_anchors = self.state["anchors"] if protected_override is None else protected_override
        protected = {n: _as_model(v) for n, v in raw_anchors.items() if n != target}
        floors = {"aggregate": gain_floor, "target": target_floor,
                  **{"retain:" + n: -retention_tolerance for n in protected}}
        scores = {key: [] for key in floors}
        streams = {n: random.Random(int.from_bytes(hashlib.sha256(seed + n.encode()).digest(), "big"))
                   for n in WEIGHTS}
        for _ in range(blocks):
            aggregate = 0.0
            for node, weight in WEIGHTS.items():
                world, variant, truth = sample(node, streams[node])
                cand = float(candidate.predict(node, world, variant) == truth)
                base = float(parent.predict(node, world, variant) == truth)
                diff = cand - base
                aggregate += weight * diff
                if node == target:
                    scores["target"].append(diff)
                if node in protected:
                    ref = float(protected[node].predict(node, world, variant) == truth)
                    scores["retain:" + node].append(cand - ref)
            scores["aggregate"].append(aggregate)
        return scores, floors

    def verify_scoring(self) -> bool:
        """Trusted replay using private seeds, recorded comparators, frozen sources."""
        self.verify_chain()
        for receipt in self.state["attempts"]:
            if receipt["status"] not in ("promoted", "rejected", "insufficient_evidence"):
                continue
            key_path = self.keys_dir / f"attempt_{receipt['attempt']:04d}.key"
            seed = key_path.read_bytes()
            if hashlib.sha256(seed).hexdigest() != receipt["seed_sha256"]:
                raise ProtocolError("private seed digest mismatch")
            scores, floors = self._score(
                _as_model(receipt["candidate_model"]),
                _as_model(receipt["parent_model"]),
                receipt["target"], seed, receipt["blocks"],
                receipt["floors"]["aggregate"], receipt["floors"]["target"],
                -next((value for k, value in receipt["floors"].items()
                       if k.startswith("retain:")), -0.08),
                receipt["anchor_models"],
            )
            if floors != receipt["floors"]:
                raise ProtocolError("changed registered endpoint floors")
            alpha = endpoint_alpha(Fraction(self.state["total_alpha"]),
                                   receipt["attempt"], len(floors))
            for k, values in scores.items():
                if not math.isclose(math.fsum(values)/len(values),
                                    receipt["mean_differences"][k], abs_tol=1e-12):
                    raise ProtocolError("replayed score mismatch")
                interval = paired_interval(values, float(alpha))
                if any(not math.isclose(a, b, abs_tol=1e-12)
                       for a, b in zip(interval, receipt["intervals"][k])):
                    raise ProtocolError("replayed interval mismatch")
        return True

    def verify_chain(self) -> bool:
        self._load()
        previous = "0" * 64
        records = []
        if not self.events_file.exists():
            return previous == self.state["event_head"] and not self.state["attempts"]
        for line in self.events_file.read_text().splitlines():
            event = json.loads(line)
            claimed = event.pop("hash")
            if event["previous"] != previous or digest(event) != claimed:
                raise ProtocolError("broken event chain")
            previous = claimed
            records.append(event["payload"])
        if previous != self.state["event_head"] or records != self.state["attempts"]:
            raise ProtocolError("event log and snapshot disagree")
        if len(records) + 1 != self.state["attempt_next"]:
            raise ProtocolError("attempt counter has a gap")
        return True


def run_demo(root: Path, blocks: int = 4096) -> dict:
    """Deterministic rule proposer; broker draws fresh private seeds each time."""
    from microlearner import heuristic_proposal, diagnose
    broker = Broker(root)
    for _ in range(2):
        proposal = heuristic_proposal(broker.incumbent)
        if proposal is None:
            break
        candidate = broker.incumbent.update(proposal["operator"])
        broker.evaluate(proposal, candidate, blocks=blocks)
    broker.verify_scoring()
    return {"final_checkpoint": broker.incumbent.checkpoint,
            "final_model": asdict(broker.incumbent),
            "exact_accuracy": diagnose(broker.incumbent)["development_accuracy"],
            "attempts": broker.state["attempts"],
            "chain_verified": True}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="new or existing broker directory")
    ap.add_argument("--blocks", type=int, default=4096)
    args = ap.parse_args()
    result = run_demo(Path(args.output), args.blocks)
    print(json.dumps(result, indent=2, sort_keys=True))
