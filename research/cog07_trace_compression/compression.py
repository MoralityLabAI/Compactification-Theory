"""Finite version-space compression with an executable, byte-counted decoder.

The class and public feature extractor are fixed before data selection. This
module proves its finite certificate by enumerating the declared hypotheses.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import struct
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


FEATURE_NAMES = (
    "all_premises_known", "conclusion_is_new", "conclusion_is_goal_ancestor",
    "conclusion_is_goal", "one_premise", "all_premises_are_initial_facts",
    "conclusion_occurs_as_a_premise", "at_least_half_the_atoms_known",
)
HEADER = struct.Struct(">4sBH")
MAGIC = b"COG7"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Example:
    bits: int
    label: int
    root_id: str = ""
    split: str = "train"
    action: int = -1
    state_sha256: str = ""


@dataclass(frozen=True)
class Hypothesis:
    required: int
    forbidden: int
    constant_false: bool = False

    def predict(self, bits: int) -> int:
        return int(not self.constant_false and bits & self.required == self.required
                   and bits & self.forbidden == 0)

    @property
    def complexity(self):
        return 1 if self.constant_false else (self.required | self.forbidden).bit_count()


@lru_cache(maxsize=10)
def hypotheses(dimension: int) -> tuple[Hypothesis, ...]:
    if type(dimension) is not int or not 1 <= dimension <= 8:
        raise ValueError("feature dimension must be in 1..8")
    values = []
    for literals in itertools.product((0, 1, -1), repeat=dimension):
        required = sum(1 << i for i, x in enumerate(literals) if x == 1)
        forbidden = sum(1 << i for i, x in enumerate(literals) if x == -1)
        values.append(Hypothesis(required, forbidden))
    return tuple(values + [Hypothesis(0, 0, True)])


@lru_cache(maxsize=2048)
def positive_mask(dimension: int, bits: int) -> int:
    if type(bits) is not int or not 0 <= bits < 1 << dimension:
        raise ValueError("feature bits exceed the declared dimension")
    return sum(1 << i for i, hypothesis in enumerate(hypotheses(dimension))
               if hypothesis.predict(bits))


def constraint(example: Example, dimension: int) -> int:
    if type(example.label) is not int or example.label not in (0, 1):
        raise ValueError("labels must be integer bits")
    mask = positive_mask(dimension, example.bits)
    return mask if example.label else ((1 << len(hypotheses(dimension))) - 1) ^ mask


def version_space(examples: Iterable[Example], dimension: int) -> int:
    mask = (1 << len(hypotheses(dimension))) - 1
    for example in examples:
        mask &= constraint(example, dimension)
    return mask


def greedy_cover(examples: list[Example], dimension: int, *, require_realizable=True) -> list[int]:
    """Preserve all full-sample survivors by covering its excluded hypotheses."""
    if any(x.split != "train" for x in examples):
        raise ValueError("compression selection only accepts the training split")
    target = version_space(examples, dimension)
    if require_realizable and not target:
        raise ValueError("the declared hypothesis class cannot fit the sample")
    all_h = (1 << len(hypotheses(dimension))) - 1
    remaining = all_h ^ target
    # Repeated feature/label pairs impose the same constraint. Keep the first
    # provenance index so the selector remains deterministic and auditable.
    unique: dict[tuple[int, int], int] = {}
    for index, example in enumerate(examples):
        unique.setdefault((example.bits, example.label), index)
    exclusions = {index: all_h ^ constraint(examples[index], dimension)
                  for index in unique.values()}
    selected = []
    while remaining:
        best = min(exclusions, key=lambda i: (-(remaining & exclusions[i]).bit_count(), i))
        removed = remaining & exclusions[best]
        if not removed:
            raise RuntimeError("disagreement-cover invariant failed")
        selected.append(best)
        remaining &= ~removed
    if version_space([examples[i] for i in selected], dimension) != target:
        raise RuntimeError("version-space certificate failed")
    return selected


def encode(examples: list[Example], dimension: int) -> bytes:
    """Transmit selected feature/label values with a seven-byte header."""
    hypotheses(dimension)
    if len(examples) > 65535:
        raise ValueError("too many selected examples")
    accumulator = 0
    for example in examples:
        constraint(example, dimension)
        accumulator = (accumulator << (dimension + 1)) | (example.bits << 1) | example.label
    bits = len(examples) * (dimension + 1)
    padding = (-bits) % 8
    payload = (accumulator << padding).to_bytes((bits + 7) // 8, "big")
    return HEADER.pack(MAGIC, dimension, len(examples)) + payload


def decode(packet: bytes) -> tuple[int, list[Example]]:
    if len(packet) < HEADER.size:
        raise ValueError("truncated packet")
    magic, dimension, count = HEADER.unpack(packet[:HEADER.size])
    if magic != MAGIC:
        raise ValueError("unsupported codec")
    hypotheses(dimension)
    bits = count * (dimension + 1)
    expected = HEADER.size + (bits + 7) // 8
    if len(packet) != expected:
        raise ValueError("packet length mismatch")
    payload = int.from_bytes(packet[HEADER.size:], "big")
    padding = (-bits) % 8
    if payload & ((1 << padding) - 1):
        raise ValueError("noncanonical padding")
    payload >>= padding
    examples = []
    field_mask = (1 << (dimension + 1)) - 1
    for _ in range(count):
        field = payload & field_mask
        examples.append(Example(field >> 1, field & 1))
        payload >>= dimension + 1
    return dimension, list(reversed(examples))


def reconstruct(examples: Iterable[Example], dimension: int) -> tuple[int, Hypothesis]:
    remaining = version_space(examples, dimension)
    if not remaining:
        raise ValueError("no consistent decoder hypothesis")
    family = hypotheses(dimension)
    index = min((i for i in range(len(family)) if remaining >> i & 1),
                key=lambda i: (family[i].complexity, i))
    return index, family[index]


def ancestors(world):
    needed = {world["goal"]}
    while True:
        extra = {p for rule in world["rules"] if rule["conclusion"] in needed
                 for p in rule["premises"]}
        if extra <= needed:
            return needed
        needed |= extra


def public_bits(observation, action: int) -> int:
    world = observation["world"]
    if type(action) is not int or not 0 <= action < len(world["rules"]):
        raise ValueError("candidate action out of range")
    rule = world["rules"][action]
    known, initial = set(observation["known"]), set(world["facts"])
    premises, conclusion = set(rule["premises"]), rule["conclusion"]
    values = (
        premises <= known, conclusion not in known, conclusion in ancestors(world),
        conclusion == world["goal"], len(premises) == 1, premises <= initial,
        any(conclusion in other["premises"] for other in world["rules"]),
        2 * len(known) >= len(world["atoms"]),
    )
    return sum(int(value) << i for i, value in enumerate(values))


def load_examples(path: Path) -> list[Example]:
    examples = []
    custody: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split, root = row["split"], row["root_id"]
        if split not in ("train", "calibration", "ordinary_test", "ood_test"):
            raise ValueError("unknown corpus split")
        if root in custody and custody[root] != split:
            raise ValueError("root occurs in multiple splits")
        custody[root] = split
        observation = row["public"]
        candidates, labels = row["candidates"], row["labels"]["useful"]
        if len(candidates) != len(labels):
            raise ValueError("label count mismatch")
        for candidate, label in zip(candidates, labels):
            if type(label) is not bool:
                raise ValueError("producer label must be boolean")
            action = candidate["action"]
            bits = public_bits(observation, action)
            # The first three bits define the specified structural curriculum
            # label. This independently checks every imported target.
            expected = int(bits & 7 == 7)
            if int(label) != expected:
                raise ValueError("producer useful-label mismatch")
            examples.append(Example(bits, expected, root, split, action, digest(observation)))
    return examples


def score(hypothesis: Hypothesis, examples: list[Example]) -> dict:
    errors = [hypothesis.predict(x.bits) != x.label for x in examples]
    by_root: dict[str, list[bool]] = {}
    for example, error in zip(examples, errors):
        by_root.setdefault(example.root_id, []).append(error)
    return {"examples": len(examples), "errors": sum(errors), "roots": len(by_root),
            "roots_with_any_error": sum(any(v) for v in by_root.values()),
            "per_root": {root: {"examples": len(v), "errors": sum(v)}
                         for root, v in sorted(by_root.items())}}


def build_report(examples: list[Example], protocol: dict, output: Path) -> dict:
    if (protocol.get("schema_version") != 1 or protocol.get("feature_count") != 8
            or protocol.get("selection_split") != "train"
            or protocol.get("development_split") != "calibration"
            or protocol.get("report_splits") != ["ordinary_test", "ood_test"]):
        raise ValueError("unsupported protocol schema, feature map or split contract")
    dimension = protocol["feature_count"]
    train = [x for x in examples if x.split == "train"]
    if not train:
        raise ValueError("empty training corpus")
    selected_indices = greedy_cover(train, dimension)
    selected = [train[i] for i in selected_indices]
    packet = encode(selected, dimension)
    decoded_dimension, decoded = decode(packet)
    target = version_space(train, dimension)
    restored = version_space(decoded, decoded_dimension)
    if target != restored:
        raise RuntimeError("decoded finite certificate failed")
    full_id, full_model = reconstruct(train, dimension)
    compressed_id, compressed_model = reconstruct(decoded, dimension)
    unique_training = {}
    for example in train:
        key = (example.bits, example.label)
        unique_training.setdefault(key, {"example": example, "count": 0})["count"] += 1
    deduplicated = [v["example"] for v in unique_training.values()]
    arms: list[tuple[str, Hypothesis, list[Example]]] = [
        ("full", full_model, train), ("greedy_cover", compressed_model, decoded)]
    for seed in protocol["random_control_seeds"]:
        subset = random.Random(seed).sample(train, min(len(selected), len(train)))
        _, model = reconstruct(subset, dimension)
        arms.append((f"random_{seed}", model, subset))
    positive = [x for x in train if x.label]
    positive_indices = greedy_cover(positive, dimension)
    positive_selected = [positive[i] for i in positive_indices]
    _, positive_model = reconstruct(positive_selected, dimension)
    arms.append(("positive_only", positive_model, positive_selected))
    scores = []
    for split in ("train", "calibration", "ordinary_test", "ood_test"):
        partition = [x for x in examples if x.split == split]
        for arm, model, subset in arms:
            measured = score(model, partition)
            measured.update(arm=arm, split=split, selected_examples=len(subset),
                            transmitted_bytes=len(encode(subset, dimension)),
                            preserved_training_version_space=version_space(subset, dimension) == target)
            scores.append(measured)
    output.mkdir(parents=True, exist_ok=True)
    (output / "compressed.bin").write_bytes(packet)
    # This is an audit sidecar. The decoder consumes compressed.bin alone, with
    # the fixed feature map and hypothesis class shared in its source.
    sidecar = {"selected_training_indices": selected_indices,
               "selected_examples": [asdict(x) for x in selected],
               "packet_sha256": hashlib.sha256(packet).hexdigest()}
    sidecar_bytes = (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode()
    (output / "selected.json").write_bytes(sidecar_bytes)
    constraint_record = {
        "feature_count": dimension, "root_ids": sorted({x.root_id for x in train}),
        "constraints": [{"bits": bits, "label": label, "multiplicity": v["count"]}
                        for (bits, label), v in sorted(unique_training.items())],
    }
    constraints_bytes = (json.dumps(constraint_record, indent=2, sort_keys=True) + "\n").encode()
    (output / "training_constraints.json").write_bytes(constraints_bytes)
    report = {"schema": "COG07/finite-compression/v1", "evidence_class": protocol["evidence_class"],
              "protocol_sha256": digest(protocol), "feature_names": list(FEATURE_NAMES),
              "hypotheses": len(hypotheses(dimension)), "surviving_hypotheses": target.bit_count(),
              "training_examples": len(train), "training_roots": len({x.root_id for x in train}),
              "distinct_training_constraints": len(deduplicated),
              "deduplicated_feature_sample_bytes": len(encode(deduplicated, dimension)),
              "training_constraints_sha256": hashlib.sha256(constraints_bytes).hexdigest(),
              "selected_examples": len(selected), "selected_roots": len({x.root_id for x in selected}),
              "selected_positive_examples": sum(x.label for x in selected),
              "selected_negative_examples": sum(1 - x.label for x in selected),
              "constraint_cover_bound": len(hypotheses(dimension)) - target.bit_count(),
              "version_space_preserved": target == restored,
              "full_decoder_id": full_id, "compressed_decoder_id": compressed_id,
              "decoder": asdict(compressed_model), "header_bytes": HEADER.size,
              "payload_bytes": len(packet) - HEADER.size,
              "transmitted_bytes": len(packet), "audit_sidecar_bytes": len(sidecar_bytes),
              "full_feature_sample_bytes": len(encode(train, dimension)),
              "shared_context": "fixed feature extractor, ordered hypothesis class and decoder source",
              "packet_sha256": hashlib.sha256(packet).hexdigest(), "scores": scores,
              "analytic_reference": {
                  "label_formula": "all_premises_known AND conclusion_is_new AND conclusion_is_goal_ancestor",
                  "hypothesis": {"required": 7, "forbidden": 0, "constant_false": False},
                  "bit_patterns_checked": 1 << dimension,
                  "decoder_disagreements": sum(compressed_model.predict(bits) != int(bits & 7 == 7)
                                               for bits in range(1 << dimension)),
                  "scope": "The exact label formula is available in the curriculum specification. "
                           "Zero error follows analytically when the unique target hypothesis is recovered.",
              },
              "scope": "Finite policy-class compression on a symbolic curriculum. "
                       "The complete training version space is certified. "
                       "Neural model size and open-problem proof discovery remain unmeasured."}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranking", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    expected_input = protocol.get("expected_ranking_sha256")
    if expected_input is not None and hashlib.sha256(args.ranking.read_bytes()).hexdigest() != expected_input:
        raise ValueError("ranking file differs from protocol commitment")
    examples = load_examples(args.ranking)
    report = build_report(examples, protocol, args.output)
    report["input_sha256"] = {"ranking.jsonl": hashlib.sha256(args.ranking.read_bytes()).hexdigest()}
    report["source_sha256"] = {"compression.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "protocol.json": hashlib.sha256(args.protocol.read_bytes()).hexdigest()}
    report["shared_decoder_source_bytes"] = Path(__file__).stat().st_size
    report["shared_protocol_bytes"] = args.protocol.stat().st_size
    report["fixed_class_hypothesis_id_bits"] = (report["hypotheses"] - 1).bit_length()
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ["training_examples", "training_roots", "hypotheses",
                     "surviving_hypotheses", "selected_examples", "transmitted_bytes", "version_space_preserved"]}))


if __name__ == "__main__":
    main()
