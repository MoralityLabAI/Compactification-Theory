"""Two native exact-check integrations with source-byte and toy-oracle parity.

Only explicitly selected, reviewed local gym modules are loaded. Never execute
a module path, command, or certificate supplied by a model or packet.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from .core import evaluate, require
from .registry import ROOT, git_blob
from .checkers import cycle_bits, pluto_count

BASE = "ec93d1e0a4a16506df0e189a1f54ae9fcbc70dd8"
PINS = {
    ("IRC-PG-01", "equal_weight_fronts"): (
        "research/pluto_gym/math_ref.py", "76684697463a8a9d9f0a0a4b216b87580cdfa179",
        "research/pluto_gym/PROTOCOL.md", "102ed9a108f6ee9302cd13e59bedc5b602f11373"),
    ("COE-01", "cycle_observer"): (
        "research/chromatic_observer_gym_v1/observer_codes.py", "c37e884cda5406195262e368978176f23ed95091",
        "research/chromatic_observer_gym_v1/README.md", "ee7307245cb1ea3021d5d31b2ba7b3556f1c4766"),
}

def _load(root, path, blob):
    p = (Path(root).resolve() / path).resolve()
    require(p.is_relative_to(Path(root).resolve()), "native module path escapes root")
    require(git_blob(p.read_bytes()) == blob, "native verifier blob mismatch")
    spec = importlib.util.spec_from_file_location("_distill_native_" + blob, p)
    require(spec is not None and spec.loader is not None, "native module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def grade_native(packet, response, *, root=ROOT):
    case = packet.get("public_case")
    key = (packet.get("gym_id"), case.get("kind")) if isinstance(case, dict) else None
    if key not in PINS or not any(q["id"] == "case.solve" for q in packet["questions"]):
        raise ValueError("no registered native check for this packet section")
    path, blob, design, design_blob = PINS[key]
    require(packet["source"] == {"repository": "MoralityLabAI/Compactification-Theory", "commit": BASE, "path": design, "blob": design_blob}, "native design pin mismatch")
    require(git_blob((Path(root) / design).read_bytes()) == design_blob, "native design bytes mismatch")
    module = _load(root, path, blob)
    def check(c, fields):
        d = c["data"]
        if key[0] == "IRC-PG-01":
            toy = pluto_count(c, fields)
            exact = module.deadline_capacity(d["fronts"], d["command_ticks"], d["switch_ticks"], d["deadline"])
        else:
            toy = cycle_bits(c, fields)
            n = d["vertices"]
            coloring = module.optimal_coloring(n, [(i, (i+1)%n) for i in range(n)])
            require(len(coloring) == n and all(coloring[i] != coloring[(i+1)%n] for i in range(n)), "invalid native coloring")
            exact = (len(set(coloring))-1).bit_length()
        accepted = fields["answer"].strip() == str(exact)
        require(accepted == toy, "native / independent toy acceptance disagreement")
        # Compare exact values as well; agreement on a wrong answer is insufficient.
        truth = dict(fields, answer=str(exact))
        require((pluto_count if key[0] == "IRC-PG-01" else cycle_bits)(c, truth), "native / independent toy value disagreement")
        return accepted
    receipt = evaluate(packet, response, checkers={"case.solve": check})
    receipt["native_verifier"] = {"repository": "MoralityLabAI/Compactification-Theory", "commit": BASE, "path": path, "blob": blob, "scope": "exact toy final answer and independent toy parity; prose pending"}
    return receipt
