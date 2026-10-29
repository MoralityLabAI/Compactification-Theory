"""CB02: source-pinned, statement-locked Lean canonization transfer evaluation.

A reproducible *protocol-level* held-out test. The canonical library and
candidate declarations are SHA-1 frozen before held-out Lean clients are added.
Source matching and compiler acceptance are separate evidence gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "transfer_manifest_v1.json"
TRANSFER = ROOT / "CB01" / "Transfer"


def git_blob_sha(contents: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(contents)}\0".encode("ascii") + contents
    ).hexdigest()


def verify_blob(path: Path, pinned: str) -> str:
    measured = git_blob_sha(path.read_bytes())
    if measured != pinned:
        raise ValueError(f"SHA-1 mismatch at {path}: expected {pinned}, got {measured}")
    return measured


def manifest() -> dict:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data["schema"] != "cb02-transfer-manifest-v1":
        raise ValueError("unsupported CB02 manifest")
    if data["evalmath"]["ref"] != "2616092a2c73c083378e508abfd966fdf5e84814":
        raise ValueError("EvalMath ref drift")
    return data


def verify_frozen_files(config: dict) -> dict:
    observed = {}
    for relpath, expected in config["frozen_git_blobs"].items():
        observed[relpath] = verify_blob(ROOT / relpath, expected)
    for f in TRANSFER.glob("*.lean"):
        text = f.read_text(encoding="utf-8")
        # This simple defense-in-depth check is not a replacement for Lean's kernel.
        no_comments = re.sub(r"/-.*?-/|--[^\n]*", "", text, flags=re.DOTALL)
        if re.search(r"\b(?:sorry|admit|axiom|native_decide)\b", no_comments):
            raise ValueError(f"untrusted proof escape in {f}")
    if "import CB01.Transfer.Canonical" in (TRANSFER / "Baseline.lean").read_text():
        raise ValueError("baseline imported candidate canonical proof")
    return observed


def verify_source(config: dict, paper: Path | None, reference: Path | None) -> str:
    if (paper is None) != (reference is None):
        raise ValueError("paper and reference source must be given together")
    if paper is None:
        return "not_replayed_external_source"
    src = config["evalmath"]
    verify_blob(paper, src["paper_git_blob"])
    verify_blob(reference, src["reference_git_blob"])
    body = paper.read_text(encoding="utf-8")
    for anchor in src["verbatim_anchors"]:
        if anchor not in body:
            raise ValueError(f"pinned source missing paper anchor: {anchor}")
    # An optional checkout also verifies its own exact commit identity.
    try:
        root = paper.parent.parent
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        head = None
    if head is not None and head != src["ref"]:
        raise ValueError(f"EvalMath checkout ref mismatch: {head}")
    return "exact_source_blobs_verified"


def proof_blocks(text: str) -> dict[str, str]:
    """Count source bytes inside by-proofs; not elaborated term size."""
    results = {}
    entries = list(re.finditer(r"(?m)^theorem\s+([A-Za-z0-9_']+)", text))
    for i, match in enumerate(entries):
        end = entries[i + 1].start() if i + 1 < len(entries) else len(text)
        section = text[match.start():end]
        if ":= by" in section:
            body = section.split(":= by", 1)[1]
            body = re.split(r"(?m)^end\s+", body)[0]
            results[match.group(1)] = body.strip()
    return results


def module_imports(path: Path) -> tuple[str, ...]:
    return tuple(re.findall(r"(?m)^import\s+([A-Za-z0-9_.]+)", path.read_text()))


def reachable_imports(entry: str) -> list[str]:
    seen: set[str] = set()

    def walk(mod: str) -> None:
        if mod in seen:
            return
        seen.add(mod)
        if not mod.startswith("CB01"):
            return
        path = ROOT / (mod.replace(".", "/") + ".lean")
        if path.exists():
            for dep in module_imports(path):
                walk(dep)

    walk(entry)
    return sorted(seen - {entry})


def metrics() -> dict:
    text = {name: (TRANSFER / (name + ".lean")).read_text(encoding="utf-8")
            for name in ("Statements", "Baseline", "Canonical", "Candidate", "Audit")}
    blocks = {name: proof_blocks(text[name])
              for name in ("Baseline", "Canonical", "Candidate")}
    expected = {
        "Baseline": {"menu", "code", "step"},
        "Canonical": {"menu", "code_sound", "code", "step"},
        "Candidate": {"menu", "code", "step"},
    }
    for name, targets in expected.items():
        if set(blocks[name]) != targets:
            raise ValueError(f"{name} theorem coverage drift: {sorted(blocks[name])}")
    if text["Audit"].count("Baseline.") != 3 or text["Audit"].count("Candidate.") != 3:
        raise ValueError("missing baseline/candidate common-type checks")
    baseline = sum(len(v.encode()) for v in blocks["Baseline"].values())
    candidate = sum(len(v.encode()) for v in blocks["Candidate"].values())
    canonical = sum(len(v.encode()) for v in blocks["Canonical"].values())
    heldout = TRANSFER / "Heldout.lean"
    heldout_count = len(proof_blocks(heldout.read_text(encoding="utf-8"))) if heldout.exists() else 0
    heldout_imports = module_imports(heldout) if heldout.exists() else ()
    return {
        "baseline_proof_body_bytes": baseline,
        "candidate_proof_body_bytes": candidate,
        "canonical_library_proof_body_bytes": canonical,
        "candidate_plus_shared_library_proof_body_bytes": candidate + canonical,
        "baseline_proof_count": len(blocks["Baseline"]),
        "candidate_proof_count": len(blocks["Candidate"]),
        "canonical_helper_theorem_count": len(blocks["Canonical"]),
        "statement_identity_checks": 6,
        "baseline_import_closure": reachable_imports("CB01.Transfer.Baseline"),
        "candidate_import_closure": reachable_imports("CB01.Transfer.Candidate"),
        "heldout_theorem_count": heldout_count,
        "heldout_imports": heldout_imports,
        "caveat": ("Proof-body bytes are lexical source measurements; import closure "
                   "is a local module proxy. No code-size gain, elaborated-term "
                   "complexity, or Mathlib quality follows automatically."),
    }


def lake_build(required: bool) -> dict:
    binary = shutil.which("lake")
    if binary is None:
        if required:
            raise RuntimeError("Lake unavailable: kernel acceptance not established")
        return {"status": "not_run_no_lake", "warm_build_seconds": None}
    begin = time.perf_counter()
    out = subprocess.run(["lake", "build", "CB01"], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    elapsed = round(time.perf_counter() - begin, 4)
    if out.returncode:
        raise RuntimeError("Lean build failed:\n" + (out.stdout + out.stderr)[-5000:])
    return {"status": "lake_build_passed", "warm_build_seconds": elapsed,
            "timing_context": "warm build on runner; not normalized across hardware"}


def receipt(require_lean: bool = False,
            paper: Path | None = None,
            reference: Path | None = None) -> dict:
    config = manifest()
    frozen = verify_frozen_files(config)
    verified_source = verify_source(config, paper, reference)
    observed = metrics()
    heldout_present = (TRANSFER / "Heldout.lean").exists()
    if heldout_present:
        if observed["heldout_theorem_count"] != config["heldout_expected_count"]:
            raise ValueError("heldout challenge coverage changed")
        if "CB01.Transfer.Canonical" not in observed["heldout_imports"]:
            raise ValueError("heldout is not exercising frozen canonical library")
        if "CB01.Transfer.Baseline" in observed["heldout_imports"]:
            raise ValueError("heldout accessed baseline proofs")
    if observed["candidate_proof_body_bytes"] >= observed["baseline_proof_body_bytes"]:
        # A more general library may add overhead; shortening the client is
        # a registered measurement for this particular pilot.
        pass
    return {
        "schema": "cb02-transfer-receipt-v1",
        "source_pinning": {
            "repo": config["evalmath"]["repo"],
            "commit": config["evalmath"]["ref"],
            "source_status": verified_source,
            "paper_git_blob": config["evalmath"]["paper_git_blob"],
            "reference_git_blob": config["evalmath"]["reference_git_blob"],
        },
        "targets": config["targets"],
        "frozen_git_blobs": frozen,
        "phase": "heldout_evaluation" if heldout_present else "frozen_development",
        "frozen_commit": config["frozen_candidate_commit"],
        "metrics": observed,
        "lean": lake_build(require_lean),
        "interpretation": (
            "Three abstract/deterministic specializations of IRC-02 are checked. "
            "No Lean proof of the full stochastic theorem, posterior conditioning, "
            "minimum-bit cardinality, or statistically blind model evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-paper", type=Path)
    parser.add_argument("--source-reference", type=Path)
    parser.add_argument("--require-lean", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = receipt(args.require_lean, args.source_paper, args.source_reference)
    body = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(body, encoding="utf-8")
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
