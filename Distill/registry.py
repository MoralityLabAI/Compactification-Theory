"""Read source-pinned, byte-verified public exam adapters without network access."""
from __future__ import annotations
import hashlib
from pathlib import Path
from .core import _source, compile_quiz, load_adapter, object_keys, read_json, require

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "Distill/registry.json"

def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()

def load_registry(path=REGISTRY, *, root=ROOT):
    root = Path(root).resolve()
    registry = read_json(path)
    object_keys(registry, {"schema_version", "entries"}, "registry")
    require(registry["schema_version"] == "distill.registry.v1", "unsupported registry")
    require(isinstance(registry["entries"], list) and 1 <= len(registry["entries"]) <= 1024, "invalid entries")
    result = {}
    for entry in registry["entries"]:
        object_keys(entry, {"gym_id", "gym_version", "local_path", "adapter", "source"}, "registry entry")
        _source(entry["adapter"])
        _source(entry["source"])
        path = (root / entry["local_path"]).resolve()
        require(path.is_relative_to(root) and path.is_file(), "adapter path escapes root or missing")
        require(path.stat().st_size <= 1_048_576, "adapter too large")
        require(git_blob(path.read_bytes()) == entry["adapter"]["blob"], "adapter blob mismatch")
        a = load_adapter(path)
        require(a["gym_id"] == entry["gym_id"] and a["gym_version"] == entry["gym_version"] and a["source"] == entry["source"], "registry binding mismatch")
        require(a["gym_id"] not in result, "duplicate gym id")
        result[a["gym_id"]] = a
    return result

def inventory(adapters=None):
    adapters = load_registry() if adapters is None else adapters
    rows = []
    for gym_id, adapter in adapters.items():
        rows.append({"gym_id": gym_id, "gym_version": adapter["gym_version"],
                     "source": adapter["source"],
                     "question_counts": {d: len(compile_quiz(adapter, difficulty=d)["questions"]) for d in ("foundation", "standard", "frontier")}})
    return {"schema_version": "distill.inventory.v1", "gym_count": len(rows), "rows": rows,
            "scope": "adapter blobs checked; source design pins retained, source files outside this checkout not fetched"}
