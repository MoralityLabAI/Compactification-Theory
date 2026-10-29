"""Small strict-JSON helpers shared by experiment and designer boundaries."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(data: bytes | str, *, max_bytes: int = 1_048_576):
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, bytes) or len(data) > max_bytes:
        raise ValueError("JSON input exceeds byte budget")
    def reject_constant(value):
        raise ValueError("non-finite JSON number")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number")
        return parsed
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=reject_constant, parse_float=finite_float)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser budget") from exc


def load_json(path: Path, *, max_bytes: int = 1_048_576):
    with Path(path).open("rb") as source:
        return strict_json(source.read(max_bytes + 1), max_bytes=max_bytes)


def write_json(path: Path, value: object, *, exclusive: bool = True):
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as target:
        target.write(text)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    files = sorted(root.glob("*.py")) + sorted(root.glob("*.json"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}

