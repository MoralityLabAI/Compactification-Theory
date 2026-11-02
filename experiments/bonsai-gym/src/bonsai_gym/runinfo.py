"""Provenance block written into every result file."""
from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def gguf_identity(path: str | Path) -> dict:
    """Size plus the sha256 sidecar written at download time (checked against Hugging Face)."""
    p = Path(path)
    side = p.with_name(p.name + ".sha256")
    sha = side.read_text(encoding="utf-8").split()[0] if side.exists() else None
    return {"path": str(p), "size": p.stat().st_size, "sha256_sidecar": sha}


def run_info(args=None, gguf: str | None = None, **extra) -> dict:
    import torch
    try:
        import tokenizers
        tok_v = tokenizers.__version__
    except Exception:
        tok_v = None
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,enforced.power.limit",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    info = {
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", ".")),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "tokenizers": tok_v, "gpu": gpu,
        "argv": sys.argv, "args": vars(args) if args is not None else None,
    }
    if gguf:
        info["gguf"] = gguf_identity(gguf)
    info.update(extra)
    return info
