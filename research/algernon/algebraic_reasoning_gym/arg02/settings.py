"""Config loading shared by the runner and the analysis."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config" / "arg02_config.json"
LOCK = HERE / "config" / "checkpoint_lock.json"


def load_config(smoke=False):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    over = cfg.pop("smoke_overrides")
    if smoke:
        cfg.update({k: v for k, v in over.items() if k != "note"})
        cfg["smoke"] = True
    return cfg


def load_lock():
    return json.loads(LOCK.read_text(encoding="utf-8"))
