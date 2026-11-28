"""Config loading shared by the runner and the analysis."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config" / "arg02_config.json"
LOCK = HERE / "config" / "checkpoint_lock.json"
PROFILES = ("qwen", "nanbeige")


def load_config(smoke=False, profile="qwen"):
    """The registered config, with a model profile's overrides and then smoke overrides applied."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile}")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    smoke_over = cfg.pop("smoke_overrides")
    profiles = cfg.pop("profiles", {})
    cfg.setdefault("model_key", "model")
    cfg.setdefault("ff_key", "ff_control")
    if profile != "qwen":
        prof = dict(profiles[profile])
        smoke_over = prof.pop("smoke_overrides", {})
        prof.pop("note", None)
        cfg.update(prof)
    if smoke:
        cfg.update({k: v for k, v in smoke_over.items() if k != "note"})
        cfg["smoke"] = True
    cfg["profile"] = profile
    return cfg


def load_lock():
    return json.loads(LOCK.read_text(encoding="utf-8"))


def model_entry(cfg, lock=None):
    lock = lock or load_lock()
    if cfg.get("smoke"):
        return lock["smoke" if cfg["profile"] == "qwen" else f"smoke_{cfg['profile']}"]
    return lock[cfg["model_key"]]


def ff_entry(cfg, lock=None):
    if not cfg.get("ff_key"):
        return None
    lock = lock or load_lock()
    if cfg.get("smoke"):
        return lock["smoke"]
    return lock[cfg["ff_key"]]
