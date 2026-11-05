"""Deterministic paired-view storyworlds and leakage-safe world-group splits."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

A = {
    "room": ["amber", "blue", "coral", "dusk", "elm"],
    "key": ["copper", "iron", "jade", "pearl"],
    "gate": ["closed", "open"],
    "energy": ["empty", "low", "warm", "full"],
    "sigil": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
}
B = {
    "room": ["atrium", "bay", "crypt", "den", "esplanade"],
    "key": ["bronze", "steel", "emerald", "ivory"],
    "gate": ["bolted", "released"],
    "energy": ["spent", "tired", "ready", "charged"],
    "sigil": ["owl", "wolf", "hare", "fox", "boar", "ibis"],
}
ACTIONS = ("move", "collect", "flip", "rest", "rotate")
ATTRS = tuple(A)
PATTERN = re.compile(r"<mask>|[a-z0-9_]+")


def state_id(s):
    return ":".join(f"{k}={s[k]}" for k in ATTRS)


def step(s, action):
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    out = dict(s)
    key = {"move": "room", "collect": "key", "flip": "gate", "rest": "energy", "rotate": "sigil"}[action]
    out[key] = (out[key] + 1) % len(A[key])
    return out


def render(s, rng, dialect):
    words = {key: lex[s[key]] for key, lex in (A if dialect == "a" else B).items()}
    if dialect == "a":
        segments = [f"room {words['room']}", f"key {words['key']}",
                    f"gate {words['gate']}", f"energy {words['energy']}",
                    f"sigil {words['sigil']}"]
    else:
        segments = [f"located {words['room']}", f"holding {words['key']}",
                    f"barrier {words['gate']}", f"vigor {words['energy']}",
                    f"crest {words['sigil']}"]
    rng.shuffle(segments)
    return " ; ".join(segments)


def synthetic(seed=11, worlds=96, steps=12):
    if worlds < 4 or steps < 2:
        raise ValueError("require at least 4 worlds and 2 steps")
    rows = []
    for world in range(worlds):
        rng = random.Random(seed * 1000003 + world * 1009)
        state = {k: rng.randrange(len(A[k])) for k in ATTRS}
        for t in range(steps):
            act = rng.choice(ACTIONS)
            nxt = step(state, act)
            rows.append({"world_id": f"world-{world:05d}",
                         "state_id": state_id(state), "next_state_id": state_id(nxt),
                         "view_a": render(state, rng, "a"),
                         "view_b": render(state, rng, "b"),
                         "next_view_a": render(nxt, rng, "a"),
                         "next_view_b": render(nxt, rng, "b"),
                         "action": act, "attributes": dict(state)})
            state = nxt
    return rows


def load_jsonl(path):
    required = {"world_id", "state_id", "next_state_id", "view_a", "view_b",
                "next_view_a", "next_view_b", "action"}
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = required - row.keys()
        if missing:
            raise ValueError(f"line {i}: missing {sorted(missing)}")
        if not all(isinstance(row[k], str) and row[k].strip() for k in required):
            raise ValueError(f"line {i}: required fields must be nonempty strings")
        if "next_world_id" in row and row["next_world_id"] != row["world_id"]:
            raise ValueError(f"line {i}: transition crosses world groups")
        rows.append(row)
    if not rows:
        raise ValueError("empty dataset")
    return rows


def split_worlds(rows, seed=11):
    groups = sorted({r["world_id"] for r in rows})
    if len(groups) < 4:
        raise ValueError("at least 4 distinct world_id groups required")
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * .15))
    n_val = max(1, round(len(groups) * .15))
    tags = {w: ("test" if i < n_test else "val" if i < n_test + n_val else "train")
            for i, w in enumerate(groups)}
    splits = {name: [r for r in rows if tags[r["world_id"]] == name]
              for name in ("train", "val", "test")}
    if any(not v for v in splits.values()):
        raise ValueError("empty split")
    return splits


def digest(rows):
    payload = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"))
                        for r in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def tokens(text):
    return PATTERN.findall(text.lower())


class Vocabulary:
    def __init__(self, rows):
        self.word_to_id = {"<pad>": 0, "<unk>": 1, "<mask>": 2, "<bos>": 3}
        words = set()
        for row in rows:
            for field in ("view_a", "view_b", "next_view_a", "next_view_b"):
                words.update(tokens(row[field]))
        for word in sorted(words):
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)

    def encode(self, text):
        return [3] + [self.word_to_id.get(w, 1) for w in tokens(text)]

    def __len__(self):
        return len(self.word_to_id)


def mask_view(text, rng, p=.35):
    words = tokens(text)
    if not words:
        return "<mask>"
    masked = ["<mask>" if rng.random() < p else w for w in words]
    if masked == words:
        masked[rng.randrange(len(masked))] = "<mask>"
    return " ".join(masked)
