"""CPU math, delayed-rule gym, source identity and physical donor routing."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

RANKS = (1, 2, 3, 4, 7, 8, 16)
LAYERS = (6, 12, 18)


def seed(*items):
    return int.from_bytes(hashlib.sha256(json.dumps(items, sort_keys=True).encode()).digest()[:8], "little")


def fingerprint(root):
    names = ("preflight.py", "smoke.py", "download.py", "core.py", "runtime.py", "campaign.py",
             "fit.py", "PREREGISTRATION.md")
    hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def gym(tokenizer):
    rows = []
    candidates = [tokenizer.encode(s, add_special_tokens=False) for s in (" A", " B")]
    if any(len(x) != 1 for x in candidates) or candidates[0] == candidates[1]:
        raise ValueError("Candidate answers must be distinct single tokens")
    for split, n, rng_seed in (("train", 128, 2026111601), ("calibration", 64, 2026111602),
                               ("heldout", 96, 2026111603)):
        rng = np.random.default_rng(rng_seed)
        design = [(j % 2, (j // 2) % 2, 2 + (j // 4) % 7) for j in range(n)]
        rng.shuffle(design)
        for j, (flip, cue, delay) in enumerate(design):
            nonce = int(rng.integers(10_000, 99_999))
            distractors = rng.choice(list("xyz"), size=delay)
            prompt = (f"ID {nonce}. Rule: {'flip' if flip else 'copy'}. Cue: {'AB'[cue]}. "
                      + " ".join(f"Wait {x}." for x in distractors)
                      + " Release. Answer:")
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            while len(ids) < 32:
                prompt = "Note. " + prompt
                ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not 32 <= len(ids) <= 64:
                raise ValueError(f"Prompt length {len(ids)} outside registered 32..64")
            rows.append(dict(state_id=f"{split}_{j:03d}", split=split, rule="flip" if flip else "copy",
                             flip=flip, cue=cue, delay=delay, answer=cue ^ flip,
                             prompt=prompt, input_ids=ids, candidate_ids=[x[0] for x in candidates]))
    if len({x["prompt"] for x in rows}) != len(rows):
        raise ValueError("Duplicate prompts")
    return rows


def haar(d, k, key):
    a = np.random.default_rng(seed(key)).standard_normal((d, k))
    q, r = np.linalg.qr(a, mode="reduced")
    return q * np.where(np.diag(r) >= 0, 1, -1)


def dark_frame(u, key):
    a = haar(len(u), u.shape[1], key)
    a -= u @ (u.T @ a)
    q, _ = np.linalg.qr(a, mode="reduced")
    if np.linalg.norm(u.T @ q) > 1e-7:
        raise ValueError("Dark frame is not orthogonal")
    return q


def normed(v, radius):
    length = np.linalg.norm(v)
    if length < 1e-12:
        raise ValueError("Zero intervention direction; cannot silently skip")
    return v * (radius / length)


def route(u, v, z, arm, radius):
    if arm == "dark_to_observed":
        direction = u @ (v.T @ z)
    elif arm == "observed_to_dark":
        direction = v @ (u.T @ z)
    elif arm == "observed_to_observed":
        direction = u @ (u.T @ z)
    else:
        raise ValueError(arm)
    return normed(direction, radius)


def stable(a, b):
    return abs(a-b) <= max(0.02, 0.25*max(abs(a), abs(b)))


def rank_p(value, null, upper=True):
    a = np.asarray(null)
    if len(a) != 128 or not np.isfinite(a).all() or np.ptp(a) < 1e-12:
        return None
    return float((1 + np.sum(a >= value if upper else a <= value)) / 129)


def holm(values):
    # Missing directions still count toward the declared family size.
    raw = [1.0 if x is None else x for x in values]
    order = np.argsort(raw)
    result = [None] * len(raw)
    previous = 0.0
    for j, i in enumerate(order):
        previous = min(1., max(previous, (len(raw)-j)*raw[i]))
        result[i] = None if values[i] is None else previous
    return result


def ridge(x, y, strength=0.01):
    return np.linalg.solve(x.T @ x / len(x) + strength*np.eye(x.shape[1]), x.T @ y / len(x))


def features(rows):
    return np.asarray([[1, 2*x["flip"]-1, 2*x["cue"]-1,
                         (2*x["flip"]-1)*(2*x["cue"]-1), x["delay"]/8] for x in rows], dtype=float)


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
