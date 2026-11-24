"""TRM router over the suite's 1-bit adapters: pick one adapter (or the bare base) per prompt.

  python router.py --data C:/Users/patri/Documents/Codex/onebit-suite/data \
                   --scores C:/Users/patri/Documents/Codex/onebit-suite/scores --out <results dir>

The TRM is the q1-adapters router (harness/routers.py, after Jolicoeur-Martineau's Tiny Recursive Model):
hashed word/bigram/char-trigram features -> x; latent z and answer y refined by two weight-shared MLPs
(n_inner latent steps per outer step, n_outer outer steps), deep supervision on every outer step.
It is trained on the TRAIN-split prompts of the four families (labels = family) and never sees an eval prompt.

Policies, all scored END-TO-END on the eval prompts with the cross-evaluation scores
scores/<arm>/<family>.jsonl ({"id","score"}; arms base, math, logic, srb, bdb):
  trm            the family adapter TRM ranks first
  trm-eu         expected utility: argmax_arm sum_f P(f|x) * S_hat(f, arm), where S_hat is the mean score of
                 the arm on family f, cross-fitted (computed on the other folds of eval items), so the router
                 can fall back to the base when an adapter does not help its own family
  oracle-family  the true family's adapter;  oracle-best  the best arm per item (upper bound)
  base / always-<arm>
Routing accuracy is over 5 seeds; the end-to-end table uses seed 0.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FAMILIES = ["math", "logic", "srb", "bdb"]
ARMS = ["base"] + FAMILIES
TRAIN_FILES = {"math": ["gold_train.jsonl", "rft_pool.jsonl"], "logic": ["rft_pool.jsonl"],
               "srb": ["train.jsonl"], "bdb": ["train.jsonl"]}
WORD = re.compile(r"[a-z0-9]+")
BUCKETS, EMB, D = 2048, 24, 48


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def text_of(row: dict, view: str) -> str:
    msgs = row["messages"] if view == "full" else [m for m in row["messages"] if m["role"] == "user"]
    return "\n".join(m["content"] for m in msgs)


def feats(text: str) -> list[int]:
    w = WORD.findall(text.lower())
    grams = w + [a + "_" + b for a, b in zip(w, w[1:])]
    t = text.lower()
    grams += ["#" + t[i:i + 3] for i in range(0, max(0, len(t) - 2), 2)]
    return [zlib.crc32(g.encode()) % BUCKETS for g in grams] or [0]


class TRM(nn.Module):
    def __init__(self, n_cls: int, n_inner: int = 3, n_outer: int = 3):
        super().__init__()
        self.emb = nn.EmbeddingBag(BUCKETS, EMB, mode="mean")
        self.inp = nn.Linear(EMB, D)
        self.f = nn.Sequential(nn.Linear(3 * D, 64), nn.GELU(), nn.Linear(64, D))  # z <- z + f(x, y, z)
        self.g = nn.Sequential(nn.Linear(2 * D, 64), nn.GELU(), nn.Linear(64, D))  # y <- y + g(y, z)
        self.head = nn.Linear(D, n_cls)
        self.n_inner, self.n_outer = n_inner, n_outer

    def forward(self, ids, offsets):
        x = self.inp(self.emb(ids, offsets))
        y, z, outs = torch.zeros_like(x), torch.zeros_like(x), []
        for _ in range(self.n_outer):
            for _ in range(self.n_inner):
                z = z + self.f(torch.cat([x, y, z], -1))
            y = y + self.g(torch.cat([y, z], -1))
            outs.append(self.head(y))
        return outs


def batchify(texts):
    ids, offs = [], []
    for t in texts:
        offs.append(len(ids))
        ids.extend(feats(t))
    return torch.tensor(ids), torch.tensor(offs)


def train_trm(texts, labels, seed=0, epochs=20):
    torch.manual_seed(seed)
    m = TRM(len(FAMILIES))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    y = torch.tensor(labels)
    idx, rng = list(range(len(texts))), random.Random(seed)
    feats_cache = [feats(t) for t in texts]
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            ids, offs = [], []
            for j in b:
                offs.append(len(ids))
                ids.extend(feats_cache[j])
            outs = m(torch.tensor(ids), torch.tensor(offs))
            loss = sum(F.cross_entropy(o, y[b]) for o in outs) / len(outs)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def probs(m, texts):
    return F.softmax(m(*batchify(texts))[-1], -1)


def load_scores(root: Path, eval_rows: list[dict]) -> dict[str, dict[str, float]]:
    """S[arm][id] for every eval row; an arm/family file that is missing leaves those cells absent."""
    S = {a: {} for a in ARMS}
    for a in ARMS:
        for fam in FAMILIES:
            p = root / a / f"{fam}.jsonl"
            if p.exists():
                for r in read(p):
                    S[a][r["id"]] = float(r["score"])
    return S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train, ev = [], []
    for k, fam in enumerate(FAMILIES):
        for fn in TRAIN_FILES[fam]:
            train += [(r, k) for r in read(data / fam / fn)]
        ev += [(r, k) for r in read(data / fam / "eval.jsonl")]
    eval_ids = {r["id"] for r, _ in ev}
    assert not eval_ids & {r["id"] for r, _ in train}, "eval prompt ids leaked into router training"
    S = load_scores(Path(a.scores), [r for r, _ in ev])
    complete = [i for i, (r, _) in enumerate(ev) if all(r["id"] in S[arm] for arm in ARMS)]
    report = {"n_train": len(train), "n_eval": len(ev), "n_eval_scored_all_arms": len(complete), "views": {}}
    md = [f"# TRM router over the 1-bit adapters\n", f"Train prompts {len(train)} (train splits only), eval prompts {len(ev)}; "
          f"{len(complete)} eval prompts have a score under every arm.\n"]

    for view in ("full", "content"):
        tr_t, tr_y = [text_of(r, view) for r, _ in train], [k for _, k in train]
        ev_t, ev_y = [text_of(r, view) for r, _ in ev], torch.tensor([k for _, k in ev])
        accs, P0, m0 = [], None, None
        for s in range(a.seeds):
            m = train_trm(tr_t, tr_y, seed=s)
            P = probs(m, ev_t)
            accs.append(float((P.argmax(-1) == ev_y).float().mean()))
            if s == 0:
                P0, m0 = P, m
        n_params = sum(p.numel() for p in m0.parameters())
        conf = [[0] * len(FAMILIES) for _ in FAMILIES]
        for k, p in zip(ev_y.tolist(), P0.argmax(-1).tolist()):
            conf[k][p] += 1
        v = {"trm_params": n_params, "routing_acc": accs, "confusion_seed0": conf}
        if view == "full":
            torch.save(m0.state_dict(), out / "trm_full_seed0.pt")

        if complete:
            rows = [ev[i][0] for i in complete]
            fam_of = [ev[i][1] for i in complete]
            Pc = P0[complete]
            rng = random.Random(0)
            order = list(range(len(rows)))
            rng.shuffle(order)
            fold = {j: n % a.folds for n, j in enumerate(order)}
            choice = {}
            choice["trm"] = [FAMILIES[int(Pc[j].argmax())] for j in range(len(rows))]
            eu = []
            for j in range(len(rows)):   # S_hat from the other folds only
                others = [i for i in range(len(rows)) if fold[i] != fold[j]]
                shat = torch.zeros(len(FAMILIES), len(ARMS))
                for f in range(len(FAMILIES)):
                    idx = [i for i in others if fam_of[i] == f]
                    for c, arm in enumerate(ARMS):
                        shat[f, c] = sum(S[arm][rows[i]["id"]] for i in idx) / max(1, len(idx))
                eu.append(ARMS[int((Pc[j] @ shat).argmax())])
            choice["trm-eu"] = eu
            choice["oracle-family"] = [FAMILIES[f] for f in fam_of]
            choice["oracle-best"] = [max(ARMS, key=lambda arm: S[arm][r["id"]]) for r in rows]
            for arm in ARMS:
                choice[f"always-{arm}"] = [arm] * len(rows)
            table = {}
            for name, ch in choice.items():
                per = {fam: [] for fam in FAMILIES}
                for r, f, arm in zip(rows, fam_of, ch):
                    per[FAMILIES[f]].append(S[arm][r["id"]])
                fam_means = {fam: sum(x) / len(x) for fam, x in per.items() if x}
                table[name] = {"macro": sum(fam_means.values()) / len(fam_means), **fam_means,
                               "base_share": sum(c == "base" for c in ch) / len(ch)}
            v["end_to_end"] = table
        report["views"][view] = v
        md.append(f"\n## view: {view} (TRM params {n_params:,})\n")
        md.append(f"Routing accuracy over {a.seeds} seeds: mean {sum(accs)/len(accs):.3f}, min {min(accs):.3f}.\n")
        md.append("\nConfusion, seed 0 (rows = true family, columns = routed adapter):\n\n| | " + " | ".join(FAMILIES) + " |\n|---|" + "---:|" * len(FAMILIES) + "\n")
        for f, row in zip(FAMILIES, conf):
            md.append(f"| {f} | " + " | ".join(map(str, row)) + " |\n")
        if "end_to_end" in v:
            md.append("\nEnd-to-end score on eval prompts (seed 0; per-family mean of the env/bench scorer, macro = mean of families):\n\n"
                      "| policy | macro | " + " | ".join(FAMILIES) + " | share routed to base |\n|---|---:|" + "---:|" * len(FAMILIES) + "---:|\n")
            for name, t in v["end_to_end"].items():
                md.append(f"| {name} | {t['macro']:.3f} | " + " | ".join(f"{t.get(f, float('nan')):.3f}" for f in FAMILIES) + f" | {t['base_share']:.2f} |\n")
    (out / "router.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "router.md").write_text("".join(md), encoding="utf-8")
    print("".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
