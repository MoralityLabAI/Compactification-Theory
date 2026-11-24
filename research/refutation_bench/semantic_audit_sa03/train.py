"""SA-03 acquisition actor. Gold and full graph are used only by train oracle and verifier."""
import argparse
import hashlib
import heapq
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "semantic_audit_sa02"))
import gym as source
from oracle import closure, minimum_nodes

BUDGETS = (8, 16, 32, 64)
SEEDS = (71, 73)
POLICIES = ("random", "lexical", "proof_first", "semantic_first", "alternating",
            "target_first", "proof_dfs", "proof_bfs", "target_dfs", "adaptive")
KINDS = ("mask", "alias", "formula", "lemma", "proof", "axiom")
FEATURES = 23
EVENTS = FEATURES + len(KINDS)
HIDDEN = 24

def fresh():
    old = list(source.families())
    old_splits = source.SPLITS
    source.SPLITS = {k: (n, 9311+i, m, d)
                     for i, (k, (n, seed, m, d)) in enumerate(old_splits.items())}
    try:
        new = list(source.families())
        source.validate(new)
    finally:
        source.SPLITS = old_splits
    assert not {source.sha(f["source"]) for f in old} & {source.sha(f["source"]) for f in new}
    assert not {a["id"] for f in old for a in f["arms"]} & {a["id"] for f in new for a in f["arms"]}
    train = [f for f in old + new if f["split"] == "train"]
    assert len(train) == 64
    return train, new

def frontier(root, seen, proof=False):
    """Read-only visible frontier with minimum observed path depths and DFS order."""
    order, depths = [], {}
    def walk(x, d, ancestors):
        if x in ancestors:
            raise ValueError("cycle")
        if x in depths and depths[x] <= d:
            return
        depths[x] = d
        if x not in seen:
            if x not in order:
                order.append(x)
            return
        node = seen[x]
        children = node.get("deps", []) if proof else source.refs(node)
        for y in children:
            walk(y, d+1, ancestors | {x})
    if root is not None:
        walk(root, 0, set())
    return order, depths

def visible(session):
    seen, r, t = session.observed, session.root, session.target
    result, ap, sp = source.evidence(r, t, seen)
    if result["verdict"] != "abstain":
        return result, [], [], [], [], [], []
    p, pd = frontier(r, seen, True) if result["axiom"] == "open" else ([], {})
    target, td = frontier(t, seen) if result["semantic"] == "open" else ([], {})
    src, sd = frontier(seen[r]["type"], seen) if r in seen and result["semantic"] == "open" else ([], {})
    all_names = sorted(set(p + target + src))
    counts = Counter(x["kind"] for x in seen.values())
    features = []
    for name in all_names:
        f = [float(name in seq) for seq in (p, target, src)]
        f += [min(32, d.get(name, 0))/32 for d in (pd, td, sd)]
        f += [len(seq)/32 for seq in (p, target, src)]
        f += [len(seen)/128, float(r in seen), float(t in seen),
              float(result["axiom"] == "clean"), float(not target),
              float(not src and r in seen)]
        f += [counts[k]/32 for k in KINDS]
        f += [sum(name in source.refs(n) for n in seen.values())/16,
              float(name in (r, t))]
        assert len(f) == FEATURES
        features.append(f)
    assert set(all_names) <= set(session.discovered) - set(seen)
    return result, all_names, features, p, target, src, (ap, sp, pd)

def event(features, kind):
    return features + [float(kind == k) for k in KINDS]

class TrainOracle:
    """Never instantiated for calibration/test acquisition."""
    def __init__(self, graph, arm):
        self.graph, self.arm = graph, arm
        self.r, self.t = arm["root"], arm["target"]
        self.target = closure(graph, [self.t], source.refs)
        self.support = (closure(graph, [self.r], lambda n: n.get("deps", []))
                        | closure(graph, [graph[self.r]["type"]], source.refs) | self.target)
    def cost(self, seen):
        if self.arm["gold"]["verdict"] == "supported":
            return len(self.support - seen)
        costs = []
        if self.arm["gold"]["semantic"] == "false":
            costs.append(len(self.target - seen))
        if self.arm["gold"]["axiom"] == "extra":
            todo = [(int(self.r not in seen), self.r)]
            dist = {}
            while todo:
                c, n = heapq.heappop(todo)
                if n in dist:
                    continue
                dist[n] = c
                if self.graph[n]["kind"] == "axiom":
                    costs.append(c)
                    break
                for x in self.graph[n].get("deps", []):
                    heapq.heappush(todo, (c + int(x not in seen), x))
        assert costs
        return min(costs)
    def labels(self, acquired, candidates):
        seen = set(acquired)
        c = self.cost(seen)
        labels = [self.cost(seen | {a}) + 1 == c for a in candidates]
        assert c > 0 and any(labels), (c, candidates)
        return labels

def train_trajectories(families):
    records = []
    for f in families:
        assert f["split"] == "train"
        graph = f["graph"]
        for arm in f["arms"]:
            oracle = TrainOracle(graph, arm)
            assert oracle.cost(set()) == len(minimum_nodes(graph, arm))
            for explore in (0.0, 0.15):
                rng = random.Random(int(arm["id"], 16) + int(explore * 100))
                ses = source.Session(graph, arm["root"], arm["target"], 128)
                frames, prev = [], [0.0] * EVENTS
                for _ in range(128):
                    result, names, feat, *_ = visible(ses)
                    if result["verdict"] != "abstain":
                        break
                    labels = oracle.labels(ses.observed, names)
                    frames.append((prev, feat, labels))
                    good = [i for i, x in enumerate(labels) if x]
                    i = rng.randrange(len(names)) if rng.random() < explore else rng.choice(good)
                    n = ses.read(names[i])
                    prev = event(feat[i], n["kind"])
                result, *_ = visible(ses)
                assert result["verdict"] == arm["gold"]["verdict"]
                source.check_certificate(graph, arm, ses.observed, result)
                assert frames
                records.append(frames)
    return records

class Actor(nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        self.recurrent = recurrent
        if recurrent:
            self.gru = nn.GRU(EVENTS, HIDDEN, batch_first=True)
        self.score = nn.Sequential(nn.Linear(FEATURES + (HIDDEN if recurrent else 0), 48),
                                   nn.Tanh(), nn.Linear(48, 1))
    def forward(self, events, candidates, hidden=None):
        if self.recurrent:
            states, hidden = self.gru(events, hidden)
            states = states.unsqueeze(2).expand(-1, -1, candidates.shape[2], -1)
            candidates = torch.cat((candidates, states), dim=-1)
        return self.score(candidates).squeeze(-1), hidden

def batch(rows):
    n, t = len(rows), max(len(x) for x in rows)
    c = max(len(frame[1]) for row in rows for frame in row)
    ev = torch.zeros(n, t, EVENTS)
    feat = torch.zeros(n, t, c, FEATURES)
    valid = torch.zeros(n, t, c, dtype=torch.bool)
    good = torch.zeros_like(valid)
    active = torch.zeros(n, t, dtype=torch.bool)
    for i, row in enumerate(rows):
        for j, (e, f, g) in enumerate(row):
            k = len(f)
            ev[i,j] = torch.tensor(e)
            feat[i,j,:k] = torch.tensor(f)
            valid[i,j,:k] = True
            good[i,j,:k] = torch.tensor(g)
            active[i,j] = True
    # Padded frames are not scored; keep at least one finite logit per row.
    valid[:,:,0] |= ~active
    good[:,:,0] |= ~active
    return ev, feat, valid, good, active

def fit(rows, recurrent, seed, out):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Actor(recurrent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    history = []
    for epoch in range(12):
        order = list(range(len(rows)))
        rng.shuffle(order)
        losses = []
        for offset in range(0, len(order), 16):
            ev, feat, valid, good, active = batch([rows[i] for i in order[offset:offset+16]])
            logits, _ = model(ev, feat)
            denom = torch.logsumexp(logits.masked_fill(~valid, -1e9), dim=-1)
            numer = torch.logsumexp(logits.masked_fill(~good, -1e9), dim=-1)
            loss = (denom - numer)[active].mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(sum(losses)/len(losses))
    tag = ("gru" if recurrent else "static") + "-" + str(seed)
    path = out / (tag + ".pt")
    torch.save(model.state_dict(), path)
    receipt = {"model": tag, "parameters": sum(x.numel() for x in model.parameters()),
               "epochs": 12, "loss": history, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return tag, model.eval(), receipt

def choose_symbolic(policy, ses, rng, i, names, p, t, s, aux):
    ap, sp, pd = aux
    if policy in ("random", "lexical"):
        all_frontier = sorted(set(ses.discovered) - set(ses.observed))
        return rng.choice(all_frontier) if policy == "random" else all_frontier[0]
    if policy == "proof_first":
        return (ap or sp)[0]
    if policy == "semantic_first":
        return (sp or ap)[0]
    if policy == "alternating":
        return ((ap or sp) if i % 2 == 0 else (sp or ap))[0]
    if policy == "target_first":
        return (t or ap or s)[0]
    if policy == "proof_dfs":
        return (p or t or s)[0]
    if policy == "proof_bfs":
        return min(p, key=lambda x: (pd[x], p.index(x))) if p else (t or s)[0]
    if policy == "target_dfs":
        return (t or p or s)[0]
    if policy == "adaptive":
        branches = [x for x in (p, t) if x]
        return min(branches, key=len)[0] if branches else s[0]
    raise ValueError(policy)

@torch.no_grad()
def evaluate_arm(graph, arm, budget, policy, model=None):
    ses = source.Session(graph, arm["root"], arm["target"], budget)
    rng = random.Random(int(arm["id"], 16) + budget)
    hidden = None
    prev = [0.0]*EVENTS
    for i in range(budget):
        result, names, feat, p, t, s, aux = visible(ses)
        if result["verdict"] != "abstain":
            break
        assert names
        if model is None:
            chosen = choose_symbolic(policy, ses, rng, i, names, p, t, s, aux)
            ses.read(chosen)
        else:
            logits, hidden = model(torch.tensor([[prev]]), torch.tensor([[feat]]), hidden)
            k = int(logits[0,0].argmax())
            n = ses.read(names[k])
            prev = event(feat[k], n["kind"])
    result, *_ = visible(ses)
    source.check_certificate(graph, arm, ses.observed, result)
    result["reads"] = len(ses.observed)
    return result

def evaluate(fs, models, out):
    rows, details = [], []
    for split in ("cal", "test_depth", "test_mechanism"):
        subset = [f for f in fs if f["split"] == split]
        for budget in BUDGETS:
            for policy, model in [(x, None) for x in POLICIES] + list(models.items()):
                counts = Counter()
                started = time.process_time()
                for f in subset:
                    joint = True
                    for arm in f["arms"]:
                        res = evaluate_arm(f["graph"], arm, budget, policy, model)
                        gold = arm["gold"]
                        correct = res["verdict"] == gold["verdict"]
                        abstain = res["verdict"] == "abstain"
                        for axis in ("axiom", "semantic"):
                            counts[axis] += res[axis] != "open"
                            assert res[axis] in ("open", gold[axis])
                        counts["correct"] += correct
                        counts["abstain"] += abstain
                        counts["error"] += not correct and not abstain
                        counts["false_rejection"] += res["verdict"] == "defect" and gold["verdict"] == "supported"
                        counts["correct_" + gold["verdict"]] += correct
                        counts["reads"] += res["reads"]
                        if correct:
                            minimum = len(minimum_nodes(f["graph"], arm))
                            assert res["reads"] >= minimum
                            counts["excess"] += res["reads"] - minimum
                        joint &= correct
                        details.append({"id": arm["id"], "split": split, "budget": budget,
                                        "policy": policy, "gold": gold, **res})
                    counts["joint"] += joint
                rows.append({"split":split, "budget":budget, "policy":policy, "n":4*len(subset),
                             **counts, "cpu_seconds":time.process_time()-started})
    (out/"scores.json").write_text(json.dumps(rows, indent=2))
    (out/"cases.jsonl").write_text("".join(json.dumps(x)+"\n" for x in details))
    primary = [x for x in rows if x["split"] == "test_depth" and x["budget"] in (16,32)]
    totals = Counter()
    for x in primary:
        totals[x["policy"]] += x["correct"]
    best_symbolic = max(totals[p] for p in POLICIES)
    passed = all(totals[f"gru-{s}"] >= best_symbolic+4 for s in SEEDS) and not any(x.get("error",0) for x in rows)
    gate = {"primary_totals": dict(totals), "best_symbolic":best_symbolic, "passed":passed}
    (out/"gate.json").write_text(json.dumps(gate, indent=2))
    print("GATE",json.dumps(gate))
    print("SCORES",json.dumps([{k:x.get(k,0) for k in ("split","budget","policy","n","correct","abstain","error","correct_supported","correct_defect","reads","excess","joint")} for x in rows if x["split"].startswith("test")]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lean", default="lean")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train, new = fresh()
    source.native(new, args.out, args.lean)
    (args.out/"source-hashes.json").write_text(json.dumps({f["family"]:source.sha(f["source"]) for f in train+new}, indent=2))
    rows = train_trajectories(train)
    print("TRAIN",len(rows),sum(len(x) for x in rows),flush=True)
    models, receipts = {}, []
    for recurrent in (False, True):
        for seed in SEEDS:
            name, model, receipt = fit(rows, recurrent, seed, args.out)
            models[name] = model
            receipts.append(receipt)
            print("FIT",json.dumps(receipt),flush=True)
    (args.out/"training.json").write_text(json.dumps(receipts, indent=2))
    evaluate(new, models, args.out)

if __name__ == "__main__":
    main()
