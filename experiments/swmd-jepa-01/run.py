"""SWMD-JEPA-01: preregistered four-arm, paired-view latent-state pilot.

Run from any working directory: python experiments/swmd-jepa-01/run.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from data import ACTIONS, Vocabulary, digest, load_jsonl, mask_view, split_worlds, synthetic
from model import Experiment, pad, vicreg

ARMS = ("next_token", "bad_view", "good_view", "good_view_transition")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_texts(model, vocab, texts, device, batch_size=128):
    model.eval()
    zs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            ids, lengths = pad([vocab.encode(t) for t in texts[start:start + batch_size]], device)
            zs.append(model.encoder(ids, lengths).cpu())
    return torch.cat(zs)


def train_arm(arm, train, vocab, actions, args, seed, device):
    seed_all(seed)
    model = Experiment(len(vocab), len(actions), args.dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(seed + 1729)
    loss_history = []
    tokens_seen = 0
    for _epoch in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), args.batch):
            rows = [train[i] for i in order[start:start + args.batch]]
            if len(rows) < 4:  # variance/covariance need nondegenerate batches
                continue
            optimizer.zero_grad(set_to_none=True)
            if arm == "next_token":
                texts = [r[k] for r in rows for k in ("view_a", "view_b")]
                ids, lengths = pad([vocab.encode(t) for t in texts], device)
                h = model.encoder(ids, lengths, sequence=True)
                logits = model.lm_head(h[:, :-1])
                target = ids[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, len(vocab)), target.reshape(-1), ignore_index=0)
                tokens_seen += int((target != 0).sum().item())
            else:
                if arm == "bad_view":
                    originals = [r[rng.choice(("view_a", "view_b"))] for r in rows]
                    a = [mask_view(t, rng) for t in originals]
                    b = [mask_view(t, rng) for t in originals]
                else:
                    a = [r["view_a"] for r in rows]
                    b = [r["view_b"] for r in rows]
                ida, lena = pad([vocab.encode(t) for t in a], device)
                idb, lenb = pad([vocab.encode(t) for t in b], device)
                za = model.encoder(ida, lena)
                zb = model.encoder(idb, lenb)
                loss = vicreg(za, zb)
                tokens_seen += int((lena + lenb - 2).sum().item())
                if arm == "good_view_transition":
                    nxt, ln = pad([vocab.encode(r["next_view_a"]) for r in rows], device)
                    with torch.no_grad():
                        z_next = model.encoder(nxt, ln)
                    action_ids = torch.tensor([actions.index(r["action"]) for r in rows], device=device)
                    loss = loss + args.dynamics_weight * F.mse_loss(
                        model.next_latent(za, action_ids), z_next.detach())
                    tokens_seen += int((ln - 1).sum().item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batches += 1
        if batches == 0:
            raise ValueError("training split smaller than one batch of four")
        loss_history.append(round(epoch_loss / batches, 6))
    return model, loss_history, tokens_seen


def ridge(x, y, alpha=1.):
    """Deterministic CPU float64 linear readout; never trained on test rows."""
    x = x.double()
    y = y.double()
    x = torch.cat((x, torch.ones((x.shape[0], 1), dtype=torch.float64)), dim=1)
    gram = x.T @ x + alpha * torch.eye(x.shape[1], dtype=torch.float64)
    gram[-1, -1] -= alpha  # intercept is unpenalized
    return torch.linalg.solve(gram, x.T @ y)


def project(x, w):
    x = x.double()
    return torch.cat((x, torch.ones((x.shape[0], 1), dtype=torch.float64)), dim=1) @ w


def transition_features(z, action_ids, n_actions):
    oh = F.one_hot(torch.tensor(action_ids), n_actions).double()
    z = z.double()
    return torch.cat((z, oh, (z[:, None, :] * oh[:, :, None]).flatten(1)), dim=1)


def compute_metrics(model, train, test, vocab, actions, device, arm):
    needed = ("view_a", "view_b", "next_view_a", "next_view_b")
    train_z = {k: encode_texts(model, vocab, [r[k] for r in train], device) for k in needed}
    test_z = {k: encode_texts(model, vocab, [r[k] for r in test], device) for k in needed}
    a, b = test_z["view_a"], test_z["view_b"]
    sim = F.normalize(a, dim=1) @ F.normalize(b, dim=1).T
    best = sim.argmax(1).tolist()
    hits = [int(test[i]["state_id"] == test[j]["state_id"]) for i, j in enumerate(best)]
    retrieval = sum(hits) / len(hits)
    embeddings = torch.cat((a, b), dim=0).double()
    centered = embeddings - embeddings.mean(0)
    spectrum = torch.linalg.svdvals(centered)
    effective_rank = float((spectrum.sum() ** 2 / spectrum.square().sum().clamp_min(1e-12)).item())
    state_train = {r["state_id"] for r in train}
    unseen = [i for i, r in enumerate(test) if r["state_id"] not in state_train]
    result = {
        "retrieval_r1": retrieval,
        "retrieval_unseen_r1": (sum(hits[i] for i in unseen) / len(unseen)) if unseen else None,
        "retrieval_chance": sum(sum(r["state_id"] == q["state_id"] for q in test)
                                for r in test) / len(test) ** 2,
        "unique_test_states": len({r["state_id"] for r in test}),
        "unseen_state_fraction": sum(r["state_id"] not in state_train for r in test) / len(test),
        "effective_rank": effective_rank,
        "state_accuracy_macro": None,
    }
    common_attrs = (set.intersection(*(set(r.get("attributes", {})) for r in train + test))
                    if train and test else set())
    if common_attrs:
        scores = {}
        xa = torch.cat((train_z["view_a"], train_z["view_b"]), dim=0)
        for attr in sorted(common_attrs):
            classes = sorted({str(r["attributes"][attr]) for r in train})
            if len(classes) < 2:
                continue
            targets = torch.tensor([classes.index(str(r["attributes"][attr])) for r in train])
            oh = F.one_hot(torch.cat((targets, targets)), len(classes))
            weight = ridge(xa, oh)
            y_hat = project(embeddings, weight).argmax(1).tolist()
            y_true = [str(r["attributes"][attr]) for r in test] * 2
            scores[attr] = sum(classes[p] == y for p, y in zip(y_hat, y_true)) / len(y_true)
        if scores:
            result["state_accuracy_by_attr"] = scores
            result["state_accuracy_macro"] = sum(scores.values()) / len(scores)
    train_action = [actions.index(r["action"]) for r in train]
    test_action = [actions.index(r["action"]) for r in test]
    xt = transition_features(torch.cat((train_z["view_a"], train_z["view_b"])),
                             train_action * 2, len(actions))
    z_train_next = (train_z["next_view_a"] + train_z["next_view_b"]) / 2
    yt = torch.cat((z_train_next, z_train_next), dim=0)
    map_w = ridge(xt, yt)
    xe = transition_features(torch.cat((a, b)), test_action * 2, len(actions))
    pred = project(xe, map_w)
    z_test_next = ((test_z["next_view_a"] + test_z["next_view_b"]) / 2).double()
    target = torch.cat((z_test_next, z_test_next), dim=0)
    variance = float((z_test_next - z_test_next.mean(0)).square().mean().clamp_min(1e-8))
    result["closure_probe_nmse"] = float((pred - target).square().mean()) / variance
    if unseen:
        selected = unseen + [i + len(test) for i in unseen]
        result["closure_probe_nmse_unseen"] = float((pred[selected] - target[selected]).square().mean()) / variance
    else:
        result["closure_probe_nmse_unseen"] = None
    result["view_disagreement_nmse"] = float((a - b).square().mean()) / variance
    result["transition_view_gap_nmse"] = float((pred[:len(test)] - pred[len(test):]).square().mean()) / variance
    if arm == "good_view_transition":
        model.eval()
        with torch.no_grad():
            aa = torch.tensor(test_action, device=device)
            p1 = model.next_latent(a.to(device), aa).cpu().double()
            p2 = model.next_latent(b.to(device), aa).cpu().double()
        result["trained_transition_nmse"] = float((torch.cat((p1, p2)) - target).square().mean()) / variance
    return result


def bootstrap(values, seed=2026, n=2000):
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return {"mean": sum(values) / len(values), "ci95": [means[int(.025*n)], means[int(.975*n)]]}


def summarize(receipts, seeds):
    by = {(r["seed"], r["arm"]): r["metrics"] for r in receipts}
    output = {"seeds": seeds, "note": "seed-bootstrap CIs are exploratory with few seeds", "arms": {}, "paired_deltas": {}}
    metrics = ("retrieval_r1", "state_accuracy_macro", "closure_probe_nmse", "view_disagreement_nmse")
    for arm in ARMS:
        output["arms"][arm] = {}
        for metric in metrics:
            vals = [by[(s, arm)].get(metric) for s in seeds]
            if all(v is not None for v in vals):
                output["arms"][arm][metric] = bootstrap(vals)
    for name, lhs, rhs, metric in (
        ("good_minus_bad_retrieval", "good_view", "bad_view", "retrieval_r1"),
        ("good_minus_next_token_retrieval", "good_view", "next_token", "retrieval_r1"),
        ("good_minus_bad_state", "good_view", "bad_view", "state_accuracy_macro"),
        ("transition_minus_good_closure", "good_view_transition", "good_view", "closure_probe_nmse"),
    ):
        a = [by[(s, lhs)].get(metric) for s in seeds]
        b = [by[(s, rhs)].get(metric) for s in seeds]
        if all(v is not None for v in a + b):
            output["paired_deltas"][name] = bootstrap([x-y for x,y in zip(a,b)])
    return output


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", type=Path, help="paired-view transitions; default: synthetic")
    ap.add_argument("--worlds", type=int, default=96)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--data-seed", type=int, default=2026)
    ap.add_argument("--seeds", default="11,29,47")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--lr", type=float, default=.001)
    ap.add_argument("--dynamics-weight", type=float, default=5.)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "pilot")
    args = ap.parse_args(argv)
    if args.epochs < 1 or args.batch < 4 or args.dim < 2 or args.threads < 1:
        ap.error("epochs >= 1, batch >= 4, dim >= 2, threads >= 1 required")
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    if not seeds or len(set(seeds)) != len(seeds) or not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        ap.error("nonempty unique seeds and valid unique arms required")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("requested CUDA but no CUDA device available")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    rows = load_jsonl(args.input_jsonl) if args.input_jsonl else synthetic(args.data_seed, args.worlds, args.steps)
    splits = split_worlds(rows, args.data_seed)
    actions = sorted({r["action"] for r in rows})
    vocab = Vocabulary(splits["train"])
    code_hash = hashlib.sha256("".join(Path(__file__).with_name(p).read_text(encoding="utf8")
                                   for p in ("data.py", "model.py", "run.py")).encode()).hexdigest()
    metadata = {"source": str(args.input_jsonl) if args.input_jsonl else "synthetic-v1",
                "code_sha256": code_hash,
                "sha256": digest(rows), "seed": args.data_seed,
                "world_counts": {k: len({r["world_id"] for r in v}) for k,v in splits.items()},
                "row_counts": {k: len(v) for k,v in splits.items()},
                "vocab_size": len(vocab), "actions": actions,
                "torch": torch.__version__, "python": sys.version.split()[0], "device": str(device)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps({"data": metadata, "config": vars(args)}, indent=2, default=str) + "\n")
    receipts = []
    for seed in seeds:
        for arm in arms:
            model, losses, exposure = train_arm(arm, splits["train"], vocab, actions, args, seed, device)
            metrics = compute_metrics(model, splits["train"], splits["test"], vocab, actions, device, arm)
            receipt = {"arm": arm, "seed": seed, "data_sha256": metadata["sha256"], "code_sha256": code_hash,
                       "epochs": args.epochs, "loss_history": losses, "tokens_seen": exposure,
                       "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
                       "metrics": metrics}
            receipts.append(receipt)
            (args.out / f"seed-{seed}-{arm}.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            print(f"{arm:22s} seed={seed:<4d} retrieval={metrics['retrieval_r1']:.3f} "
                  f"state={metrics['state_accuracy_macro']} closure_nmse={metrics['closure_probe_nmse']:.3f}", flush=True)
    if set(ARMS) == set(arms):
        (args.out / "summary.json").write_text(json.dumps(summarize(receipts, seeds), indent=2, allow_nan=False) + "\n")
    return receipts


if __name__ == "__main__":
    main()
