"""Build SRB-REV-02 SFT rows (initial + update stage, count + bitmap codecs).

Stdlib only. Messages are rendered with the bench's own SYSTEM prompt and
user-message serialisation (adapters/revision_llamacpp.py) and every packet is
checked with the bench's validate_revision_packet. Targets are the Bayes oracle
reply (examples/finance_revision_fixture_adapter.py) in the schema property
order the llama.cpp adapter constrains decoding to: posterior, memory, action.
"""
import argparse, hashlib, json, os, sys

DEFAULT_ROOT = r"C:\Users\patri\Documents\Codex\2026-11-16\star\outputs\SystemicRiskBench"


def load_bench(srb_root):
    srb_root = os.path.abspath(srb_root)
    if srb_root not in sys.path:
        sys.path.insert(0, srb_root)
    ex = os.path.join(srb_root, "examples")
    if ex not in sys.path:
        sys.path.append(ex)
    from systemic_risk_bench import agent_revision as ar, finance_revision as fr
    from adapters import revision_llamacpp as rl
    import finance_revision_fixture_adapter as oracle
    return ar, fr, rl, oracle


def used_roots(srb_root):
    """Every root any committed config/result run could have touched."""
    p = json.load(open(os.path.join(srb_root, "configs", "finance-revision-panels.json")))
    used = set()
    for panel in p["panels"].values():
        used |= set(panel["indices"])
    used |= set(range(18))  # legacy fixture runs: cases=18 -> roots 0..17
    return used


def fmt_posterior(post, digits):
    vals = [float(x) for x in post]
    if digits <= 0:
        return vals
    r = [round(v, digits) for v in vals]
    big = max(range(3), key=lambda i: r[i])
    r[big] = round(1.0 - sum(r[i] for i in range(3) if i != big), digits)
    return r


def reply_for(packet, oracle, fr, digits):
    exact = oracle.reply(packet)
    post = fmt_posterior(exact["posterior"], digits)
    out = {"posterior": post}
    if "memory" in exact:
        out["memory"] = exact["memory"]
    out["action"] = exact["action"]
    # rounded posterior must keep the oracle's action
    assert fr.optimal_action(post, packet["finance_context"]["family"]) == exact["action"]
    return exact, out


def rows_for_root(index, ar, fr, rl, oracle, split, digits):
    world = ar.fixture(index)
    family = fr.FAMILIES[index % 3]
    rows = []
    for mode in ar.MODES:
        p1 = fr.packet(world, mode, family, "initial")
        rl.validate_revision_packet(p1)
        ex1, rep1 = reply_for(p1, oracle, fr, digits)
        p2 = fr.packet(world, mode, family, "update", rep1["memory"])  # oracle's own memory
        rl.validate_revision_packet(p2)
        ex2, rep2 = reply_for(p2, oracle, fr, digits)
        ref1 = ar.initial_posterior(world.old_count)
        ref2 = world.exact_count if mode == "count" else world.exact_full
        for phase, pk, rep, ref in (("initial", p1, rep1, ref1), ("update", p2, rep2, ref2)):
            rows.append({
                "id": f"srb-{index:03d}-{mode}-{phase}", "family": "srb", "split": split,
                "messages": [
                    {"role": "system", "content": rl.SYSTEM},
                    {"role": "user", "content": json.dumps({"packet": pk}, sort_keys=True, allow_nan=False)},
                ],
                "target": json.dumps(rep, separators=(",", ":"), allow_nan=False),
                "gold": {
                    "root": index, "scenario": family, "mode": mode, "phase": phase,
                    "reference_posterior": [float(x) for x in ref],
                    "optimal_action": fr.optimal_action(ref, family),
                },
            })
    return rows


def pick_roots(lo, hi, used, n=None, per_family=None):
    pool = [i for i in range(lo, hi) if i not in used]
    if per_family is not None:
        out = []
        for f in range(3):
            out += [i for i in pool if i % 3 == f][:per_family]
        return sorted(out)
    return pool[:n]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srb-root", default=DEFAULT_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-range", default="48:200", help="lo:hi root indices (hi exclusive)")
    ap.add_argument("--val-range", default="200:224")
    ap.add_argument("--eval-range", default="224:256")
    ap.add_argument("--train-roots", type=int, default=113, help="x4 rows per root")
    ap.add_argument("--val-roots", type=int, default=9)
    ap.add_argument("--eval-per-family", type=int, default=5)
    ap.add_argument("--posterior-digits", type=int, default=6, help="0 = full float repr")
    a = ap.parse_args()
    ar, fr, rl, oracle = load_bench(a.srb_root)
    used = used_roots(a.srb_root)
    rng = lambda s: tuple(int(x) for x in s.split(":"))
    plan = {
        "train": pick_roots(*rng(a.train_range), used, n=a.train_roots),
        "val": pick_roots(*rng(a.val_range), used, n=a.val_roots),
        "eval": pick_roots(*rng(a.eval_range), used, per_family=a.eval_per_family),
    }
    seen = set()
    for s, r in plan.items():
        assert not (set(r) & seen) and not (set(r) & used), s
        seen |= set(r)
    os.makedirs(a.out, exist_ok=True)
    manifest = {"schema": "srb_data.v1", "posterior_digits": a.posterior_digits,
                "modes": list(ar.MODES), "phases": ["initial", "update"],
                "excluded_used_roots": sorted(used), "splits": {}}
    for s, roots in plan.items():
        rows = [r for i in roots for r in rows_for_root(i, ar, fr, rl, oracle, s, a.posterior_digits)]
        path = os.path.join(a.out, s + ".jsonl")
        write_jsonl(path, rows)
        manifest["splits"][s] = {
            "rows": len(rows), "roots": roots, "n_roots": len(roots),
            "range_arg": getattr(a, s + "_range"), "sha256": sha256_file(path),
            "rows_per_scenario": {f: sum(r["gold"]["scenario"] == f for r in rows) for f in fr.FAMILIES},
        }
    with open(os.path.join(a.out, "manifest.json"), "w", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps({s: {k: v[k] for k in ("rows", "n_roots", "sha256")} for s, v in manifest["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
