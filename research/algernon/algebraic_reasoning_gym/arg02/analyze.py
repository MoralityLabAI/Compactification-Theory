"""ARG-02 oracle side: export public rows, select dev settings, grade, report.

    export  --run-dir D [--sample N]   public JSONL for the runner + split audit
    select  --run-dir D --stage shots|blocks
    report  --run-dir D --out DIR      grade every part, statistics, probes, tape
    replay  --tape T --check R         regrade a committed answer tape (CPU, no model)

Grading uses `oracle.search` / `oracle.grade` only; futures never leave them.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path

from spiral_ln import algebraic_reasoning_oracle as oracle

from research.algernon.algebraic_reasoning_gym import external
from . import flops, scoring, splits, stats
from .settings import load_config

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config" / "arg02_config.json"
LOCK = HERE / "config" / "checkpoint_lock.json"
RESULT_SCHEMA = "algernon.arg02.receipt.v1"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(obj):
    return json.dumps(obj, sort_keys=True, indent=2) + "\n"


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- export

def _sample(cases, n, key):
    out, count = [], {}
    for c in cases:
        k = key(c)
        if count.get(k, 0) < n:
            out.append(c)
            count[k] = count.get(k, 0) + 1
    return out


def cmd_export(args):
    run = Path(args.run_dir)
    test = splits.test_cases()
    ext_rows = list(external.public_records())
    if ext_rows != [splits.public_record(c) for c in test]:
        raise AssertionError("ARG-02 test rows differ from the ARG-01 public export")
    dev = splits.development_cases()
    if args.sample:
        test = _sample(test, args.sample, lambda c: c.split)
        dev = _sample(dev, args.sample, lambda c: (c.split, c.family))
    splits.write_public(run / "public" / "test.jsonl", test)
    splits.write_public(run / "public" / "development.jsonl", dev)
    write(run / "public" / "shots.json", json.dumps([splits.shot_record(c) for c in splits.shot_cases()], indent=1) + "\n")
    audit = splits.audit()
    write(run / "splits_audit.json", dump(audit))
    manifest = {"sample": args.sample, "files": {p.name: file_sha256(p) for p in sorted((run / "public").iterdir())},
                "audit_sha256": sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()}
    write(run / "export_manifest.json", dump(manifest))
    print("exported", len(test), "test and", len(dev), "development rows to", run / "public")


# ---------------------------------------------------------------- loading

def all_cases():
    cases = {c.identifier: c for c in splits.test_cases()}
    cases.update({c.identifier: c for c in splits.development_cases()})
    return cases


def public_ids(run, split):
    return [json.loads(l)["case_id"] for l in (Path(run) / "public" / f"{split}.jsonl").read_text().splitlines() if l.strip()]


def load_label(run, label, ids, *, allow_partial=False):
    root = Path(run) / "parts" / label
    out, keys = {}, set()
    for cid in ids:
        import re
        p = root / (re.sub(r"[^A-Za-z0-9_.-]", "_", cid) + ".json")
        if not p.exists():
            if allow_partial:
                continue
            raise FileNotFoundError(f"missing part {label}/{cid}")
        rec = load_json(p)
        if rec["id"] != cid:
            raise AssertionError("part id mismatch")
        keys.add(json.dumps(rec["key"], sort_keys=True))
        out[cid] = rec["value"]
    if len(keys) > 1:
        raise AssertionError(f"{label}: parts written under {len(keys)} different keys")
    key = json.loads(next(iter(keys))) if keys else None
    return out, key


def load_states(run, label, cid):
    import re
    import numpy as np
    p = Path(run) / "parts" / label / (re.sub(r"[^A-Za-z0-9_.-]", "_", cid) + ".npz")
    with np.load(p) as z:
        return z["states"]


def op_ids(run, model_key="model"):
    return load_json(Path(run) / f"op_ids_{model_key}.json")


# ---------------------------------------------------------------- selection

def _future12(case, rec, ops):
    return scoring.score_record(case, rec, ops)["at12"]["future_correct"]


def is_smoke(run):
    return load_json(Path(run) / "export_manifest.json").get("sample") is not None


def cmd_select(args):
    run = Path(args.run_dir)
    cfg = load_config(is_smoke(run))
    cases = all_cases()
    ids = public_ids(run, "development")
    ops = op_ids(run)
    if args.stage == "shots":
        table = {}
        for k in cfg["shot_options"]:
            parts, _ = load_label(run, f"dev_shots_k{k}", ids, allow_partial=args.allow_partial)
            table[str(k)] = {"cases": len(parts),
                             "future12": sum(_future12(cases[c], v["records"]["1"], ops) for c, v in parts.items())}
        best = max(cfg["shot_options"], key=lambda k: (table[str(k)]["future12"], -k))
        sel = {"stage": "shots", "shots": best, "table": table,
               "rule": "max development future-correct at 12 checks, depth 1; tie -> fewer shots"}
        write(run / "selection_shots.json", dump(sel))
        print(dump(sel), end="")
        return
    shots_sel = load_json(run / "selection_shots.json")
    table = {}
    for block in cfg["block_candidates"]:
        label = f"dev_block_{block[0]}_{block[1]}"
        parts, _ = load_label(run, label, ids, allow_partial=args.allow_partial)
        row = {}
        for d in cfg["depths"]:
            row[str(d)] = sum(_future12(cases[c], v["records"][str(d)], ops) for c, v in parts.items())
        table[label] = {"cases": len(parts), "future12_by_depth": row}
    candidates = []
    for order, block in enumerate(cfg["block_candidates"]):
        label = f"dev_block_{block[0]}_{block[1]}"
        for d in cfg["depths"]:
            if d > 1:
                candidates.append((table[label]["future12_by_depth"][str(d)], -d, -order, block, d))
    best = max(candidates)
    block, depth = best[3], best[4]
    base = table[f"dev_block_{block[0]}_{block[1]}"]["future12_by_depth"]["1"]
    sel = {"stage": "blocks", "shots": shots_sel["shots"], "block": block, "depth": depth,
           "dev_future12_selected": best[0], "dev_future12_depth1": base, "dev_gain": best[0] - base,
           "table": table, "selection_shots_sha256": file_sha256(run / "selection_shots.json"),
           "rule": "max development future-correct at 12 checks over blocks x depths>1; tie -> smaller depth, earlier block"}
    write(run / "selection.json", dump(sel))
    print(dump(sel), end="")


# ---------------------------------------------------------------- grading

def _row(case):
    return {"family": case.family, "split": case.split, "topology": case.topology,
            "positive": case.split in splits.POSITIVE_SPLITS, "novel": case.family in splits.NOVEL_FAMILIES}


def _compact(g):
    return {"top1": g["top1"], "rank_of_source": g["rank_of_source"], "greedy_valid": g["greedy_valid"],
            "greedy_program": g["greedy"]["program"], "greedy_accepted": g["greedy"]["accepted"],
            "greedy_future": g["greedy"]["future_correct"],
            "at1": {k: g["at1"][k] for k in ("accepted", "future_correct", "false_future", "program")},
            "at12": {k: g["at12"][k] for k in ("accepted", "future_correct", "false_future", "queries",
                                               "integer_loop_proxy", "program", "trace")},
            "verify_ns": g["at12"]["verify_ns"]}


TAPE_DECIMALS = 6


def _rounded(rec):
    """Tape form of a runner record: log-probs rounded to 1e-6 before grading,
    so the committed tape regrades to exactly the committed receipt."""
    r = lambda x: round(float(x), TAPE_DECIMALS)
    return {"lp1": [r(x) for x in rec["lp1"]], "lp2": [[r(x) for x in row] for row in rec["lp2"]],
            "lp3": [[[r(x) for x in col] for col in row] for row in rec["lp3"]],
            "g1": int(rec["g1"]), "g2": [int(x) for x in rec["g2"]], "g3": [[int(x) for x in row] for row in rec["g3"]],
            "row_spread": r(rec.get("row_spread", 0.0))}


def answer_tape(run, ids, sel, cfg):
    """Every model answer record of the test run, keyed (case, arm, step)."""
    tape = _tape(run, ids)
    for item in tape:
        if "record" in item:
            item["record"] = _rounded(item["record"])
    return tape


def has_amended_shuffle(run, ids):
    """Amendment 1: the donor shuffle rerun with corrected alignment (all test rows present)."""
    root = Path(run) / "parts" / "test_shuffle_a1"
    return root.is_dir() and len(list(root.glob("*.json"))) >= len(ids)


def _tape(run, ids):
    tape = []
    loop, _ = load_label(run, "test_loop", ids)
    for cid in ids:
        for t, r in loop[cid]["records"].items():
            tape.append({"case": cid, "arm": "loop", "step": int(t), "record": r})
    labels = [("test_open", "open"), ("test_shuffle", "shuffle"), ("test_perturb", "perturb")]
    if has_amended_shuffle(run, ids):
        labels.append(("test_shuffle_a1", "shuffle_a1"))
    for label, arm in labels:
        parts, _ = load_label(run, label, ids)
        for cid in ids:
            for t, r in parts[cid]["records"].items():
                tape.append({"case": cid, "arm": arm, "step": int(t), "record": r})
    ens, _ = load_label(run, "test_ensemble", ids)
    for cid in ids:
        for v, r in ens[cid]["records"].items():
            tape.append({"case": cid, "arm": "ensemble:" + v, "step": 1, "record": r})
    patch, _ = load_label(run, "test_patch", ids)
    for cid in ids:
        for t, r in patch[cid]["patches"].items():
            tape.append({"case": cid, "arm": "patch", "step": int(t), "record": r["record"],
                         "donor": patch[cid]["donor"]})
    ff, _ = load_label(run, "test_ff", ids)
    for cid in ids:
        tape.append({"case": cid, "arm": "ff2b", "step": 1, "record": ff[cid]["records"]["1"]})
    ctl, _ = load_label(run, "test_controller", ids)
    for cid in ids:
        tape.append({"case": cid, "arm": "controller", "step": len(ctl[cid]["tested"]),
                     "tested": ctl[cid]["tested"]})
    return tape


def grade_tape(tape, cases, ops, cfg, sel):
    """Deterministic grades of a tape. Returns {arm_key: {case: compact grade}}."""
    by = {}
    ens = {}
    for item in tape:
        case = cases[item["case"]]
        arm = item["arm"]
        if arm == "controller":
            g = scoring.score_controller(case, [tuple(p) for p in item["tested"]])
            by.setdefault("controller", {})[item["case"]] = {
                "accepted": g["accepted"], "future_correct": g["future_correct"], "false_future": g["false_future"],
                "queries": g["queries"], "integer_loop_proxy": g["integer_loop_proxy"], "program": g["program"],
                "trace": g["trace"], "verify_ns": g["verify_ns"]}
            continue
        if arm.startswith("ensemble:"):
            ens.setdefault(item["case"], {})[arm.split(":", 1)[1]] = item["record"]
            continue
        key = f"{arm}@{item['step']}"
        by.setdefault(key, {})[item["case"]] = _compact(scoring.score_record(case, item["record"], ops))
    block = sel["block"]
    n = cfg_layers(sel)
    sizes = flops.ensemble_sizes(n, block, cfg["depths"], len(cfg["ensemble_variants"]))
    for cid, recs in ens.items():
        for d, m in sizes.items():
            chosen = [recs[v] for v in cfg["ensemble_variants"][:m]]
            g = scoring.score_scores(cases[cid], scoring.ensemble_scores(chosen), None)
            by.setdefault(f"ensemble@{d}", {})[cid] = _compact(g)
    return by, sizes


def cfg_layers(sel):
    return load_json(LOCK)["model"]["text_config"]["num_hidden_layers"] if not sel.get("smoke") else \
        load_json(LOCK)["smoke"]["text_config"]["num_hidden_layers"]


def write_tape(path, tape):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        for item in tape:
            fh.write((json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode())


def read_tape(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ---------------------------------------------------------------- summaries

def summarise(grades, ids, cases):
    pos = [c for c in ids if cases[c].split in splits.POSITIVE_SPLITS]
    out = {}
    for arm, g in sorted(grades.items()):
        def tot(cids, path):
            s = 0
            for c in cids:
                v = g[c]
                for k in path:
                    v = v[k]
                s += v
            return s
        nested = "at12" in next(iter(g.values()))
        f12 = ("at12", "future_correct") if nested else ("future_correct",)
        a12 = ("at12", "accepted") if nested else ("accepted",)
        rec = {"positives": len(pos), "future12": tot(pos, f12), "accepted12": tot(pos, a12)}
        if nested:
            rec.update({"future1": tot(pos, ("at1", "future_correct")), "accepted1": tot(pos, ("at1", "accepted")),
                        "greedy_valid": tot(pos, ("greedy_valid",)), "greedy_accepted": tot(pos, ("greedy_accepted",)),
                        "greedy_future": tot(pos, ("greedy_future",)), "queries12": tot(pos, ("at12", "queries")),
                        "mean_rank_of_source": sum(g[c]["rank_of_source"] for c in pos) / len(pos)})
            rec["invalid_program_rate"] = 1 - rec["greedy_valid"] / len(pos)
        rec["by_family"] = {f: sum(1 for c in pos if cases[c].family == f and _get(g[c], f12))
                            for f in splits.POSITIVE_SPLITS}
        rec["novel_future12"] = sum(1 for c in pos if cases[c].family in splits.NOVEL_FAMILIES and _get(g[c], f12))
        neg = [c for c in ids if cases[c].split == "negative"]
        rec["negative_accepted"] = sum(_get(g[c], a12) for c in neg)
        col = [c for c in ids if cases[c].split == "collision"]
        rec["collision_false_future"] = sum(_get(g[c], ("at12", "false_future") if nested else ("false_future",)) for c in col)
        pairs = {}
        for c in col:
            pairs.setdefault(cases[c].pair_id, []).append(c)
        tr = ("at12", "trace") if nested else ("trace",)
        rec["collision_pairs_identical_trace"] = sum(1 for p in pairs.values() if len(p) == 2 and
                                                     _get(g[p[0]], tr) == _get(g[p[1]], tr))
        out[arm] = rec
    return out


def _get(v, path):
    for k in path:
        v = v[k]
    return v


def outcome(g, path=("at12", "future_correct")):
    return {c: (_get(v, path) if "at12" in v else v["future_correct"]) for c, v in g.items()}


def compare(grades, a, b, pos, cases, path=("at12", "future_correct")):
    va, vb = outcome(grades[a], path), outcome(grades[b], path)
    cl = [f"{cases[c].family}|{cases[c].topology}" for c in pos]
    return stats.paired([va[c] for c in pos], [vb[c] for c in pos], cl)


def symbolic_grades(sym):
    return {f"symbolic:{arm}": rows for arm, rows in sym.items()}


# ---------------------------------------------------------------- compute

def _macs(tc, length, layers, rows=16):
    return flops.forward_macs(tc, length, rows, layers)


def compute_table(run, ids, sel, cfg, grades, sizes, sym, sym_meta):
    lock = load_json(LOCK)
    entry = lock["smoke"] if sel.get("smoke") else lock["model"]
    ff_entry = lock["smoke"] if sel.get("smoke") else lock["ff_control"]
    tc, tcff = entry["text_config"], ff_entry["text_config"]
    block = tuple(sel["block"])
    loop, _ = load_label(run, "test_loop", ids)
    opn, _ = load_label(run, "test_open", ids)
    shuffles = [("shuffle", load_label(run, "test_shuffle", ids)[0])]
    if has_amended_shuffle(run, ids):
        shuffles.append(("shuffle_a1", load_label(run, "test_shuffle_a1", ids)[0]))
    ens, _ = load_label(run, "test_ensemble", ids)
    ff, _ = load_label(run, "test_ff", ids)
    ctl, _ = load_label(run, "test_controller", ids)
    per = {}

    def add(arm, cid, macs, gpu_ms, tokens, layer_evals, passes, extra_cpu_ns=0):
        g = grades[arm][cid]
        vns = g["verify_ns"] + extra_cpu_ns
        per.setdefault(arm, []).append({"case": cid, "macs": macs, "gpu_ms": gpu_ms, "tokens": tokens,
                                        "layer_evals": layer_evals, "forward_passes": passes,
                                        "verifier_queries": g["at12"]["queries"] if "at12" in g else g["queries"],
                                        "integer_loop_proxy": g["at12"]["integer_loop_proxy"] if "at12" in g else g["integer_loop_proxy"],
                                        "verify_ms": vns / 1e6, "end_to_end_ms": gpu_ms + vns / 1e6})

    a, b = block
    for cid in ids:
        lv = loop[cid]
        L = lv["usage"]["length"]
        tm = lv["usage"]["timing"]
        for d in range(1, max(cfg["depths"]) + 1):
            layers = flops.path(tc, block, d)
            ms = tm["prelude_ms"] + sum(tm["block_ms"][:d]) + tm["readout_ms"][str(d)]
            add(f"loop@{d}", cid, _macs(tc, L, layers), ms, 16 * L, 16 * L * len(layers), 1)
        ov = opn[cid]
        otm = ov["usage"]["timing"]
        for d in [x for x in cfg["depths"] if x > 1]:
            layers = flops.path(tc, block, 1)
            ms = otm["prelude_ms"] + sum(otm["block_ms"][:d]) + otm["readout_ms"][str(d)]
            add(f"open@{d}", cid, _macs(tc, L, layers), ms, 16 * L, 16 * L * len(layers), 1)
        for name, shf in shuffles:
            sv = shf[cid]
            stm = sv["usage"]["timing"]
            Ld = sv["usage"]["donor_tokens"] // 16
            for d in [x for x in cfg["depths"] if x > 1]:
                own = flops.path(tc, block, 1)
                donor_layers = list(range(a)) + list(range(a, b)) * d
                macs = _macs(tc, L, own) + 16 * sum(flops.layer_macs(tc, i, Ld) for i in donor_layers)
                dt = sv["usage"]["donor_timing"]
                ms = stm["prelude_ms"] + stm["block_ms"][0] + stm["readout_ms"][str(d)] + dt["prelude_ms"] + sum(dt["block_ms"][:d])
                add(f"{name}@{d}", cid, macs, ms, 16 * (L + Ld), 16 * (L * len(own) + Ld * len(donor_layers)), 2)
        ev = ens[cid]
        for d, m in sizes.items():
            vs = cfg["ensemble_variants"][:m]
            macs = sum(_macs(tc, ev["usage_by_variant"][v]["length"], flops.path(tc)) for v in vs)
            ms = sum(ev["usage_by_variant"][v]["gpu_ms"] for v in vs)
            tok = sum(16 * ev["usage_by_variant"][v]["length"] for v in vs)
            add(f"ensemble@{d}", cid, macs, ms, tok, sum(ev["usage_by_variant"][v]["layer_evals"] for v in vs), m)
        fv = ff[cid]
        add("ff2b@1", cid, _macs(tcff, fv["usage"]["length"], flops.path(tcff)), fv["usage"]["gpu_ms"],
            16 * fv["usage"]["length"], fv["usage"]["layer_evals"], 1)
        cv = ctl[cid]
        macs = sum(_macs(tc, r["prompt_tokens"] + 2, flops.path(tc)) for r in cv["rounds"])
        tok = sum(16 * (r["prompt_tokens"] + 2) for r in cv["rounds"])
        add("controller", cid, macs, cv["usage"]["gpu_ms"], tok, cv["usage"]["layer_evals"], len(cv["rounds"]),
            extra_cpu_ns=sum(r["check_ns"] for r in cv["rounds"]))
    for arm, rows in sym.items():
        for cid in ids:
            r = rows[cid]
            per.setdefault(f"symbolic:{arm}", []).append({
                "case": cid, "macs": r["model_macs"], "gpu_ms": 0.0, "tokens": 0, "layer_evals": 0,
                "forward_passes": 1 if r["model_macs"] else 0, "verifier_queries": r["queries"],
                "integer_loop_proxy": r["integer_loop_proxy"], "verify_ms": r["verify_ns"] / 1e6,
                "cpu_order_ms": r["order_ns"] / 1e6, "end_to_end_ms": (r["order_ns"] + r["verify_ns"]) / 1e6})
    summary = {}
    for arm, rows in per.items():
        e2e = [r["end_to_end_ms"] for r in rows]
        summary[arm] = {"cases": len(rows), "total_macs": sum(r["macs"] for r in rows),
                        "total_pflop": 2 * sum(r["macs"] for r in rows) / 1e15,
                        "gpu_seconds": sum(r["gpu_ms"] for r in rows) / 1e3,
                        "verifier_cpu_seconds": sum(r["verify_ms"] + r.get("cpu_order_ms", 0.0) for r in rows) / 1e3,
                        "end_to_end_seconds": sum(e2e) / 1e3,
                        "tokens": sum(r["tokens"] for r in rows), "layer_evals": sum(r["layer_evals"] for r in rows),
                        "forward_passes": sum(r["forward_passes"] for r in rows),
                        "verifier_queries": sum(r["verifier_queries"] for r in rows),
                        "integer_loop_proxy": sum(r["integer_loop_proxy"] for r in rows),
                        "latency_ms_p50": stats.quantile(e2e, 0.5), "latency_ms_p95": stats.quantile(e2e, 0.95)}
    summary["_shared"] = {"symbolic_checkpoint_load_ms": sym_meta["checkpoint_load_ns"] / 1e6,
                          "symbolic_class_compile_ms": sym_meta["class_compile_ns"] / 1e6}
    return summary, per


def peaks(run, ids):
    out = {}
    labels = ["test_loop", "test_perturb", "test_open", "test_shuffle", "test_patch", "test_ensemble",
              "test_controller", "test_ff"] + (["test_shuffle_a1"] if has_amended_shuffle(run, ids) else [])
    for label in labels:
        parts, _ = load_label(run, label, ids)
        vals = [v.get("peak_bytes") for v in parts.values() if v.get("peak_bytes") is not None]
        out[label] = max(vals) if vals else None
    return out


# ---------------------------------------------------------------- intermediate states

def depth_analysis(grades, ids, cases, cfg, run):
    pos = [c for c in ids if cases[c].split in splits.POSITIVE_SPLITS]
    T = max(cfg["depths"])
    curve = {}
    for t in range(1, T + 1):
        g = grades[f"loop@{t}"]
        curve[str(t)] = {"future12": sum(g[c]["at12"]["future_correct"] for c in pos),
                         "future1": sum(g[c]["at1"]["future_correct"] for c in pos),
                         "accepted1": sum(g[c]["at1"]["accepted"] for c in pos),
                         "greedy_valid": sum(g[c]["greedy_valid"] for c in pos),
                         "greedy_accepted": sum(g[c]["greedy_accepted"] for c in pos)}
    first = {}
    for c in pos:
        hit = next((t for t in range(1, T + 1) if grades[f"loop@{t}"][c]["at1"]["future_correct"]), None)
        first[c] = hit
    hist = {}
    for v in first.values():
        hist[str(v)] = hist.get(str(v), 0) + 1
    later = [c for c, v in first.items() if v is not None and v > 1]
    damage = {}
    for d in [x for x in cfg["depths"] if x > 1]:
        for k, path in (("at1", ("at1", "future_correct")), ("at12", ("at12", "future_correct"))):
            g1, gd = grades["loop@1"], grades[f"loop@{d}"]
            damage[f"{k}@{d}"] = {"lost": sum(1 for c in pos if _get(g1[c], path) and not _get(gd[c], path)),
                                  "gained": sum(1 for c in pos if _get(gd[c], path) and not _get(g1[c], path))}
    loop, _ = load_label(run, "test_loop", ids)
    res = [[loop[c]["residual"][t] for c in ids] for t in range(T)]
    residual = {str(t + 1): {"median": stats.quantile(res[t], 0.5), "p95": stats.quantile(res[t], 0.95)} for t in range(T)}
    reduction = {str(t + 1): stats.quantile([loop[c]["residual"][t] / max(loop[c]["residual"][t - 1], 1e-12) for c in ids], 0.5)
                 for t in range(1, T)}
    return {"curve": curve, "first_success_step_hist": hist, "first_success_after_step1": len(later),
            "first_success_after_step1_kept_at_T": sum(1 for c in later if grades[f"loop@{T}"][c]["at1"]["future_correct"]),
            "damage": damage, "residual_relative_norm": residual, "residual_reduction_median": reduction}


def perturb_analysis(grades, ids, cases, cfg, run):
    pos = [c for c in ids if cases[c].split in splits.POSITIVE_SPLITS]
    pert, _ = load_label(run, "test_perturb", ids)
    out = {"sigma": cfg["perturb"]["sigma"], "step": cfg["perturb"]["step"], "readouts": {}}
    for t in cfg["perturb"]["readout_steps"]:
        gp, gc = grades[f"perturb@{t}"], grades[f"loop@{t}"]
        out["readouts"][str(t)] = {"future12": sum(gp[c]["at12"]["future_correct"] for c in pos),
                                   "clean_future12": sum(gc[c]["at12"]["future_correct"] for c in pos),
                                   "top1_agrees_with_clean": sum(gp[c]["top1"] == gc[c]["top1"] for c in pos)}
    T = cfg["perturb"]["depth"]
    rel = [[pert[c]["distance"][t] / max(pert[c]["kick_norm"], 1e-12) for c in ids] for t in range(T)]
    out["distance_over_kick_median"] = {str(t + 1): stats.quantile(rel[t], 0.5) for t in range(T)}
    return out


def _grammar_dist(rec):
    s = scoring.joint_scores(rec)
    m = max(s)
    z = sum(math.exp(x - m) for x in s)
    return [math.exp(x - m) / z for x in s]


def patch_analysis(run, ids, cases, grades, sel, tape_index):
    pos = [c for c in ids if cases[c].split in splits.POSITIVE_SPLITS]
    patch, _ = load_label(run, "test_patch", ids)
    depth = patch[ids[0]]["depth"]
    out = {"depth": depth, "steps": {}}
    for t in patch[ids[0]]["patches"]:
        n_own = n_donor = n_changed = 0
        kls = []
        for c in pos:
            donor = patch[c]["donor"]
            g = grades[f"patch@{t}"][c]
            own = grades[f"loop@{depth}"][c]["top1"]
            don = grades[f"loop@{depth}"][donor]["top1"]
            n_own += g["top1"] == own
            n_donor += g["top1"] == don and don != own
            n_changed += g["top1"] != own
            p = _grammar_dist(tape_index[("patch", t, c)])
            q = _grammar_dist(tape_index[("loop", str(depth), c)])
            kls.append(sum(x * (math.log(x) - math.log(y)) for x, y in zip(p, q) if x > 0))
        out["steps"][t] = {"cases": len(pos), "top1_kept": n_own, "top1_to_donor": n_donor, "top1_changed": n_changed,
                           "kl_median": stats.quantile(kls, 0.5), "kl_p95": stats.quantile(kls, 0.95),
                           "future12": sum(grades[f"patch@{t}"][c]["at12"]["future_correct"] for c in pos)}
    return out


def probe_analysis(run, ids, cases, sel, cfg):
    import numpy as np
    from research.rrc_compactification import learn18
    from . import probe
    classes_map = learn18.class_map()

    def labels(case):
        rep = classes_map[case.topology]["mapping"][tuple(case.source_graph)]
        return {"outer_op": rep[2], "uses_R": "R" in rep}

    dev_ids = public_ids(run, "development")
    block = sel["block"]
    dlabel = f"dev_block_{block[0]}_{block[1]}"
    train = [c for c in dev_ids if not c.startswith("calibration:")]
    select_ids = [c for c in dev_ids if c.startswith("calibration:")]
    pos = [c for c in ids if cases[c].split in splits.POSITIVE_SPLITS]
    seen = [c for c in pos if cases[c].family in splits.SEEN_FAMILIES]
    novel = [c for c in pos if cases[c].family in splits.NOVEL_FAMILIES]
    S = {c: load_states(run, dlabel, c)[:, 0, :] for c in train + select_ids}
    S.update({c: load_states(run, "test_loop", c)[:, 0, :] for c in pos})
    steps = sorted(set(range(0, max(cfg["depths"]) + 1)) & set(range(S[train[0]].shape[0])) &
                   set(range(S[pos[0]].shape[0])))
    out = {"train": len(train), "select": len(select_ids), "test_seen": len(seen), "test_novel": len(novel),
           "site": "block output at the last prompt token, step 0 = block input", "labels": {}}
    for name, classes in (("outer_op", list(oracle.OPS)), ("uses_R", [False, True])):
        per = {}
        for t in steps:
            X = lambda cs: np.stack([S[c][t] for c in cs])
            Y = lambda cs: [labels(cases[c])[name] for c in cs]
            per[str(t)] = probe.evaluate(X(train), Y(train), X(select_ids), Y(select_ids),
                                         {"seen": (X(seen), Y(seen)), "novel": (X(novel), Y(novel))}, classes)
        out["labels"][name] = per
    return out


def external_crosscheck(grades, ids, cases, revision, tmp):
    """Top-1 loop answers through ARG-01's external.py scorer (depths 1, 2, 4, 8)."""
    rows = []
    depths = [d for d in external.DEPTHS if f"loop@{d}" in grades]
    for d in depths:
        for c in ids:
            rows.append({"case_id": c, "public_sha256": cases[c].signature, "depth": d,
                         "program": grades[f"loop@{d}"][c]["top1"], "model_revision": revision})
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    ext = external.score(tmp)
    agree = {}
    for d in depths:
        mine = sum(grades[f"loop@{d}"][c]["at1"]["accepted"] for c in ids)
        agree[str(d)] = {"arg01_exact_accepted": ext["scores"][str(d)]["exact_accepted"], "arg02_at1_accepted": mine,
                         "arg01_future_correct": ext["scores"][str(d)]["future_correct"],
                         "arg02_at1_future": sum(grades[f"loop@{d}"][c]["at1"]["future_correct"] for c in ids)}
        if agree[str(d)]["arg01_exact_accepted"] != mine:
            raise AssertionError("ARG-01 external scorer disagrees with ARG-02 grading")
    return {"proposal_sha256": ext["proposal_sha256"], "by_depth": agree}


# ---------------------------------------------------------------- decisions

def decisions(cmp, compute, summary, sel, sizes):
    d = sel["depth"]
    ok = lambda r: r["difference"] > 0 and r["cluster_ci95"][0] > 0
    eff = lambda arm: summary[arm]["future12"] / max(compute[arm]["end_to_end_seconds"], 1e-12)
    res = {
        "H1_depth": ok(cmp[f"loop@{d}_vs_loop@1"]) and cmp[f"loop@{d}_vs_loop@1"]["mcnemar_p"] < 0.05,
        "H2a_feedback": ok(cmp[f"loop@{d}_vs_open@{d}"]),
        # Amendment 1: H2b is evaluated on the corrected shuffle when it exists; the
        # original arm's outcome is kept alongside.
        "H2b_state": ok(cmp.get(f"loop@{d}_vs_shuffle_a1@{d}", cmp[f"loop@{d}_vs_shuffle@{d}"])),
        "H2b_state_original_shuffle": ok(cmp[f"loop@{d}_vs_shuffle@{d}"]),
        "H3a_repeat": ok(cmp[f"loop@{d}_vs_ensemble@{d}"]),
        "H3b_ff": ok(cmp[f"loop@{d}_vs_ff2b@1"]),
        "H3c_efficiency": eff(f"loop@{d}") > eff("loop@1") and eff(f"loop@{d}") > eff(f"ensemble@{d}"),
        "H4_symbolic": summary[f"loop@{d}"]["future12"] >= summary["symbolic:certified_mass_full"]["future12"] and
                       eff(f"loop@{d}") > eff("symbolic:certified_mass_full"),
    }
    res["mechanism_claim"] = all(res[k] for k in ("H1_depth", "H2a_feedback", "H2b_state", "H3a_repeat", "H3b_ff"))
    res["efficiency_future12_per_e2e_second"] = {arm: eff(arm) for arm in
                                                 (f"loop@{d}", "loop@1", f"ensemble@{d}", "ff2b@1", "controller",
                                                  "symbolic:certified_mass_full", "symbolic:certified_mass12",
                                                  "symbolic:exhaustive_64", "symbolic:frozen_gru")}
    res["selected"] = {"block": sel["block"], "depth": d, "shots": sel["shots"], "ensemble_size": sizes[d]}
    return res


def cmd_report(args):
    run = Path(args.run_dir)
    smoke = is_smoke(run)
    cfg = load_config(smoke)
    sel = load_json(run / "selection.json")
    sel["smoke"] = smoke
    cases = all_cases()
    ids = public_ids(run, "test")
    ops = op_ids(run)
    tape = answer_tape(run, ids, sel, cfg)
    grades, sizes = grade_tape(tape, cases, ops, cfg, sel)
    sym, sym_meta = __import__("research.algernon.algebraic_reasoning_gym.arg02.symbolic", fromlist=["run"]).run(
        [cases[c] for c in ids])
    grades.update(symbolic_grades(sym))
    summary = summarise(grades, ids, cases)
    pos = [c for c in ids if cases[c].split in splits.POSITIVE_SPLITS]
    d = sel["depth"]
    pairs = [(f"loop@{x}", "loop@1") for x in cfg["depths"] if x > 1]
    pairs += [(f"loop@{d}", f"open@{d}"), (f"loop@{d}", f"shuffle@{d}"), (f"loop@{d}", f"ensemble@{d}"),
              (f"loop@{d}", "ff2b@1"), ("controller", "loop@1"), (f"loop@{d}", "symbolic:certified_mass12"),
              (f"loop@{d}", "symbolic:certified_mass_full"), (f"loop@{d}", "symbolic:exhaustive_64"),
              ("loop@1", "symbolic:frozen_gru"), ("loop@1", "symbolic:frozen_ff"),
              ("loop@1", "symbolic:certified_mass12"), ("ff2b@1", "loop@1")]
    pairs += [(f"open@{x}", "loop@1") for x in cfg["depths"] if x > 1]
    pairs += [(f"shuffle@{x}", "loop@1") for x in cfg["depths"] if x > 1]
    if f"shuffle_a1@{d}" in grades:
        pairs += [(f"loop@{d}", f"shuffle_a1@{d}")]
        pairs += [(f"shuffle_a1@{x}", "loop@1") for x in cfg["depths"] if x > 1]
    cmp = {f"{a}_vs_{b}": compare(grades, a, b, pos, cases) for a, b in pairs}
    cmp1 = {f"{a}_vs_{b}": compare(grades, a, b, pos, cases, ("at1", "future_correct"))
            for a, b in pairs if not (a.startswith("symbolic") or b.startswith("symbolic") or a == "controller")}
    compute, per_case_compute = compute_table(run, ids, sel, cfg, grades, sizes, sym, sym_meta)
    tape_index = {}
    for item in tape:
        if "record" in item:
            tape_index[(item["arm"], str(item["step"]), item["case"])] = item["record"]
    lock = load_json(LOCK)
    entry = lock["smoke"] if smoke else lock["model"]
    receipt = {
        "schema": RESULT_SCHEMA, "smoke": smoke, "op_ids": ops,
        "model": {k: entry[k] for k in ("repo", "revision", "dtype") if k in entry},
        "ff_control": {k: (lock["smoke"] if smoke else lock["ff_control"])[k] for k in ("repo", "revision")},
        "selection": {k: sel[k] for k in ("shots", "block", "depth", "dev_gain", "dev_future12_selected",
                                          "dev_future12_depth1") if k in sel},
        "ensemble_sizes": {str(k): v for k, v in sizes.items()},
        "splits": splits.audit(),
        "evalmath_pin": load_json(HERE.parent / "evalmath_pin.json"),
        "arg01_receipt_sha256": file_sha256(HERE.parent / "results" / "arg01_reference.json"),
        "summary": summary, "paired_future12": cmp, "paired_future1": cmp1,
        "depth": depth_analysis(grades, ids, cases, cfg, run),
        "perturbation": perturb_analysis(grades, ids, cases, cfg, run),
        "patching": patch_analysis(run, ids, cases, grades, sel, tape_index),
        "probes": probe_analysis(run, ids, cases, sel, cfg),
        "external_crosscheck": external_crosscheck(grades, ids, cases, entry["revision"], run / "arg01_external.jsonl"),
    }
    profile = {"schema": "algernon.arg02.profile.v1", "compute": compute, "peak_allocated_bytes": peaks(run, ids),
               "manifests": {p.name: load_json(p) for p in sorted(run.glob("manifest_*.json"))},
               "timing_scope": "CUDA-event timings per segment; a depth-d latency is prelude + d block steps + its readout, "
                               "measured inside the combined trajectory; verifier time is this machine's CPU wall time at grading"}
    receipt["decisions"] = decisions(cmp, compute, summary, sel, sizes)
    out = Path(args.out)
    write_tape(out / "arg02_answer_tape.jsonl.gz", tape)
    receipt["answer_tape_sha256"] = file_sha256(out / "arg02_answer_tape.jsonl.gz")
    write(out / "arg02_receipt.json", dump(receipt))
    write(out / "arg02_profile.json", dump(profile))
    with gzip.GzipFile(out / "arg02_traces.jsonl.gz", "wb", mtime=0) as fh:
        for arm in sorted(grades):
            for cid in ids:
                row = {"arm": arm, "case_id": cid, **_row(cases[cid]), "grade": grades[arm][cid]}
                fh.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    write(out / "arg02_selection.json", dump({k: v for k, v in sel.items() if k != "smoke"}))
    print("wrote", out)


def cmd_replay(args):
    """Regrade the committed tape on CPU and compare the grade-derived receipt fields."""
    receipt = load_json(args.check)
    tape = read_tape(args.tape)
    if file_sha256(args.tape) != receipt["answer_tape_sha256"]:
        raise SystemExit("answer tape differs from receipt")
    cfg = load_config(receipt["smoke"])
    sel = dict(receipt["selection"])
    sel["smoke"] = receipt["smoke"]
    cases = all_cases()
    ids = sorted({t["case"] for t in tape}, key=[c.identifier for c in splits.test_cases()].index)
    ops = load_json(Path(args.op_ids)) if args.op_ids else receipt.get("op_ids")
    grades, _ = grade_tape(tape, cases, ops, cfg, sel)
    from . import symbolic
    sym, _ = symbolic.run([cases[c] for c in ids])
    grades.update(symbolic_grades(sym))
    summary = summarise(grades, ids, cases)
    if summary != receipt["summary"]:
        raise SystemExit("regraded summary differs from receipt")
    print("ARG-02 answer tape regraded: summary matches receipt for", len(summary), "arms")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["export", "select", "report", "replay"])
    p.add_argument("--run-dir")
    p.add_argument("--stage", choices=["shots", "blocks"])
    p.add_argument("--sample", type=int)
    p.add_argument("--out")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--tape")
    p.add_argument("--check")
    p.add_argument("--op-ids")
    args = p.parse_args(argv)
    {"export": cmd_export, "select": cmd_select, "report": cmd_report, "replay": cmd_replay}[args.command](args)


if __name__ == "__main__":
    main()
