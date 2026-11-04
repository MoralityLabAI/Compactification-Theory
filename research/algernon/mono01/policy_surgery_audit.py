"""MONO-01 Phase B: algebra of the released Policy Surgery edits.

Run inside the policy-surgery environment (pinned commit 46f95e6):
  uv run --frozen --extra weights python policy_surgery_audit.py OUT_DIR
Measures, per reconstructed edit and per augmented layer [W | b]: algebraic rank of the delta, nonzero support,
changed float32 entries, bf16-effective changed entries, and the relative error that bf16 storage adds to the delta.
Also rebuilds the reversed-thruster action swap as the exact rank-one update (I - d d^T) [W3 | b3], d = e1 - e3.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monoshot as M
from policy_surgery import campaign
from policy_surgery.analysis import SOURCE, read
from policy_surgery.io import ROOT
from policy_surgery.policy import Policy, apply_ops, fetch_actor

LAYERS = {"layer1": ("w1", "b1"), "layer2": ("w2", "b2"), "output": ("w3", "b3")}
ACCEPTED = ["crosswind-llm-completion-seed0", "crosswind-llm-reward-seed1", "crosswind-llm-reward-seed2",
            "weak_main-llm-completion-seed1"]


def aug(p: Policy, layer: str) -> np.ndarray:
    w, b = LAYERS[layer]
    return np.concatenate([p.tensors[w], p.tensors[b][:, None]], axis=1).astype(np.float64)


def layer_stats(base: Policy, edited: Policy) -> dict:
    out = {}
    for layer in LAYERS:
        W0, W1 = aug(base, layer), aug(edited, layer)
        D = W1 - W0
        nz = int(np.count_nonzero(D))
        s = np.linalg.svd(D, compute_uv=False) if nz else np.zeros(1)
        rank = int(np.sum(s > 1e-6 * max(s[0], 1e-30))) if nz else 0
        b0, b1 = M.to_bf16(W0.astype(np.float32)), M.to_bf16(W1.astype(np.float32))
        eff = int(np.count_nonzero(b1 != b0))
        dn = np.linalg.norm(D)
        rel = float(np.linalg.norm((b1.astype(np.float64) - b0.astype(np.float64)) - D) / dn) if dn else 0.0
        out[layer] = {"entries": D.size, "nonzero": nz, "rank": rank, "bf16_changed": eff, "bf16_rel_delta_error": rel}
    return out


def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base, meta = fetch_actor(campaign.CACHE)

    # 1) reversed thrusters: exact rank-one swap of the left (1) and right (3) engine logits, bias included
    d = np.zeros(4); d[1], d[3] = 1.0, -1.0
    W3 = aug(base, "output")
    swapped = (np.eye(4) - np.outer(d, d)) @ W3
    ref = apply_ops(base, [{"op": "output_affine", "permutation": [0, 3, 2, 1]}])
    delta = swapped - W3
    rev = {
        "rank_of_delta": int(np.linalg.matrix_rank(delta)),
        "matches_upstream_permutation_float64": bool(np.array_equal(swapped.astype(np.float32), aug(ref, "output").astype(np.float32))),
        "rows_changed": int(np.count_nonzero(np.abs(delta).sum(1))),
        "nonzero_entries": int(np.count_nonzero(delta)),
        "entries_in_augmented_output_layer": int(delta.size),
        "nonzero_in_bf16_effective": layer_stats(base, ref)["output"]["bf16_changed"],
    }

    # 2) every exactly reconstructed candidate + the four accepted policies
    rows, skipped = [], []
    for path in sorted((ROOT / "results").glob("*/search/*/candidates.json")):
        for c in read(path):
            if "actor_sha256" not in c:
                continue
            m = apply_ops(base, c["ops"])
            if m.digest() != c["actor_sha256"]:
                skipped.append({"run": path.parent.name, "id": c["id"]})
                continue
            st = layer_stats(base, m)
            rows.append({"run": path.parent.name, "id": c["id"],
                         "total_nonzero": sum(v["nonzero"] for v in st.values()),
                         "total_bf16_changed": sum(v["bf16_changed"] for v in st.values()),
                         "max_layer_rank": max(v["rank"] for v in st.values()),
                         **{f"{l}_{k}": v[k] for l, v in st.items() for k in ("nonzero", "rank", "bf16_changed", "bf16_rel_delta_error")}})
    accepted = {}
    sel = {r["run"]: r["candidate"] for r in read(ROOT / "results/control-objective-comparison-gate-v1/selection.json")["runs"]}
    for name in ACCEPTED:
        want = sel[name]
        cand = next(c for c in read(SOURCE / "search" / name / "candidates.json") if c["id"] == want["id"])
        m = apply_ops(base, cand["ops"])
        assert m.digest() == want["actor_sha256"] == cand["actor_sha256"]
        accepted[name] = {"candidate_id": want["id"], "ops": [o["op"] for o in cand["ops"]], "layers": layer_stats(base, m),
                          "changed_float32_entries": base.changed(m)}
    with open(out / "all_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {"upstream_commit": "46f95e64dc9057e6d025b7141b607230905c04ee", "base_actor_sha256": meta["actor_sha256"],
               "reversal_rank_one": rev, "accepted_policies": accepted,
               "reconstructed_candidates": len(rows), "hash_mismatch_candidates": skipped,
               "rank_histogram_max_layer": {str(r): sum(1 for x in rows if x["max_layer_rank"] == r) for r in sorted({x["max_layer_rank"] for x in rows})}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("reversal_rank_one", "reconstructed_candidates", "rank_histogram_max_layer")}, indent=1))
    for n, a in accepted.items():
        print(n, a["ops"], {l: (v["rank"], v["nonzero"], v["bf16_changed"], round(v["bf16_rel_delta_error"], 4)) for l, v in a["layers"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
