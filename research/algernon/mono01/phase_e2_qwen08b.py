"""MONO-01 Phase E2: rebuild the published Qwen3.5-0.8B sparse edits (RikiyaT/lm_understand-logs exp8_math) and audit their algebra.

Our own reimplementation of the rebuild rule (not upstream's code): read manifest.json and edit.npz as data, apply the
sparse or low-rank edit to the pinned bf16 base tensor, and check the base and edited sha256 recorded in the manifest.
Usage: python phase_e2_qwen08b.py BASE_SAFETENSORS EXP8_DIR OUT_JSON
Base: Qwen/Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monoshot as M


def bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def f32_to_bits(x: np.ndarray) -> np.ndarray:
    return (M.to_bf16(x).view(np.uint32) >> 16).astype(np.uint16)


def sha(b: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()


def rebuild(base_bits, t, z, i):
    if t["form"] == "lowrank":
        w = f32(base_bits).astype(np.float64)
        a, b = z[f"t{i}_a"], z[f"t{i}_b"]
        for k in range(a.shape[1]):
            w = w + np.outer(a[:, k], b[k])
        e = f32_to_bits(w.astype(np.float32)).reshape(-1)
    else:
        e = base_bits.copy().reshape(-1)
    e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
    return e.reshape(base_bits.shape)


def audit(W0: np.ndarray, W1: np.ndarray) -> dict:
    D = W1.astype(np.float64) - W0.astype(np.float64)
    cols = np.nonzero(np.abs(D).sum(0))[0]
    s = np.linalg.svd(D[:, cols], compute_uv=False) if len(cols) else np.zeros(1)
    rank = int(np.sum(s > 1e-3 * s[0])) if len(cols) else 0
    out = {"entries": int(D.size), "nonzero": int(np.count_nonzero(D)), "changed_columns": int(len(cols)),
           "rank_tol1e-3": rank, "singular_values_top5": [float(x) for x in s[:5]],
           "energy_in_top1": float(s[0] ** 2 / np.sum(s ** 2)) if len(cols) else 0.0,
           "relative_change_fro": float(np.linalg.norm(D) / np.linalg.norm(W0))}
    if len(cols):
        U, S_, Vt = np.linalg.svd(D[:, cols], full_matrices=False)
        u = U[:, 0]
        coef = u @ W0[:, cols].astype(np.float64)               # u^T W_S
        # fit D_S ~ -c u (u^T W_S): c = -<D_S, u coef^T> / ||coef||^2 ... a single scalar
        outer = np.outer(u, coef)
        c = -float(np.sum(D[:, cols] * outer) / np.sum(outer * outer))
        out["fit_D_eq_minus_c_u_uTW_on_support"] = {"c": c, "relative_residual": float(np.linalg.norm(D[:, cols] + c * outer) / np.linalg.norm(D[:, cols]))}
        out["_u"] = u
    return out


def main(base_path, exp8, out_json):
    exp8 = Path(exp8)
    res, us = {}, {}
    with safe_open(base_path, framework="pt") as f:
        for run in ("p10c", "p25c", "p50c"):
            man = json.loads((exp8 / "runs" / run / "edit" / "manifest.json").read_text())
            z = np.load(exp8 / "runs" / run / "edit" / "edit.npz", allow_pickle=False)
            res[run] = {"tensors": []}
            for i, t in enumerate(man["tensors"]):
                base = bits(f.get_tensor(t["name"]))
                assert sha(base) == t["base_sha256"], f"{run} {t['name']}: base differs"
                e = rebuild(base, t, z, i)
                assert sha(e) == t["edited_sha256"], f"{run} {t['name']}: edited hash mismatch"
                a = audit(f32(base), f32(e))
                us[(run, t["name"])] = a.pop("_u", None)
                res[run]["tensors"].append({"name": t["name"], "form": t["form"], "shape": t["shape"], "hash_verified": True,
                                            "manifest_changed_entries": t["changed_entries"], **a})
            res[run]["changed_entries_total"] = man["changed_entries_total"]
    names = [k for k in us if k[0] == "p50c" and us[k] is not None]
    if len(names) == 2:
        c = abs(float(us[names[0]] @ us[names[1]]))
        res["p50c_cosine_between_layer_directions"] = c
    Path(out_json).write_text(json.dumps(res, indent=1))
    for run, r in res.items():
        if not isinstance(r, dict):
            print(run, r); continue
        for t in r["tensors"]:
            print(run, t["name"].split("language_model.")[-1], t["form"], "nonzero", t["nonzero"], "cols", t["changed_columns"],
                  "rank", t["rank_tol1e-3"], "top1_energy", round(t["energy_in_top1"], 4), "fit", t.get("fit_D_eq_minus_c_u_uTW_on_support"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
