"""Summarise adapter_bits.py runs as a markdown report next to their results.jsonl."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def first_bits(curve: dict, target: float) -> int | None:
    """Fewest bits whose error is within `target`; `curve` maps str(bits) -> error."""
    for bits in sorted(curve, key=int):
        if curve[bits] <= target:
            return int(bits)
    return None


def fit_slope(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of log y on log x."""
    pts = [(math.log(x), math.log(y)) for x, y in points if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    var = sum((x - mx) ** 2 for x, _ in pts)
    return None if var == 0 else sum((x - mx) * (y - my) for x, y in pts) / var


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="directory holding results.jsonl")
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    results = [r for r in records if r.get("event") == "result"]
    lines = [f"# adapter_bits report: {args.run.name}", ""]

    lines += ["## Runs", "",
              "W = trainable adapter parameters, M = items (random label bits). I = M(1 - H2(err)) is the Fano estimate "
              "of stored label information. Knee = fewest bits per weight keeping error within 1 point of full precision "
              "(on the gradient sample for the predicted columns).", "",
              "| adapter | size | lr | W | M | M/W | err | I/W (bits/param) | knee | knee, typical-noise model | knee, worst-case (l-inf K) | median worst-case bits/item | K_inf | K_2 | tangent rank (1e-2) init / trained | top Hessian eig | counting bound held |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        sample_err = r["err_sample"]
        quant_sample = {b: q["err_sample"] for b, q in r["quantized"].items()}
        knee_sample = first_bits(quant_sample, sample_err + 0.01)
        knee_typ = first_bits(r["err_pred_typical"], sample_err + 0.01)
        knee_worst = first_bits(r["err_bound_worst"], sample_err + 0.01)
        held = all(q["counting_bound_ok"] for q in r["quantized"].values())
        worst_median = r.get("worst_case_bits_quantiles", {}).get("0.5")
        lines.append(
            f"| {r['adapter']} | {r['size']} | {r['lr']} | {r['params']} | {r['items']} | {r['label_bits_per_param']} | "
            f"{r['err']} | {r['bits_per_param']} | {r['knee_bits']} (sample {knee_sample}) | {knee_typ} | {knee_worst} | "
            f"{worst_median} | {r['K_inf_max']} | {r['K_2_max']} | "
            f"{r.get('tangent_init', {}).get('rank_rel_0.01')} / {r.get('tangent_trained', {}).get('rank_rel_0.01')} | "
            f"{r['hessian_top']} | {'yes' if held else '**no**'} |")

    lines += ["", "## Information per parameter after rounding", "",
              "Each cell is I_b / W in bits per parameter. The counting bound caps it at b (plus 32 bits per tensor for the scale).", "",
              "| adapter | size | lr | M | " + " | ".join(f"b={b}" for b in results[0]["quantized"]) + " |" if results else "",
              "|---|---|---|---|" + "---|" * (len(results[0]["quantized"]) if results else 0)]
    for r in results:
        lines.append(f"| {r['adapter']} | {r['size']} | {r['lr']} | {r['items']} | "
                     + " | ".join(str(q["bits_per_param"]) for q in r["quantized"].values()) + " |")

    lines += ["", "## Stored bits against the gradient matrix's rank", "",
              "Cover's theorem caps a linearized adapter near twice the rank of its items-by-parameters gradient "
              "matrix. Rank is counted at two relative thresholds on a sample of items, so it cannot exceed the "
              "sample size.", "",
              "| adapter | size | lr | M | I (bits) | sample rows | 2 x rank at 1e-2 | 2 x rank at 1e-3 | participation ratio |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        t = r.get("tangent_trained") or {}
        if not t:
            continue
        lines.append(f"| {r['adapter']} | {r['size']} | {r['lr']} | {r['items']} | {int(r['info_bits'])} | {t['rows']} | "
                     f"{2 * t['rank_rel_0.01']} | {2 * t['rank_rel_0.001']} | {t['participation_ratio']} |")

    lines += ["", "## Capacity scaling", ""]
    groups: dict[tuple, list] = {}
    for r in results:
        groups.setdefault((r["adapter"], r["lr"]), []).append(r)
    for (adapter, lr), runs in groups.items():
        by_size: dict[int, float] = {}
        for r in runs:
            by_size[r["params"]] = max(by_size.get(r["params"], 0.0), r["info_bits"])
        slope = fit_slope(sorted(by_size.items()))
        lines.append(f"- {adapter} (lr {lr}): best stored bits by W = "
                     + ", ".join(f"{w}: {int(i)}" for w, i in sorted(by_size.items()))
                     + (f"; log-log slope {slope:.2f}" if slope is not None else ""))
    out = args.run / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
