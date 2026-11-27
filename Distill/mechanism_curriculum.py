"""Synthetic oracle-verified DataGoblin curriculum from discovered campaign mechanisms.

Generates disjoint train, calibration, heldout_roots, and heldout_structural splits
for three exact-rational decision mechanisms identified during DISTILL-09/Fable audits:
  1. receipt_quotient       : receipt-flag & budget-cap state equivalence soundness
  2. belief_quantization    : decision-boundary alignment vs Brier-regret decoupling
  3. overhead_regime_switch : dense-feature overhead vs sparse analytic regime switching

Strictly synthetic_oracle_only: zero provider text, zero private/sealed data,
training_admission=False pending native DataGoblin owning-pipeline admission.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path
from .core import canonical, digest, require, write_json
from .instances import Rng

SCHEMA = "distill.mechanism_curriculum.v1"
SPLITS = (
    ("train", 2026113001, 24, "standard"),
    ("calibration", 2026113002, 9, "standard"),
    ("heldout_roots", 2026113003, 18, "standard"),
    ("heldout_structural", 2026113004, 18, "structural_shift"),
)


def _frac(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def _ri(rng: Rng, lo: int, hi: int) -> int:
    return lo + rng.below(hi - lo + 1)


def _gen_receipt_quotient(rng: Rng, ordinal: int, regime: str) -> tuple[dict, dict]:
    """Synthetic state-pair quotient soundness task with receipt-flag & budget-cap logic."""
    cert_r = _ri(rng, 8, 12) if regime == "standard" else _ri(rng, 11, 16)
    rec_r = _ri(rng, 4, 7) if regime == "standard" else _ri(rng, 5, 8)
    wrong_r = -_ri(rng, 10, 14)
    c_direct = _ri(rng, 3, 4)
    c_trans = 1 if regime == "standard" else 2
    p_num = _ri(rng, 1, 5)
    p_den = p_num + _ri(rng, 1, 5)
    p1 = Fraction(p_num, p_den)

    # Construct two states at stage 1 (2 remaining in-scope projects + 0 out-of-scope)
    b_a = _ri(rng, 2, 6)
    b_b = _ri(rng, 2, 6)
    has_anchor_a = bool(rng.below(2))
    has_anchor_b = bool(rng.below(2))
    has_law_a = bool(rng.below(2))
    has_law_b = bool(rng.below(2))
    # Force non-trivial receipt mismatch on half of instances
    if ordinal % 2 == 0:
        has_anchor_a, has_law_a = True, True
        has_anchor_b, has_law_b = False, True
        b_a = max(b_a, c_trans * 2)
        b_b = b_a

    def opt_val(budget: int, anchor_rcpt: bool, law_rcpt: bool) -> Fraction:
        # Exact single-project continuation under known truth p1 and legal actions:
        # defer (0), recommend best (max(p1*rec_r + (1-p1)*wrong_r, (1-p1)*rec_r + p1*wrong_r)),
        # direct_audit+certify (cert_r - c_direct if budget >= c_direct),
        # transport+certify (cert_r - c_trans if anchor_rcpt and law_rcpt and budget >= c_trans).
        best = Fraction(0)
        ev1 = p1 * rec_r + (1 - p1) * wrong_r
        ev0 = (1 - p1) * rec_r + p1 * wrong_r
        best = max(best, ev1, ev0)
        if budget >= c_direct:
            best = max(best, Fraction(cert_r - c_direct))
        if anchor_rcpt and law_rcpt and budget >= c_trans:
            best = max(best, Fraction(cert_r - c_trans))
        return best

    va = opt_val(b_a, has_anchor_a, has_law_a)
    vb = opt_val(b_b, has_anchor_b, has_law_b)
    gap = abs(va - vb)
    sound_merge = (gap == 0)

    # Cheap baselines:
    # 1. posterior_only_merge: merges whenever posteriors match (always true here)
    # 2. receipt_and_cap_merge: merges iff receipt eligibility matches and min(b, c_direct) matches
    cap = c_direct
    heur_merge = ((has_anchor_a and has_law_a) == (has_anchor_b and has_law_b)) and (
        min(b_a, cap) == min(b_b, cap)
    )
    # Penalty = gap if unsound merge chosen, else 0 if sound or separated (with small 1/20 missed-merge cost)
    miss_cost = Fraction(1, 20)

    def action_regret(choice_merge: bool) -> Fraction:
        best_cost = Fraction(0)
        chosen_cost = gap if choice_merge else (miss_cost if sound_merge else Fraction(0))
        opt_cost = Fraction(0) if sound_merge else Fraction(0)
        return chosen_cost - opt_cost

    pub = {
        "mechanism": "receipt_quotient",
        "regime": regime,
        "ordinal": ordinal,
        "payoffs": {"cert_reward": cert_r, "rec_reward": rec_r, "wrong_reward": wrong_r},
        "costs": {"direct_audit": c_direct, "transport": c_trans},
        "posterior_p1": _frac(p1),
        "state_a": {"budget": b_a, "anchor_receipt": has_anchor_a, "law_receipt": has_law_a},
        "state_b": {"budget": b_b, "anchor_receipt": has_anchor_b, "law_receipt": has_law_b},
        "legal_actions": ["merge", "separate"],
    }
    ref = {
        "value_a": _frac(va),
        "value_b": _frac(vb),
        "soundness_gap": _frac(gap),
        "optimal_action": "merge" if sound_merge else "separate",
        "action_regrets": {
            "merge": _frac(action_regret(True)),
            "separate": _frac(action_regret(False)),
        },
        "baselines": {
            "stop_separate": _frac(action_regret(False)),
            "naive_posterior_merge": _frac(action_regret(True)),
            "cheap_analytic_cap_receipt": _frac(action_regret(heur_merge)),
        },
    }
    return pub, ref


def _gen_belief_quantization(rng: Rng, ordinal: int, regime: str) -> tuple[dict, dict]:
    """Synthetic exact-Fraction 2-region belief quantization showing Brier vs Bayes-regret."""
    # Asymmetric decision threshold tau = L_fp / (L_fp + L_fn)
    l_fn = _ri(rng, 1, 4) if regime == "standard" else _ri(rng, 2, 6)
    l_fp = _ri(rng, 2, 7) if regime == "standard" else _ri(rng, 5, 11)
    tau = Fraction(l_fp, l_fp + l_fn)

    # 4 discrete belief atoms with integer weights
    atoms = []
    for _ in range(4):
        num = _ri(rng, 1, 19)
        w = _ri(rng, 1, 6)
        atoms.append({"p": Fraction(num, 20), "w": w})
    atoms.sort(key=lambda x: x["p"])
    tot_w = sum(a["w"] for a in atoms)

    # Candidate 2-bin partitions of the 4 sorted atoms: split after index k in {1, 2, 3}
    def eval_split(k: int) -> tuple[Fraction, Fraction]:
        bins = [atoms[:k], atoms[k:]]
        brier_excess = Fraction(0)
        regret = Fraction(0)
        for b in bins:
            bw = sum(x["w"] for x in b)
            mean_p = sum(x["w"] * x["p"] for x in b) / bw
            bin_act = 1 if mean_p >= tau else 0
            for x in b:
                prob = Fraction(x["w"], tot_w)
                brier_excess += prob * (x["p"] - mean_p) ** 2
                # Optimal Bayes loss vs bin_act loss
                # action 1 loss = (1-p)*l_fp; action 0 loss = p*l_fn
                l1 = (1 - x["p"]) * l_fp
                l0 = x["p"] * l_fn
                opt_l = min(l0, l1)
                act_l = l1 if bin_act == 1 else l0
                regret += prob * (act_l - opt_l)
        return brier_excess, regret

    evals = {f"split_{k}": eval_split(k) for k in (1, 2, 3)}
    # Best regret split (tie-break by lowest Brier)
    opt_act = min(evals, key=lambda k: (evals[k][1], evals[k][0], k))
    brier_act = min(evals, key=lambda k: (evals[k][0], k))
    mid_act = "split_2"

    opt_reg = evals[opt_act][1]
    pub = {
        "mechanism": "belief_quantization",
        "regime": regime,
        "ordinal": ordinal,
        "loss_fn": l_fn,
        "loss_fp": l_fp,
        "decision_threshold": _frac(tau),
        "belief_atoms": [{"p": _frac(a["p"]), "weight": a["w"]} for a in atoms],
        "legal_actions": ["split_1", "split_2", "split_3"],
    }
    ref = {
        "optimal_action": opt_act,
        "brier_minimizing_action": brier_act,
        "brier_regret_decoupled": evals[brier_act][1] > opt_reg,
        "action_metrics": {
            k: {"brier_excess": _frac(v[0]), "decision_regret": _frac(v[1])}
            for k, v in evals.items()
        },
        "action_regrets": {k: _frac(v[1] - opt_reg) for k, v in evals.items()},
        "baselines": {
            "stop_midpoint": _frac(evals[mid_act][1] - opt_reg),
            "naive_brier_minimizer": _frac(evals[brier_act][1] - opt_reg),
            "cheap_analytic_threshold_split": _frac(
                evals[
                    min(
                        ("split_1", "split_2", "split_3"),
                        key=lambda k: (
                            sum(
                                1
                                for i, a in enumerate(atoms)
                                if (a["p"] >= tau) != (i >= int(k.split("_")[1]))
                            ),
                            k,
                        ),
                    )
                ][1]
                - opt_reg
            ),
        },
    }
    return pub, ref


def _gen_overhead_regime_switch(rng: Rng, ordinal: int, regime: str) -> tuple[dict, dict]:
    """Synthetic exact-Fraction overhead-aware regime switcher (MTR-02I mechanism)."""
    budget = _ri(rng, 4, 10) if regime == "standard" else _ri(rng, 3, 14)
    # Two cheap analytic routes (Route_A = optimistic/low-budget, Route_B = route/high-budget)
    # plus a Dense_Learned route that pays overhead_frac of budget for feature scoring
    a_slope = Fraction(_ri(rng, 3, 6), 2)
    a_cap = Fraction(_ri(rng, 7, 12), 1)
    b_thresh = _ri(rng, 4, 7)
    b_slope = Fraction(_ri(rng, 5, 9), 2)
    dense_overhead = Fraction(_ri(rng, 20, 32), 100) if regime == "standard" else Fraction(_ri(rng, 10, 38), 100)
    dense_bonus = Fraction(_ri(rng, 0, 4), 1) if regime == "standard" else Fraction(_ri(rng, 0, 7), 1)

    def u_opt(b: Fraction) -> Fraction:
        return min(a_cap, a_slope * b)

    def u_route(b: Fraction) -> Fraction:
        return max(Fraction(0), b_slope * (b - b_thresh))

    b_eff = Fraction(budget) * (1 - dense_overhead)
    val_opt = u_opt(Fraction(budget))
    val_route = u_route(Fraction(budget))
    val_dense = max(u_opt(b_eff), u_route(b_eff)) + dense_bonus
    val_stop = Fraction(0)

    vals = {
        "stop": val_stop,
        "analytic_optimistic": val_opt,
        "analytic_route": val_route,
        "dense_learned": val_dense,
    }
    best_act = max(vals, key=lambda k: (vals[k], k))
    best_val = vals[best_act]

    # Cheap budget-conditioned analytic baseline (picks best of optimistic vs route at zero overhead)
    cheap_act = "analytic_route" if val_route >= val_opt else "analytic_optimistic"

    pub = {
        "mechanism": "overhead_regime_switch",
        "regime": regime,
        "ordinal": ordinal,
        "budget": budget,
        "optimistic_params": {"slope": _frac(a_slope), "cap": _frac(a_cap)},
        "route_params": {"threshold": b_thresh, "slope": _frac(b_slope)},
        "dense_scorer": {"overhead_fraction": _frac(dense_overhead), "gross_bonus": _frac(dense_bonus)},
        "legal_actions": ["stop", "analytic_optimistic", "analytic_route", "dense_learned"],
    }
    ref = {
        "optimal_action": best_act,
        "optimal_value": _frac(best_val),
        "action_values": {k: _frac(v) for k, v in vals.items()},
        "action_regrets": {k: _frac(best_val - v) for k, v in vals.items()},
        "baselines": {
            "stop": _frac(best_val - val_stop),
            "always_dense_learned": _frac(best_val - val_dense),
            "always_route": _frac(best_val - val_route),
            "cheap_analytic_budget_switch": _frac(best_val - vals[cheap_act]),
        },
    }
    return pub, ref


GENERATORS = (
    _gen_receipt_quotient,
    _gen_belief_quantization,
    _gen_overhead_regime_switch,
)


def export(out_dir: str | Path) -> dict:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    split_summaries = []

    for split_name, seed, count, regime in SPLITS:
        rng = Rng(["distill.mechanism_curriculum.v1", split_name, seed])
        pub_rows = []
        ref_rows = []
        per_mech = {"receipt_quotient": 0, "belief_quantization": 0, "overhead_regime_switch": 0}
        baseline_sums: dict[str, Fraction] = {}
        baseline_ties: dict[str, int] = {}

        idx = 0
        attempts = 0
        while idx < count:
            attempts += 1
            require(attempts <= count * 20, "hash collision loop exceeded bound")
            gen = GENERATORS[idx % len(GENERATORS)]
            pub, ref = gen(rng, idx, regime)
            chash = digest(pub)
            if chash in seen_hashes:
                continue
            seen_hashes.add(chash)
            pub_row = {"row_id": chash, "split": split_name, **pub}
            ref_row = {"row_id": chash, "split": split_name, **ref}
            pub_rows.append(pub_row)
            ref_rows.append(ref_row)
            per_mech[pub["mechanism"]] += 1
            for bname, rstr in ref["baselines"].items():
                key = f"{pub['mechanism']}:{bname}"
                rf = Fraction(rstr)
                baseline_sums[key] = baseline_sums.get(key, Fraction(0)) + rf
                if rf == 0:
                    baseline_ties[key] = baseline_ties.get(key, 0) + 1
            idx += 1

        pub_path = root / f"{split_name}.public.jsonl"
        ref_path = root / f"{split_name}.referee.jsonl"
        pub_bytes = b"".join(canonical(r) + b"\n" for r in pub_rows)
        ref_bytes = b"".join(canonical(r) + b"\n" for r in ref_rows)
        pub_path.write_bytes(pub_bytes)
        ref_path.write_bytes(ref_bytes)

        b_summary = {}
        for k, total in sorted(baseline_sums.items()):
            mech = k.split(":")[0]
            n_m = per_mech[mech]
            b_summary[k] = {
                "mean_regret": _frac(total / n_m),
                "oracle_ties": baseline_ties.get(k, 0),
                "n": n_m,
            }

        split_summaries.append({
            "split": split_name,
            "seed": seed,
            "regime": regime,
            "rows": len(pub_rows),
            "per_mechanism": per_mech,
            "public_sha256": digest(pub_rows),
            "referee_sha256": digest(ref_rows),
            "baseline_regrets": b_summary,
        })

    manifest = {
        "schema_version": SCHEMA,
        "source_kind": "synthetic_oracle_only",
        "provider_outputs_admitted": False,
        "training_admission": False,
        "scope": "public development synthetic mechanism curriculum; disjoint hashes; requires owning-pipeline admission",
        "total_rows": len(seen_hashes),
        "splits": split_summaries,
    }
    manifest["manifest_id"] = digest(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(json.dumps(export(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
