"""Extract, measure finite directions, fit CPU observers, intervene and report."""
import argparse
import hashlib
import json
import subprocess
import time
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from preflight import ROOT, RESULTS, save, sha
from core import gym, LAYERS, RANKS, haar, dark_frame, route, normed, stable, seed, write_csv, fingerprint


def extract(runtime, tokenizer):
    rows = gym(tokenizer)
    path = RESULTS/"dataset.json"
    if path.exists() and json.loads(path.read_text()) != rows:
        raise RuntimeError("Existing dataset differs")
    save(path, rows)
    progress_path = RESULTS/"extraction_progress.json"
    existing = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if existing and existing["identity"] != runtime.identity:
        raise RuntimeError("Extraction source/settings changed; choose a new results directory")
    manifest_path = RESULTS/"activation_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("complete"):
            for name,field in (("activations.npy","activations_sha256"),("margins.npy","margins_sha256")):
                if sha(RESULTS/name) != previous[field]:
                    raise RuntimeError(f"Cached {name} hash mismatch")
    done = set(existing.get("completed", []))
    mode = "r+" if (RESULTS/"activations.npy").exists() else "w+"
    acts = np.lib.format.open_memmap(RESULTS/"activations.npy", mode=mode, dtype=np.float16,
                                    shape=(len(rows),len(LAYERS),1024))
    margins = np.lib.format.open_memmap(RESULTS/"margins.npy", mode=mode, dtype=np.float32, shape=(len(rows),))
    for i,row in enumerate(rows):
        if i in done:
            continue
        receipt, vectors = runtime.forward(row,capture=LAYERS)
        for li,layer in enumerate(LAYERS):
            acts[i,li] = vectors[layer]
        margins[i] = receipt["margin_a_minus_b"]
        acts.flush()
        margins.flush()
        done.add(i)
        save(progress_path,dict(identity=runtime.identity,completed=sorted(done)))
        if i % 16 == 15:
            runtime.batch(f"extract_{i+1}")
    save(RESULTS/"activation_manifest.json",dict(identity=runtime.identity, shape=list(acts.shape),
         dtype="float16", selected_token_position=-1,layers_one_based=list(LAYERS),
         dataset_sha256=sha(path), activations_sha256=sha(RESULTS/"activations.npy"),
         margins_sha256=sha(RESULTS/"margins.npy"), complete=len(done)==len(rows)))
    return rows,acts,margins


def directional(runtime,row,direction,baseline):
    estimates, receipts = [], []
    for epsilon in (0.25,0.5):
        plus,_ = runtime.forward(row,epsilon*direction)
        minus,_ = runtime.forward(row,-epsilon*direction)
        estimates.append((plus["margin_a_minus_b"]-minus["margin_a_minus_b"])/(2*epsilon))
        receipts.append(dict(epsilon=epsilon, plus_key=plus["key"],minus_key=minus["key"],
                             plus_actual_l2=plus["actual_l2"],minus_actual_l2=minus["actual_l2"]))
    return estimates, stable(*estimates), receipts


def sketches(runtime,rows,acts,margins):
    path = RESULTS/"sensitivity_sketches.json"
    train = [i for i,r in enumerate(rows) if r["split"]=="train"]
    x = acts[train,1].astype(float)
    _,_,vh = np.linalg.svd(x-x.mean(0),full_matrices=False)
    pca = vh[:8].T
    random = haar(1024,8,("sketch",runtime.identity["revision"]))
    random -= pca @ (pca.T@random)
    random,_ = np.linalg.qr(random,mode="reduced")
    directions = np.column_stack([pca,random])
    np.save(RESULTS/"sensitivity_directions.npy", directions)
    selected = train[:8]+[i for i,r in enumerate(rows) if r["split"]=="calibration"][:8]
    output = []
    if path.exists():
        output = json.loads(path.read_text())
    done = {r["state_id"] for r in output}
    for i in selected:
        row = rows[i]
        if row["state_id"] in done:
            continue
        g, tests = np.zeros(1024), []
        for j in range(16):
            estimates, accepted, receipts = directional(runtime,row,directions[:,j],margins[i])
            if accepted:
                g += estimates[1]*directions[:,j]
            tests.append(dict(direction=j, estimates=estimates, accepted=accepted, receipts=receipts))
        output.append(dict(state_id=row["state_id"],split=row["split"],layer=12,
                           gradient=g.tolist(),directions=tests,accepted_count=sum(t["accepted"] for t in tests),
                           scope="projection_onto_measured_16_direction_span"))
        save(path,output)
        runtime.batch(f"sketch_{row['state_id']}")
    return output


def intervene(runtime,rows,acts,margins,observers):
    output = []
    path = RESULTS/"causal_interventions.csv"
    for i,row in enumerate(rows):
        if row["split"] != "heldout":
            continue
        sign = 1 if row["answer"] == 0 else -1
        donor = next(j for j,r in enumerate(rows) if r["split"]=="train"
                     and r["rule"]==row["rule"] and r["answer"]!=row["answer"])
        h, z = acts[i,1].astype(float), acts[donor,1].astype(float)-acts[i,1].astype(float)
        for name in ("causal","task_only","random","pca"):
            obs = observers[f"12_7_{name}"]
            u = obs["u"]
            v = dark_frame(u,("dark",runtime.identity["revision"],row["state_id"],name))
            c = np.random.default_rng(seed("matched",row["state_id"])).standard_normal(7)
            for arm in ("dark_to_observed","observed_to_dark","observed_to_observed"):
                delta = route(u,v,z,arm,0.5)
                receipt,_ = runtime.forward(row,delta)
                random_delta = normed((v if arm=="observed_to_dark" else u)@c,0.5)
                random_receipt,_ = runtime.forward(row,random_delta)
                estimates,accepted,fd_receipts = directional(runtime,row,delta/0.5,margins[i])
                actual = np.asarray(receipt["actual_displacement"])
                actual_r = np.asarray(random_receipt["actual_displacement"])
                effect = sign*(receipt["margin_a_minus_b"]-margins[i])
                prediction = sign*((actual/obs["scale"])@u@obs["margin_w"])
                actual_target = u if arm!="observed_to_dark" else v
                target_error = np.linalg.norm(actual-actual_target@(actual_target.T@actual))
                record = dict(state_id=row["state_id"],rule=row["rule"],delay=row["delay"],
                    donor_state_id=rows[donor]["state_id"],layer=12,rank=7,observer=name,arm=arm,
                    radius=0.5,actual_l2=receipt["actual_l2"],matched_random_actual_l2=random_receipt["actual_l2"],
                    norm_match_relative_error=abs(receipt["actual_l2"]-random_receipt["actual_l2"])/0.5,
                    fp16_target_frame_error=float(target_error),
                    physical_route_valid=int(target_error/0.5<=0.05 and abs(receipt["actual_l2"]-0.5)/0.5<=0.05
                        and abs(receipt["actual_l2"]-random_receipt["actual_l2"])/0.5<=0.05),
                    baseline_correct_margin=sign*float(margins[i]),effect=float(effect),
                    matched_random_effect=float(sign*(random_receipt["margin_a_minus_b"]-margins[i])),
                    predicted_effect=float(prediction),prediction_abs_error=float(abs(prediction-effect)),
                    input_prediction_abs_error=float(abs(effect)),
                    derivative_epsilon_025=sign*estimates[0],derivative_epsilon_050=sign*estimates[1],
                    derivative_stable=int(accepted),forward_key=receipt["key"],
                    random_forward_key=random_receipt["key"],effect_type="conditional_residual_state_effect",
                    source_hash=runtime.source_hash)
                for predictor in ("causal","task_only","random","pca"):
                    model = observers[f"12_7_{predictor}"]
                    predicted = sign*((actual/model["scale"])@model["u"]@model["margin_w"])
                    record[f"same_delta_prediction_{predictor}"] = float(predicted)
                    record[f"same_delta_error_{predictor}"] = float(abs(predicted-effect))
                    opposite = "copy" if row["rule"]=="flip" else "flip"
                    transferred = observers[f"transfer_{opposite}_{predictor}"]
                    tp = sign*((actual/transferred["scale"])@transferred["u"]@transferred["margin_w"])
                    record[f"cross_rule_error_{predictor}"] = float(abs(tp-effect))
                output.append(record)
            # Full centered residual erasure is exploratory, not a pure readout ablation.
            delta = -u@(u.T@(h-obs["mean"]))
            receipt,_ = runtime.forward(row,delta)
            rr,_ = runtime.forward(row,normed(haar(1024,1,("erasure_control",row["state_id"],name))[:,0],np.linalg.norm(delta)))
            output.append(dict(state_id=row["state_id"],rule=row["rule"],delay=row["delay"],
                donor_state_id="",layer=12,rank=7,observer=name,arm="exploratory_centered_residual_erasure",
                radius=float(np.linalg.norm(delta)),actual_l2=receipt["actual_l2"],
                matched_random_actual_l2=rr["actual_l2"],
                norm_match_relative_error=abs(receipt["actual_l2"]-rr["actual_l2"])/max(receipt["actual_l2"],1e-12),
                effect=float(sign*(receipt["margin_a_minus_b"]-margins[i])),
                matched_random_effect=float(sign*(rr["margin_a_minus_b"]-margins[i])),
                effect_type="conditional_residual_state_effect",forward_key=receipt["key"],random_forward_key=rr["key"],
                source_hash=runtime.source_hash))
        write_csv(path,output)
        runtime.batch(f"interventions_{row['state_id']}")
    return output


def summarize(runtime,rows,per_rank,interventions,hardware):
    paired = []
    arms = ("dark_to_observed","observed_to_dark","observed_to_observed")
    lookup = {(r["state_id"],r["observer"],r["arm"]):r for r in interventions}
    for rule in ("all","copy","flip"):
        states = [r["state_id"] for r in rows if r["split"]=="heldout" and (rule=="all" or r["rule"]==rule)]
        for arm in arms:
            for control in ("task_only","random","pca","input_rule_cue"):
                diffs = []
                effects = []
                for s in states:
                    causal = lookup[(s,"causal",arm)]
                    comparison = causal if control=="input_rule_cue" else lookup[(s,control,arm)]
                    error = causal["input_prediction_abs_error"] if control=="input_rule_cue" else causal[f"same_delta_error_{control}"]
                    diffs.append(causal["same_delta_error_causal"]-error)
                    effects.append(causal["effect"]-comparison["effect"])
                paired.append(dict(rule=rule,arm=arm,comparison=f"causal-minus-{control}",n=len(states),
                    mean_prediction_abs_error_difference=float(np.mean(diffs)),
                    paired_state_standard_error=float(np.std(diffs,ddof=1)/np.sqrt(len(diffs))),
                    mean_effect_difference=float(np.mean(effects)),
                    replication_unit="task_state_one_checkpoint"))
                if control!="input_rule_cue":
                    transfer_diffs = [lookup[(s,"causal",arm)]["cross_rule_error_causal"]-
                                      lookup[(s,"causal",arm)][f"cross_rule_error_{control}"] for s in states]
                    paired[-1]["cross_rule_prediction_error_difference"] = float(np.mean(transfer_diffs))
    write_csv(RESULTS/"paired_comparisons.csv",paired)
    source_commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    manifest = json.loads((ROOT/"model_manifest.json").read_text())
    summary = dict(status="completed_registered_residual_campaign", checkpoint=manifest["checkpoint"],
        checkpoint_revision=manifest["revision"],source_commit=source_commit,source_hash=runtime.source_hash,
        source_files=runtime.source_files,hardware=hardware,identity=runtime.identity,
        split_counts={s:sum(r["split"]==s for r in rows) for s in ("train","calibration","heldout")},
        layers=list(LAYERS),ranks=list(RANKS),nulls_per_state_rank=128,
        neural_same_state_jacobian=False,full_ROC_card=False,TB_closure="postponed",
        ablate_readout="unsupported_at_intermediate_residual_boundary",
        effect_type="conditional_residual_state_effect",directional_sketch_dimension=16,
        model_seed_replications=1,successful_forward_passes=runtime.successes,cache_hits=runtime.hits,
        paired_comparisons=paired,heldout_task=[r for r in per_rank if r["split"]=="heldout"],
        finite_difference_stable_fraction=float(np.mean([r["derivative_stable"] for r in interventions if r["arm"] in arms])),
        maximum_norm_match_relative_error=float(max(r["norm_match_relative_error"] for r in interventions)),
        physical_route_valid_fraction=float(np.mean([r["physical_route_valid"] for r in interventions if r["arm"] in arms])),
        peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in runtime.events))
    summary["result_hashes"]={p.name:sha(p) for p in RESULTS.iterdir() if p.is_file()
        and p.name in ("dataset.json","activations.npy","margins.npy","per_state.csv","per_rank.csv",
                       "null_distribution.csv","causal_interventions.csv","paired_comparisons.csv",
                       "sensitivity_sketches.json","observers.npz")}
    save(RESULTS/"summary.json",summary)
    report = ["# ROC-QWEN-01 / RTX 3050", "", f"Status: {summary['status']}.","",
        f"Checkpoint revision `{manifest['revision']}`. Source commit `{source_commit}`.",
        f"Peak CUDA allocation: {summary['peak_allocated_bytes']/1e9:.3f} GB (decimal 3.4 GB ceiling).",
        f"Forward passes this run: {runtime.successes}; cached passes reused: {runtime.hits}.","",
        "128 training / 64 calibration / 96 held-out examples; layers 6/12/18; ranks 1,2,3,4,7,8,16;",
        "128 independently seeded Haar null bases per state/layer/rank. CPU fitting only.","",
        "## Held-out task accuracy at layer 12, rank 7", ""]
    for r in summary["heldout_task"]:
        if (r["layer"]==12 and r["rank"]==7) or r["observer"]=="input_rule_cue":
            report.append(f"- {r['observer']}: {r['task_accuracy']:.4f} (n={r['n']}).")
    report += ["", "## Paired prediction error differences", "",
               "Negative means the causal observer has lower absolute prediction error. All predictors",
               "receive the same actual physical displacement generated by the causal observer arm.",
               "Effect-size comparisons separately pair observer-routed arms by recipient state.",
               "See paired_comparisons.csv for cohorts and causal_interventions.csv for every routing owner.", ""]
    for r in paired:
        if r["rule"]=="all":
            report.append(f"- {r['arm']}, {r['comparison']}: {r['mean_prediction_abs_error_difference']:.6f}.")
    report += ["", "## Limits", "", "One checkpoint, no model-seed replication. Finite differences measure a 16-direction",
        "projection for fitting, not the full output gradient. Unknown orthogonal sensitivity is not zero.",
        "State/rank null sensitivity tests are conditional on that sketch; activation capture is descriptive.",
        "Causal fitting exists at layer 12 only; other layers have PCA/task/random comparisons.",
        "Cross-rule tables distinguish readout-only transfer from primary layer-12/rank-7 bases fitted",
        "on one rule alone and evaluated on the other. Pure readout ablation is unsupported.",
        "Intermediate residual interventions with use_cache=False are conditional residual-state effects.",
        "No neural T-I, A/B/C, mechanism tags, full four-arm ROC card or recurrent quotient closure is claimed.",
        f"Stable FD fraction: {summary['finite_difference_stable_fraction']:.4f}. Actual FP16 displacement",
        f"norm matching max relative error: {summary['maximum_norm_match_relative_error']:.5f}.", "",
        "## Reproduction", "", "From repository root in WSL with the locked packages:", "", "```sh",
        "python3 -m unittest discover -s experiments/roc_qwen01_3050/tests -v",
        "python3 experiments/roc_qwen01_3050/campaign.py", "```", "",
        "Existing matching source/settings receipts are reused. See PREREGISTRATION.md, environment.lock.txt,",
        "model_manifest.json, hardware_profile.json and summary.json. No branch is merged."]
    (RESULTS/"REPORT.md").write_text("\n".join(report)+"\n")


def main():
    from smoke import torch,hardware,load
    from runtime import Runtime
    from fit import fit_all
    model,tokenizer,manifest,_ = load()
    runtime = Runtime(torch,model,manifest)
    try:
        rows,acts,margins = extract(runtime,tokenizer)
        measured = sketches(runtime,rows,acts,margins)
        observers,per_rank = fit_all(rows,acts,np.asarray(margins),measured)
        interventions = intervene(runtime,rows,acts,margins,observers)
        runtime.close()
        summarize(runtime,rows,per_rank,interventions,hardware)
    except Exception as e:
        runtime.batch("failed")
        save(RESULTS/"campaign_failure.json",dict(error=repr(e),identity=runtime.identity))
        raise


if __name__ == "__main__":
    main()
