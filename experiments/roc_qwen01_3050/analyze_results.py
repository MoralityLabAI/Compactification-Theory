"""Descriptive paired task/cohort analysis from frozen public CPU records."""
import csv
import json
import numpy as np
from core import features, ridge, write_csv
from preflight import ROOT, RESULTS, sha, save


def read(name):
    with open(RESULTS/name,newline="") as f:
        return list(csv.DictReader(f))


def main():
    summary=json.loads((RESULTS/"summary.json").read_text())
    rows=json.loads((RESULTS/"dataset.json").read_text())
    held=[i for i,r in enumerate(rows) if r["split"]=="heldout"]
    train=[i for i,r in enumerate(rows) if r["split"]=="train"]
    truth=np.asarray([r["answer"]==0 for r in rows])
    margins=np.load(RESULTS/"margins.npy",mmap_mode="r")
    native=(margins>=0)==truth  # deterministic A tie-break, also used by probes
    f=features(rows)
    w=ridge(f[train],2*truth[train].astype(float)-1)
    input_correct=(f@w>=0)==truth
    state=read("per_state.csv")
    correct={(r["state_id"],r["observer"]):int(r["task_correct"]) for r in state
             if r["layer"]=="12" and r["rank"]=="7" and r["split"]=="heldout"}
    cohorts={"all":held,"copy":[i for i in held if rows[i]["rule"]=="copy"],
             "flip":[i for i in held if rows[i]["rule"]=="flip"]}
    cohorts.update({f"delay_{d}":[i for i in held if rows[i]["delay"]==d] for d in range(2,9)})
    paired=[]
    for cohort,ix in cohorts.items():
        causal=np.asarray([correct[(rows[i]["state_id"],"causal")] for i in ix])
        for control in ("task_only","random","pca","input_rule_cue"):
            other=input_correct[ix].astype(int) if control=="input_rule_cue" else np.asarray([
                correct[(rows[i]["state_id"],control)] for i in ix])
            paired.append(dict(cohort=cohort,control=control,n=len(ix),
                causal_accuracy=float(causal.mean()),control_accuracy=float(other.mean()),
                paired_accuracy_difference=float(np.mean(causal-other)),
                causal_only_correct=int(np.sum((causal==1)&(other==0))),
                control_only_correct=int(np.sum((causal==0)&(other==1))),
                inference="descriptive_single_checkpoint_prompt_pairs"))
    write_csv(RESULTS/"paired_task_comparisons.csv",paired)
    # Cross-rule bases and readouts were both fitted using the other rule.
    arrays=np.load(RESULTS/"observers.npz")
    activations=np.load(RESULTS/"activations.npy",mmap_mode="r")
    transfer=[]
    for i in held:
        source="copy" if rows[i]["rule"]=="flip" else "flip"
        for method in ("causal","task_only","random","pca"):
            key=f"transfer_{source}_{method}"
            x=(activations[i,1].astype(float)-arrays[key+"_mean"])/arrays[key+"_scale"]
            pred=x@arrays[key+"_u"]@arrays[key+"_task_w"]
            transfer.append(dict(state_id=rows[i]["state_id"],source_rule=source,
                target_rule=rows[i]["rule"],observer=method,prediction=float(pred),
                correct=int((pred>=0)==truth[i])))
    write_csv(RESULTS/"cross_rule_per_state.csv",transfer)
    summary["native_qwen_heldout_accuracy"]=float(np.mean(native[held]))
    summary["native_qwen_heldout_by_cohort"]={k:float(np.mean(native[ix])) for k,ix in cohorts.items()}
    summary["paired_task_comparisons"]=paired
    summary["analysis_source_sha256"]=sha(ROOT/"analyze_results.py")
    summary["budget_scope"]="same rank, training examples, 100 CPU optimizer steps for task/causal; causal also uses label-free finite-difference measurements"
    for name in ("paired_task_comparisons.csv","cross_rule_per_state.csv"):
        summary["result_hashes"][name]=sha(RESULTS/name)
    for item in summary["paired_comparisons"]:
        if item["comparison"]=="causal-minus-input_rule_cue":
            item["mean_effect_difference"]=None
    save(RESULTS/"summary.json",summary)
    report=RESULTS/"REPORT.md"
    text=report.read_text()
    text+=f"\n## Native Qwen and paired task outcomes\n\nNative held-out answer accuracy: {summary['native_qwen_heldout_accuracy']:.4f} (96 prompts, A tie-break).\n\n"
    text+="| Comparator | Causal probe accuracy | Comparator accuracy | Paired difference |\n|---|---:|---:|---:|\n"
    for r in paired:
        if r["cohort"]=="all":
            text+=f"| {r['control']} | {r['causal_accuracy']:.4f} | {r['control_accuracy']:.4f} | {r['paired_accuracy_difference']:+.4f} |\n"
    text+="\nRule and delay cohorts are in paired_task_comparisons.csv. Genuine cross-rule per-state predictions are in cross_rule_per_state.csv.\n\n"
    text+="The task-only and causal fits match rank, training examples, initialization and 100 CPU optimizer steps. The causal fit additionally uses label-free directional measurements; total forward acquisition cost is not claimed equal.\n\n"
    text+="The 23-token hardware pilot is retained as smoke_short_prompt_pilot.json and is excluded from the acceptance smoke. The corrected acceptance smoke uses 35 tokens.\n\n"
    text+=f"Physical donor-route validity fraction under the preregistered 5% FP16 tolerance: {summary['physical_route_valid_fraction']:.4f}.\n"
    report.write_text(text)
    print(json.dumps({"native_qwen_heldout_accuracy":summary["native_qwen_heldout_accuracy"],
                      "paired_task_all":[x for x in paired if x["cohort"]=="all"]},indent=2))


if __name__=="__main__":
    main()
