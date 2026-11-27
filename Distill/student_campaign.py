"""Frozen research-student benchmark: logical caps and separate process CPU timing.

All evaluation world outcomes are replayed by the host. Only train-root labels
fit the models; calibration chooses the gate without evaluation-root labels.
No provider answers or native DataGoblin checkpoints are involved.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
import time
from .core import digest, require, write_json
from .program_curriculum import root_records
from .program_oracle import Oracle, Baseline, actual_transition, frac
from .program_student import ACTIONS, FEATURES, FEATURE_VERSION, REGRET_SCALE, Meter, Exhausted, Ranker, public_features, state_from_observation
from .research_program import Model, new_spec

SPLITS=(("train",2026112831,"shared_control"),("calibration",2026112832,"shared_control"),
        ("ordinary",2026112833,"shared_control"),("shifted",2026112834,"shifted_control"))
SEEDS=(131,137)
CAPS=(8192,32768)
PRICE=4096  # one utility unit per 4096 declared logical-work units
CPU_ENVELOPE_NS=25_000_000  # descriptive qualification, not a preemptive limit
ARMS=("stop","direct_first","myopic","reuse_first","flat","recurrent","shuffled","gated")

def _legal(model,s,meter):
    meter.charge(16)
    return model.legal(s)

def _reward(model,s,a,meter):
    meter.charge(65+(64 if a.startswith("recommend:") else 0))
    return model.reward_sum(s,a)

def _terminal(model,s,meter):
    return max((_reward(model,s,a,meter),a) for a in _legal(model,s,meter) if model.terminal(a))

def cheap(model,s,meter,name):
    legal=_legal(model,s,meter)
    meter.charge(8)
    if name=="direct_first" and "direct_audit" in legal:
        return "direct_audit"
    if name=="reuse_first":
        if s.stage==0 and s.anchor<0 and "direct_audit" in legal:
            return "direct_audit"
        if s.stage<=1 and s.anchor>=0 and s.law<0 and "law_audit" in legal:
            return "law_audit"
        if "transport" in legal:
            return "transport"
        if "direct_audit" in legal:
            return "direct_audit"
    best=_terminal(model,s,meter)
    if name in ("stop","direct_first","reuse_first"):
        return best[1]
    require(name=="myopic","unknown cheap policy")
    choices=[best]
    for a in legal:
        if not model.terminal(a):
            # A branch construction is charged even when mask arithmetic/cache is cheap.
            meter.charge(64)
            v=_reward(model,s,a,meter)
            for _,t in model.branches(s,a):
                meter.charge(16)
                v+=_terminal(model,t,meter)[0]
            choices.append((v,a))
    return max(choices,key=lambda row:(row[0],-model.cost(row[1]),row[1]))[1]

def decision(model,s,meter,arm,ranker=None,threshold=None):
    """All arms have the same cap; paid failed work remains charged."""
    fallback=False
    try:
        if arm in ("stop","direct_first","myopic","reuse_first"):
            action=cheap(model,s,meter,arm)
        elif arm=="gated":
            base=cheap(model,s,meter,"reuse_first")
            if threshold is None:
                return base,False
            try:
                action,y=ranker.choose(model,s,meter)
            except Exhausted:
                return base,True
            meter.charge(1)
            if y[ACTIONS.index(action)]-y[ACTIONS.index(base)]<=threshold:
                action=base
        else:
            require(arm in ("flat","recurrent","shuffled") and ranker is not None,"missing student")
            action,_=ranker.choose(model,s,meter)
        return action,False
    except Exhausted:
        fallback=True
        try:
            action=cheap(model,s,meter,"reuse_first")
        except Exhausted:
            action="defer"  # host-defined zero-additional-planner-work safe action
    return action,fallback

def labeled(spec):
    public,referee,_=root_records(spec,per_stage=4)
    result=[]
    for p,r in zip(public,referee):
        require(p["state_id"]==r["state_id"] and digest(p)==r["public_sha256"],"label binding")
        model,state=state_from_observation(p["observation"])
        targets={ACTIONS.index(a["action"]):-float(Fraction(a["regret"]))/REGRET_SCALE for a in r["actions"]}
        result.append((model,state,public_features(model,state),targets))
    return result

def calibration_gate(ranker,roots):
    # State-level, root-balanced calibration only. The cost of scoring is paid
    # even when the model selects the same action as the baseline.
    thresholds=(0.0,.05,.15,.30,.60)
    totals={t:[] for t in thresholds}
    for rows in roots:
        scores={t:[] for t in thresholds}
        for model,state,x,targets in rows:
            base_meter=Meter(32768)
            base=cheap(model,state,base_meter,"reuse_first")
            meter=Meter(32768)
            action,y=ranker.choose(model,state,meter)
            delta=y[ACTIONS.index(action)]-y[ACTIONS.index(base)]
            for t in thresholds:
                candidate=action if delta>t else base
                benefit=REGRET_SCALE*(targets[ACTIONS.index(candidate)]-targets[ACTIONS.index(base)])
                scores[t].append(benefit-(meter.used+1)/PRICE)
        for t in thresholds:
            totals[t].append(sum(scores[t])/len(scores[t]))
    means={t:sum(v)/len(v) for t,v in totals.items()}
    best=max(thresholds,key=lambda t:(means[t],t))
    return {"threshold":best if means[best]>0 else None,
            "state_level_mean_priced_regret_improvement":{str(t):means[t] for t in thresholds},
            "scope":"calibration-root mean of sampled-state optimal-continuation regret improvements; not trajectory utility"}

def evaluate(spec,arm,cap,ranker=None,threshold=None):
    model=Model(spec)
    sums=defaultdict(int)
    total=model.mass(model.full_mask)
    cpu_total=0
    cpu_max=0
    cpu_over=0
    max_work=0
    ntrans=0
    # Use fresh meter per world, no score cache across worlds. The kernel state
    # equals the public-ledger sufficient state, and contains no actual-world id.
    for i,w in enumerate(model.worlds):
        s=model.initial()
        meter=Meter(cap)
        counts=defaultdict(int)
        elapsed=0
        while s.stage<4:
            require(counts["actions"]<32,"nonterminating policy")
            started=time.process_time_ns()
            a,fallback=decision(model,s,meter,arm,ranker,threshold)
            elapsed+=time.process_time_ns()-started
            require(a in model.legal(s),"illegal actor output")
            bit,t,reward=actual_transition(model,s,a,i)
            require([(b,c) for b,c in model.branches(s,a) if c.mask>>i&1]==[(bit,t)],"world replay mismatch")
            counts["utility"]+=reward
            counts["credits"]+=model.cost(a)
            counts["certified"]+=int(a.startswith("certify:"))
            counts["transports"]+=int(a=="transport")
            counts["false_recommendations"]+=int(a.startswith("recommend:") and int(a[-1])!=w.truths[s.stage])
            counts["deferred"]+=int(a=="defer")
            counts["fallbacks"]+=int(fallback)
            counts["actions"]+=1
            s=t
            ntrans+=1
        require(meter.used<=cap and counts["credits"]<=spec["budget"],"budget violation")
        counts["work"]=meter.used
        counts["priced_numerator"]=counts["utility"]*PRICE-meter.used
        for k in ("utility","credits","certified","transports","false_recommendations","deferred","fallbacks","actions","work","priced_numerator"):
            sums[k]+=w.mass*counts[k]
        max_work=max(max_work,meter.used)
        cpu_max=max(cpu_max,elapsed)
        cpu_total+=elapsed
        cpu_over+=int(elapsed>CPU_ENVELOPE_NS)
    result={k:frac(Fraction(v,total)) for k,v in sums.items() if k!="priced_numerator"}
    result["priced_utility"]=frac(Fraction(sums["priced_numerator"],total*PRICE))
    result.update(max_work=max_work,world_transitions=ntrans,policy_cpu_ns=cpu_total,
                  max_world_policy_cpu_ns=cpu_max,cpu_envelope_overruns=cpu_over,
                  cpu_scope="process CPU around policy decision only; host setup, evaluator replay and training excluded")
    return result

def campaign(out,*,counts=(24,8,8,8),epochs=18,seeds=SEEDS,caps=CAPS):
    require(len(counts)==4 and all(type(n) is int and 1<=n<=64 for n in counts),"invalid split counts")
    require(seeds and len(set(seeds))==len(seeds) and all(type(s) is int for s in seeds),"invalid seeds")
    require(caps and len(set(caps))==len(caps) and all(type(c) is int and 1<=c<=1000000 for c in caps),"invalid caps")
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    specs={}
    seen=set()
    manifest=[]
    for (split,seed,family),n in zip(SPLITS,counts):
        rows=[]
        for i in range(n):
            spec=new_spec(seed,i,family,("standard","frontier")[i%2])
            rid=digest(spec)
            require(rid not in seen,"split root collision")
            seen.add(rid)
            rows.append(spec)
        specs[split]=rows
        manifest.append({"split":split,"seed":seed,"family":family,"root_ids":[digest(s) for s in rows]})
    started=time.process_time_ns()
    train_roots=[labeled(s) for s in specs["train"]]
    cal_roots=[labeled(s) for s in specs["calibration"]]
    train=[(x,y) for rows in train_roots for _,_,x,y in rows]
    label_cpu=time.process_time_ns()-started
    models={}
    fit_receipts=[]
    gates={}
    for seed in seeds:
        for arm,visits,shuffle in (("flat",1,False),("recurrent",3,False),("shuffled",3,True)):
            ranker=Ranker(seed,visits)
            started=time.process_time_ns()
            fit=ranker.fit(train,epochs=epochs,shuffle_labels=shuffle)
            fit["training_cpu_ns"]=time.process_time_ns()-started
            fit.update(arm=arm,seed=seed,visits=visits)
            models[(arm,seed)]=ranker
            fit_receipts.append(fit)
            write_json(root/(arm+"-"+str(seed)+".json"),{"feature_version":FEATURE_VERSION,"seed":seed,
                "visits":visits,"weights":ranker.p,"source_kind":"synthetic_oracle_only","training_admission":False})
        gates[seed]=calibration_gate(models[("recurrent",seed)],cal_roots)
    rows=[]
    for split in ("ordinary","shifted"):
        for spec in specs[split]:
            rid=digest(spec)
            m=Model(spec)
            oracle=Oracle(m)
            optimum=oracle.expected()
            oracle.close()
            for cap in caps:
                policies={}
                for arm in ARMS[:4]:
                    policies[arm]=[evaluate(spec,arm,cap)]
                for arm in ARMS[4:]:
                    policies[arm]=[evaluate(spec,arm,cap,
                        models[("recurrent" if arm=="gated" else arm,seed)],
                        gates[seed]["threshold"] if arm=="gated" else None) for seed in seeds]
                for arm,vals in policies.items():
                    for v in vals:
                        require(Fraction(v["utility"])<=optimum,"policy exceeds exact optimum")
                rows.append({"split":split,"root_id":rid,"cap":cap,"oracle_value":frac(optimum),"policies":policies})
    summary={}
    for split in ("ordinary","shifted"):
        for cap in caps:
            subset=[r for r in rows if r["split"]==split and r["cap"]==cap]
            def root_mean(row,arm,key):
                vals=row["policies"][arm]
                return sum((Fraction(v[key]) for v in vals),Fraction(0))/len(vals)
            metrics={}
            for arm in ARMS:
                means={k:frac(sum((root_mean(r,arm,k) for r in subset),Fraction(0))/len(subset))
                       for k in ("utility","priced_utility","work","fallbacks","certified","transports","false_recommendations")}
                paired=[root_mean(r,arm,"priced_utility")-root_mean(r,"reuse_first","priced_utility") for r in subset]
                means.update(priced_delta_vs_reuse=frac(sum(paired,Fraction(0))/len(paired)),
                    paired_wins=sum(v>0 for v in paired),paired_ties=sum(v==0 for v in paired),
                    max_work=max(v["max_work"] for r in subset for v in r["policies"][arm]),
                    cpu_envelope_overruns=sum(v["cpu_envelope_overruns"] for r in subset for v in r["policies"][arm]),
                    policy_cpu_ns=sum(v["policy_cpu_ns"] for r in subset for v in r["policies"][arm]),
                    world_transitions=sum(v["world_transitions"] for r in subset for v in r["policies"][arm]))
                means["mean_policy_cpu_ns_per_decision"]=means["policy_cpu_ns"]/means["world_transitions"]
                metrics[arm]=means
            summary[split+"/"+str(cap)]={"roots":len(subset),"policies":metrics}
    report={"schema_version":"distill.student_campaign.v1","source_kind":"synthetic_oracle_only","training_admission":False,
            "features":FEATURES,"feature_version":FEATURE_VERSION,"caps":list(caps),"price_per_unit":"1/"+str(PRICE),
            "cpu_envelope_ns":CPU_ENVELOPE_NS,"seeds":list(seeds),"epochs":epochs,"split_manifest":manifest,
            "train_states":len(train),"calibration_states":sum(map(len,cal_roots)),
            "label_cpu_ns":label_cpu,"fit":fit_receipts,"calibration":{str(k):v for k,v in gates.items()},
            "summary":summary,"rows":rows,
            "promotion":"none; public development experiment",
            "scope":"matched deterministic logical-work caps, separate measured process CPU; exact finite expected utility, no native model claims"}
    report["report_id"]=digest(report)
    write_json(root/"report.json",report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--train",type=int,default=24)
    p.add_argument("--calibration",type=int,default=8)
    p.add_argument("--ordinary",type=int,default=8)
    p.add_argument("--shifted",type=int,default=8)
    p.add_argument("--epochs",type=int,default=18)
    a=p.parse_args()
    r=campaign(a.out,counts=(a.train,a.calibration,a.ordinary,a.shifted),epochs=a.epochs)
    print(json.dumps({k:v for k,v in r.items() if k not in ("rows","split_manifest")},sort_keys=True))

if __name__=="__main__":
    main()
