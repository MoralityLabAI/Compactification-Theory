"""CG-09: context-shared proof acquisition, matched isolated controls and native replay."""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "critique_generation_cg08"))
import corpus as C
import search as prior

MODES = ("isolated", "shared", "dominant")
POLICIES = ("global_first", "goal_slice", "shared_relevance")
SEEDS = (141, 149)
FEATURES = 21

def fresh_cases():
    old = C.cases()
    original = C.SPLITS
    C.SPLITS = {s: (count, 109901+i, depths, gap)
                for i, (s, (count, _, depths, gap)) in enumerate(original.items())}
    try:
        rows = C.cases()
    finally:
        C.SPLITS = original
    sig = lambda r: C.sha(json.dumps(C.public(r), sort_keys=True))
    assert not {sig(r) for r in old} & {sig(r) for r in rows}
    assert not {r["id"] for r in old} & {r["id"] for r in rows}
    assert len(rows) == 512
    return rows

class Session:
    def __init__(self, case, mode):
        assert mode in MODES
        self.case = case
        self.mode = mode
        self.local = list(map(set, C.assumptions(case)))
        self.known = [C.initial(case, i) for i in range(4)]
        self.traces = [[] for _ in range(4)]
        self.firings = 0
        self.transports = 0
        self.scans = 0
        self.rank_evaluations = 0
        self.dominance_checks = 0
        self.dominated = 0
        self.transport_checks = 0
        self.paid_trace = []

    def candidates(self):
        actions = []
        closed = []
        for ctx, targets in enumerate(C.goals(self.case)):
            known = self.known[ctx]
            if all(a in known for a in targets):
                closed.append(False)
                continue
            self.scans += len(self.case["rules"])
            legal = [(ctx,j) for j,r in enumerate(self.case["rules"])
                     if r["head"] not in known and all(a in known for a in r["prem"])]
            actions.extend(legal)
            closed.append(not legal)
        # Closure status is computed BEFORE dominance filtering. Removing a
        # dominated action is not evidence that its local context is saturated.
        if self.mode == "dominant":
            by_rule = {}
            for c,j in actions:
                by_rule.setdefault(j,[]).append(c)
            filtered = []
            for c,j in actions:
                weaker = False
                for d in by_rule[j]:
                    self.dominance_checks += 1
                    if self.local[d] < self.local[c]:
                        weaker = True
                        break
                if weaker:
                    self.dominated += 1
                else:
                    filtered.append((c,j))
            actions = filtered
        return actions, closed

    def destinations(self, action):
        c,j = action
        h = self.case["rules"][j]["head"]
        if self.mode == "isolated":
            return [c]
        return [d for d in range(4)
                if self.local[c] <= self.local[d] and h not in self.known[d]]

    def step(self, action):
        c,j = action
        r = self.case["rules"][j]
        assert r["head"] not in self.known[c] and set(r["prem"]) <= self.known[c]
        dest = self.destinations(action)
        assert c in dest
        self.transport_checks += 1 if self.mode == "isolated" else 4
        for d in dest:
            # Each copied step is itself a legal rule firing in the target
            # context. Traces remain independently replayable by CG-08.
            assert set(r["prem"]) <= self.known[d]
            self.known[d].add(r["head"])
            self.traces[d].append(j)
        self.firings += 1
        self.transports += len(dest)-1
        self.paid_trace.append({"context":c, "rule":j, "destinations":dest})
        if self.mode != "isolated":
            assert all(self.known[a] <= self.known[b]
                       for a in range(4) for b in range(4)
                       if self.local[a] <= self.local[b])

    def statistics(self):
        return {"firings":self.firings, "transported_firings":self.transports,
                "local_trace_steps":sum(map(len,self.traces)),
                "rule_scans":self.scans, "rank_evaluations":self.rank_evaluations,
                "dominance_checks":self.dominance_checks, "dominated":self.dominated,
                "transport_checks":self.transport_checks}

def features(session, actions):
    base, distances = prior.features(session, actions)
    result = []
    relevance = []
    for f,action in zip(base,actions):
        c,j = action
        h = session.case["rules"][j]["head"]
        ds = session.destinations(action)
        scores = [1./(1+distances[d][h]) for d in ds if h in distances[d]]
        total = sum(scores)
        hits = sum(h in distances[d] and distances[d][h] == 0 for d in ds)
        row = f+[len(ds)/4,len(scores)/4,max(scores,default=0.),total/4,hits/4]
        assert len(row) == FEATURES
        result.append(row)
        relevance.append(total)
    return result,distances,relevance

def training_rows(rows):
    data = []
    count = 0
    for row in rows:
        if row["split"] != "train":
            continue
        case = C.public(row)
        session = Session(case,"dominant")
        rng = random.Random(int(row["id"],16))
        for _ in range(257):
            actions,closed = session.candidates()
            cert = C.certificate(case,session.known,session.traces,closed)
            if cert is not None:
                break
            assert actions
            plan = prior.teacher_plan(session)
            fs,_,_ = features(session,actions)
            hits = [sum((d,j) in plan for d in session.destinations((c,j)))
                    for c,j in actions]
            best = max(hits)
            assert best > 0,(row["id"],actions,plan)
            good = [x == best for x in hits]
            data.append((fs,good))
            labels = [i for i,x in enumerate(good) if x]
            i = rng.randrange(len(actions)) if rng.random() < .15 else rng.choice(labels)
            session.step(actions[i])
        else:
            raise AssertionError("training trajectory exhausted")
        assert C.verify(case,cert) and cert["kind"] == row["expected"]
        count += 1
    assert count == 256 and data
    return data

class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURES,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):
        return self.net(x).squeeze(-1)

def fit(data,seed,out):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Ranker()
    opt = torch.optim.AdamW(model.parameters(),lr=.003)
    history = []
    for _ in range(12):
        order = list(range(len(data)))
        rng.shuffle(order)
        losses = []
        for off in range(0,len(order),64):
            rows = [data[i] for i in order[off:off+64]]
            width = max(len(x[0]) for x in rows)
            x = torch.zeros(len(rows),width,FEATURES)
            valid = torch.zeros(len(rows),width,dtype=torch.bool)
            good = torch.zeros_like(valid)
            for i,(fs,labels) in enumerate(rows):
                x[i,:len(fs)] = torch.tensor(fs)
                valid[i,:len(fs)] = True
                good[i,:len(fs)] = torch.tensor(labels)
            logits = model(x)
            loss = (torch.logsumexp(logits.masked_fill(~valid,-1e9),-1)
                    -torch.logsumexp(logits.masked_fill(~good,-1e9),-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        history.append(sum(losses)/len(losses))
    path = out/f"ranker-{seed}.pt"
    torch.save(model.state_dict(),path)
    info = {"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
            "action_groups":len(data),"candidate_rows":sum(len(fs) for fs,_ in data),
            "loss":history,"sha256":C.sha(path.read_bytes())}
    print("FIT",json.dumps(info),flush=True)
    return model.eval(),info

@torch.no_grad()
def search(case,budget,mode,policy,model=None):
    session = Session(case,mode)
    for _ in range(budget+1):
        actions,closed = session.candidates()
        cert = C.certificate(case,session.known,session.traces,closed)
        if cert is not None or session.firings == budget or not actions:
            break
        if policy == "global_first":
            i = 0
        else:
            fs,ds,relevance = features(session,actions)
            session.rank_evaluations += len(actions)
            if model is not None:
                logits = model(torch.tensor(fs)).tolist()
                i = max(range(len(actions)),key=lambda j:logits[j])
            elif policy == "goal_slice":
                i = min(range(len(actions)),key=lambda j:(
                    ds[actions[j][0]].get(case["rules"][actions[j][1]]["head"],case["n"]),
                    actions[j]))
            elif policy == "shared_relevance":
                i = min(range(len(actions)),key=lambda j:(-relevance[j],actions[j]))
            else:
                raise ValueError(policy)
        session.step(actions[i])
    assert session.firings <= budget
    stats = session.statistics()
    assert stats["local_trace_steps"] == stats["firings"]+stats["transported_firings"]
    return cert,stats,session.paid_trace

def evaluate(rows,models,out):
    fixed = [(f"{mode}:{policy}",mode,policy,None) for mode in MODES for policy in POLICIES]
    arms = fixed+[(f"learned-{s}","dominant","learned",models[s]) for s in SEEDS]
    scores,details,accepted = [],[],{}
    paired = {}
    for split in ("cal","test_depth","test_bridge"):
        subset = [r for r in rows if r["split"] == split]
        for budget in C.BUDGETS:
            for name,mode,policy,model in arms:
                counts = Counter()
                start = time.process_time()
                for row in subset:
                    case = C.public(row)
                    cert,stats,trace = search(case,budget,mode,policy,model)
                    ok = cert is not None and C.verify(case,cert)
                    assert cert is None or (ok and cert["kind"] == row["expected"]), (row["id"],cert,row["expected"])
                    counts["n"] += 1
                    counts["complete"] += ok
                    counts["expected_"+row["expected"]] += 1
                    counts["abstain"] += cert is None
                    counts["invalid"] += cert is not None and not ok
                    if ok:
                        counts["correct_"+cert["kind"]] += 1
                        accepted.setdefault(row["id"],{})[C.sha(json.dumps(cert,sort_keys=True))] = cert
                    for k,v in stats.items():
                        counts[k] += v
                    if budget == 256 and policy == "goal_slice" and mode in ("isolated","shared"):
                        assert ok
                        x = paired.setdefault(row["id"],{"id":row["id"],"split":split,"kind":row["expected"]})
                        x[mode] = stats["firings"]
                    details.append({"id":row["id"],"split":split,"expected":row["expected"],
                                    "budget":budget,"policy":name,"certificate":cert,"verified":ok,
                                    "paid_trace":trace,**stats})
                result = {"split":split,"budget":budget,"policy":name,**counts,
                          "cpu_seconds":time.process_time()-start}
                for kind in C.KINDS:
                    result.setdefault("correct_"+kind,0)
                scores.append(result)
                print("SCORE",json.dumps(result),flush=True)

    architecture = {}
    differences = list(paired.values())
    for split in ("test_depth","test_bridge"):
        p = [r for r in differences if r["split"] == split]
        assert len(p) == 96
        iso = sum(r["isolated"] for r in p)
        shared = sum(r["shared"] for r in p)
        architecture[split] = {"isolated":iso,"shared":shared,"saved":iso-shared,
                               "relative_savings":1-shared/iso,"improved":sum(r["shared"]<r["isolated"] for r in p),
                               "ties":sum(r["shared"]==r["isolated"] for r in p),
                               "regressions":sum(r["shared"]>r["isolated"] for r in p),
                               "passed":10*shared<=9*iso}

    totals = Counter()
    for x in scores:
        if x["split"].startswith("test") and x["budget"] in (16,32):
            totals[x["policy"]] += x["complete"]
    best = max(totals[name] for name,_,_,_ in fixed)
    gate = {"architecture":architecture,"architecture_passed":all(x["passed"] for x in architecture.values()),
            "totals":dict(totals),"best_symbolic":best,
            "learned_passed":all(totals[f"learned-{s}"]>=best+6 for s in SEEDS),
            "invalid":sum(s["invalid"] for s in scores)}
    assert gate["invalid"] == 0
    (out/"paired-costs.json").write_text(json.dumps(differences,indent=2))
    (out/"scores.json").write_text(json.dumps(scores,indent=2))
    with (out/"cases.jsonl").open("w") as f:
        for record in details:
            f.write(json.dumps(record)+"\n")
    (out/"gate.json").write_text(json.dumps(gate,indent=2))
    print("GATE",json.dumps(gate),flush=True)
    return {k:[v[z] for z in sorted(v)] for k,v in accepted.items()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = fresh_cases()
    native = C.Native(args.source,args.out,args.lean)
    s = C.run_lean(args.lean,HERE/"Weakening.lean",args.out,native.env,args.out/"Weakening.log")
    C.axiom_check(s,["CG09Weakening.transport","CG09Weakening.countermodel_to_weaker",
                     "CG09Weakening.global_supplies_bridge","CG09Weakening.global_excludes_final_countermodel"])
    print("NATIVE_WEAKENING",4,flush=True)
    refs = {}
    inputs,gold = [],[]
    for row in rows:
        case = C.public(row)
        ref = C.reference(case)
        assert C.verify(case,ref) and ref["kind"] == row["expected"]
        refs[row["id"]] = [ref]
        inputs.append({"id":row["id"],"source":native.raw,"source_sha256":native.source_hash,
                       "selected":C.SELECTED,"advertised":C.prop(case,0,case["roles"][-1]),
                       "interface":case})
        gold.append({"id":row["id"],"family":row["family"],"split":row["split"],"certificate":ref})
        for mode in MODES:
            cert,stats,_ = search(case,256,mode,"global_first")
            assert C.verify(case,cert) and cert["kind"] == row["expected"]
    for name,records in (("inputs",inputs),("gold",gold)):
        (args.out/(name+".jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    assert native.replay(rows,refs,"reference") == 512
    data = training_rows(rows)
    print("TRAIN",256,len(data),flush=True)
    models,info = {},[]
    for seed in SEEDS:
        models[seed],receipt = fit(data,seed,args.out)
        info.append(receipt)
    (args.out/"training.json").write_text(json.dumps(info,indent=2))
    accepted = evaluate(rows,models,args.out)
    count = native.replay([r for r in rows if r["split"]!="train"],accepted,"actor")
    assert count == sum(map(len,accepted.values()))
    print("NATIVE_GATE",count,"distinct actor certificates",flush=True)

if __name__ == "__main__":
    main()
