"""QO-13 fixed matched-stream, cost-accounted digital certificate pilot."""
from fractions import Fraction as Q
from itertools import product
from random import Random
from hashlib import sha256
import json
from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger,certificate,frozen_error
from research.ingression_dynamics_v1.certificate_reachability import select_certificate_probe
from research.ingression_dynamics_v1.paired_probe_pilot import run_policy,stream_seed,verify_common_stream_prefix
from research.ingression_dynamics_v1.two_step_gap_pilot import SCENARIOS as OLD

SEEDS=(541,547,557);CHECKPOINTS=(24,64,96)
POLICIES=("bottleneck","certificate_h2","certificate_h3")
SCENARIOS=(
 {**OLD[0],"panel":"retrospective QO-12"},
 {**OLD[1],"panel":"retrospective QO-12"},
 {"id":"near_margin_infeasible","rates":((Q(1,4),),(Q(39,50),)),
  "classes":1,"tolerance":Q(1,5),"panel":"QO-13 new fixed synthetic"},
 {"id":"near_margin_feasible","rates":((Q(7,20),),(Q(11,20),)),
  "classes":1,"tolerance":Q(1,5),"panel":"QO-13 new fixed synthetic"},
)

def run_new(ci,case,seed,policy,checkpoints=CHECKPOINTS):
 if policy not in POLICIES[1:]:raise ValueError("unknown policy")
 n=len(case["rates"]);d=len(case["rates"][0]);k=case["classes"]
 ledger=ProbeLedger(n,d,Q(1,20),10)
 rng={(i,a):Random(stream_seed(ci,seed,i,a)) for i,a in product(range(n),range(d))}
 truth=max((max(row[a] for row in case["rates"])-min(row[a] for row in case["rates"]))/2
           for a in range(d))<=case["tolerance"]
 rows=[];calls=0;bellman=0;positive=0;fallback=0
 for t in range(checkpoints[-1]+1):
  if t in checkpoints:
   result=certificate(ledger.snapshot(),k,case["tolerance"])
   dec=result["decision"]
   covered=all(lo<=case["rates"][i][a]<=hi
               for i,row in enumerate(ledger.bounds) for a,(lo,hi) in enumerate(row))
   wrong=(dec=="certified-feasible" and not truth or dec=="certified-infeasible" and truth)
   if covered and wrong:raise AssertionError("invalid confidence certificate")
   error=frozen_error(case["rates"],result["transient_labels"],result["frozen_midpoints"])
   if covered and error>result["robust_macro_error_upper"]:raise AssertionError("invalid robust bound")
   rows.append({"scenario":case["id"],"seed":seed,"policy":policy,"queries":t,
    "decision":dec,"false_cert":bool(wrong),"covered":covered,
    "gap":str(result["optimality_gap_upper"]),"true_error":str(error),
    "extra_gap_calls":calls,"bellman_candidates":bellman,"positive_choices":positive,
    "fallback_choices":fallback,"counts":[list(v) for v in ledger.counts]})
  if t==checkpoints[-1]:break
  (i,a),m=select_certificate_probe(ledger,k,case["tolerance"],
                                  horizon=2 if policy=="certificate_h2" else 3)
  calls+=m["decision_gap_evaluations"];bellman+=m["bellman_candidates"]
  positive+=int(m["mode"]=="positive_certificate_probability")
  fallback+=int(m["mode"]!="positive_certificate_probability")
  ledger.observe(i,a,int(rng[i,a].random()<float(case["rates"][i][a])))
 return rows,tuple(ledger.receipts)

def mean(seq):
 items=tuple(Q(x) for x in seq)
 return str(sum(items,Q(0))/len(items))

def full_report(scenarios=SCENARIOS,seeds=SEEDS,policies=POLICIES,
                checkpoints=CHECKPOINTS):
 rows=[];paths=[]
 for ci,case in enumerate(scenarios):
  for seed in seeds:
   for policy in policies:
    if policy=="bottleneck":
     rr,obs=run_policy(ci,case,seed,policy,checkpoints)
     for x in rr:
      x["gap"]=str(Q(x["robust_upper"])-Q(x["true_optimum_lower"]))
      x["true_error"]=x["true_frozen_error"]
      x["covered"]=x["true_rates_covered"]
      x["extra_gap_calls"]=0;x["bellman_candidates"]=0
    else:rr,obs=run_new(ci,case,seed,policy,checkpoints)
    if len(obs)!=checkpoints[-1] or not verify_common_stream_prefix(ci,case,seed,obs):
     raise AssertionError("budget/stream mismatch")
    rows.extend(rr)
    paths.append({"scenario":case["id"],"seed":seed,"policy":policy,
                  "sha256":sha256(json.dumps(obs,separators=(",",":")).encode()).hexdigest()})
 groups=[]
 for case in scenarios:
  for budget in checkpoints:
   for policy in policies:
    records=[r for r in rows if r["scenario"]==case["id"] and
             r["queries"]==budget and r["policy"]==policy]
    if len(records)!=len(seeds):raise AssertionError("incomplete paired group")
    groups.append({"scenario":case["id"],"budget":budget,"policy":policy,
     "correct":sum(r["decision"]!="undetermined" and not r["false_cert"] for r in records),
     "undetermined":sum(r["decision"]=="undetermined" for r in records),
     "wrong":sum(r["false_cert"] for r in records),
     "coverage_failures":sum(not r["covered"] for r in records),
     "mean_gap":mean(r["gap"] for r in records),
     "mean_error":mean(r["true_error"] for r in records),
     "mean_extra_gap_calls":mean(r["extra_gap_calls"] for r in records)})
 return {"contract":"QO-13 certificate probability and compute budget",
  "policies":list(policies),"seeds":list(seeds),"checkpoints":list(checkpoints),
  "scenarios":[{"id":c["id"],"panel":c["panel"],
     "rates":[[str(v) for v in row] for row in c["rates"]],
     "classes":c["classes"],"tolerance":str(c["tolerance"])} for c in scenarios],
  "groups":groups,"trial_rows":rows,"trajectory_hashes":paths,
  "counts":{"trajectories":len(paths),"snapshots":len(rows),
            "queries":len(paths)*checkpoints[-1]}}
