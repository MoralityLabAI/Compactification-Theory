"""Exact research decision oracles and independently replayable certificates.

Optimization uses finite enumeration or exact dynamic programming. A pass means
the submitted decision certificate is feasible and optimal under the stipulated
model; it does not establish scientific correctness of its surrounding prose.
"""
from __future__ import annotations
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from .core import (_packet, canonical, evaluate, require, split_quiz,
                   validate_response)
from .research_cases import DOCKET, compile_research, fraction

def _subsets(rows):
    for k in range(len(rows)+1):
        yield from combinations(rows,k)

def _ids(rows):
    return sorted(r["id"] for r in rows)

def _chosen(rows,ids):
    require(isinstance(ids,list) and all(isinstance(x,str) for x in ids),"invalid selected IDs")
    require(ids==sorted(set(ids)),"IDs must be unique and sorted")
    lookup={r["id"]:r for r in rows}
    require(set(ids)<=set(lookup),"unknown ID")
    return [lookup[x] for x in ids]

def _keys(value,keys):
    require(isinstance(value,dict) and set(value)==set(keys),"wrong certificate shape")

def _int(value):
    require(type(value) is int,"expected integer")
    return value

def _frac(value):
    require(isinstance(value,str) and len(value)<=128,"expected reduced fraction string")
    x=Fraction(value)
    require(value==fraction(x.numerator,x.denominator),"noncanonical fraction")
    return x

def _eq(a,b):
    return canonical(a)==canonical(b)

def adaptive_oracle(d):
    h,tests=d["hypotheses"],d["tests"]
    n=len(h)
    masks=[sum(1<<i for i,b in enumerate(t["bits"]) if b) for t in tests]
    @lru_cache(None)
    def mass(mask):
        return sum(x["weight"] for i,x in enumerate(h) if mask>>i&1)
    @lru_cache(None)
    def terminal(mask,decision):
        if decision=="defer":
            return 0
        return sum(x["weight"]*(d["correct_reward"] if x["class"]==decision else d["wrong_reward"])
                   for i,x in enumerate(h) if mask>>i&1)
    @lru_cache(None)
    def dp(mask,budget,used):
        value,decision=max((terminal(mask,k),k) for k in ("A","B","defer"))
        best={"decision":decision}
        for i,t in enumerate(tests):
            cost=t["cost"]
            if used>>i&1 or cost>budget:
                continue
            one=mask&masks[i]
            zero=mask^one
            if not one or not zero:
                continue
            a,p=dp(zero,budget-cost,used|(1<<i))
            b,q=dp(one,budget-cost,used|(1<<i))
            v=a+b-cost*mass(mask)
            if v>value:
                value,best=v,{"test":t["id"],"zero":p,"one":q}
        return value,best
    v,p=dp((1<<n)-1,d["budget"],0)
    return Fraction(v,mass((1<<n)-1)),p

def replay_policy(d,policy):
    """Validate reachable tree and replay every hypothesis independently of DP."""
    hs=d["hypotheses"]
    tests={t["id"]:t for t in d["tests"]}
    def walk(node,indices,budget,used):
        require(isinstance(node,dict),"invalid policy node")
        if set(node)=={"decision"}:
            require(node["decision"] in ("A","B","defer"),"unknown decision")
            return
        _keys(node,("test","zero","one"))
        name=node["test"]
        require(isinstance(name,str) and name in tests and name not in used,"unknown or repeated test")
        t=tests[name]
        require(t["cost"]<=budget,"path budget exceeded")
        zero=[i for i in indices if t["bits"][i]==0]
        one=[i for i in indices if t["bits"][i]==1]
        require(bool(zero) and bool(one),"noninformative or unreachable branch")
        walk(node["zero"],zero,budget-t["cost"],used|{name})
        walk(node["one"],one,budget-t["cost"],used|{name})
    walk(policy,list(range(len(hs))),d["budget"],set())
    numerator=0
    for i,h in enumerate(hs):
        node=policy
        spent=0
        while "test" in node:
            t=tests[node["test"]]
            spent+=t["cost"]
            node=node["one"] if t["bits"][i] else node["zero"]
        decision=node["decision"]
        utility=0 if decision=="defer" else (d["correct_reward"] if decision==h["class"] else d["wrong_reward"])
        numerator+=h["weight"]*(utility-spent)
    return Fraction(numerator,sum(h["weight"] for h in hs))

def panel_value(d,rows):
    ids=set(_ids(rows))
    cost=sum(r["cost"] for r in rows)
    if cost>d["budget"] or any(not set(r["requires"])<=ids for r in rows):
        return None
    n=len(d["mechanisms"])
    if any(sum(r["predictions"][i]!=r["predictions"][j] for r in rows)<2*d["error_bits"]+1
           for i,j in combinations(range(n),2)):
        return None
    return cost

def portfolio_value(d,rows):
    ids=set(_ids(rows))
    groups=[r["group"] for r in rows if r["group"] is not None]
    if (sum(r["cpu"] for r in rows)>d["cpu_budget"] or
        max((r["ram"] for r in rows),default=0)>d["ram_cap"] or
        len(groups)!=len(set(groups)) or
        any(not r["permitted"] or not set(r["requires"])<=ids for r in rows)):
        return None
    return min(sum(r["gains"][i] for r in rows) for i in range(len(d["scenarios"])))

def paired_value(d):
    answers=[]
    eligible=[]
    for arm in d["arms"]:
        differences=[Fraction(sum(r["active"]),len(r["active"]))-Fraction(sum(r["sham"]),len(r["sham"]))
                     for r in arm["roots"]]
        delta=sum(differences,Fraction(0))/len(differences)
        positive=sum(x>0 for x in differences)
        matched=all(r["active_visits"]==r["sham_visits"] and r["active_dose"]==r["sham_dose"]
                    for r in arm["roots"])
        answers.append({"id":arm["id"],"mean_delta":fraction(delta.numerator,delta.denominator),
                        "positive_roots":positive,"matched":matched})
        if matched and positive>=d["min_positive"] and delta>=Fraction(d["min_mean"]):
            eligible.append(arm["id"])
    return {"arms":answers,"eligible":sorted(eligible)}

def closure(d,rows):
    resolved={c for j in rows for c in j["resolves"]}
    checked=set()
    for c in d["claims"]:
        own=c["state"]=="checked" or (c["state"]=="open" and c["id"] in resolved)
        if own and set(c["requires"])<=checked:
            checked.add(c["id"])
    return sorted(checked),sum(c["value"] for c in d["claims"] if c["id"] in checked)

def docket_value(d):
    return {"ordinary_loss_millionths":d["cheap_route"]["ordinary"]-d["forced_scored_route"]["ordinary"],
            "ood_loss_millionths":d["cheap_route"]["ood"]-d["forced_scored_route"]["ood"],
            "learned_promotion_supported":False}

def reference(case):
    kind,d=case["kind"],case["data"]
    if kind=="adaptive_evidence":
        v,p=adaptive_oracle(d)
        require(replay_policy(d,p)==v,"internal adaptive policy replay mismatch")
        return {"policy":p,"value":fraction(v.numerator,v.denominator)}
    if kind=="robust_discrimination":
        best=None
        for rows in _subsets(d["experiments"]):
            c=panel_value(d,rows)
            if c is not None and (best is None or c<best["cost"]):
                best={"status":"plan","experiments":_ids(rows),"cost":c}
        return {"status":"infeasible"} if best is None else best
    if kind=="research_portfolio":
        best={"jobs":[],"worst_gain":0}
        for rows in _subsets(d["jobs"]):
            v=portfolio_value(d,rows)
            if v is not None and v>best["worst_gain"]:
                best={"jobs":_ids(rows),"worst_gain":v}
        return best
    if kind=="paired_causal_gate":
        return paired_value(d)
    if kind=="proof_obligations":
        current,v=closure(d,[])
        best={"current":current,"jobs":[],"conditional":current,"conditional_value":v}
        for rows in _subsets(d["jobs"]):
            if sum(j["cost"] for j in rows)>d["budget"]:
                continue
            claims,value=closure(d,rows)
            if value>best["conditional_value"]:
                best={"current":current,"jobs":_ids(rows),"conditional":claims,"conditional_value":value}
        return best
    if kind==DOCKET:
        return docket_value(d)
    raise ValueError("unknown research family")

def assess(case,answer):
    """Compare submitted decision to an evaluator-owned optimum, accepting ties."""
    kind,d=case["kind"],case["data"]
    ref=reference(case)
    result={"feasible":False,"optimal":False,"certificate_consistent":False,
            "candidate_value":None,"oracle_value":None,"gap":None}
    if kind=="adaptive_evidence":
        optimal=Fraction(ref["value"])
        result["oracle_value"]=ref["value"]
    elif kind=="robust_discrimination":
        optimal=ref.get("cost")
        result["oracle_value"]=optimal
    elif kind=="research_portfolio":
        optimal=ref["worst_gain"]
        result["oracle_value"]=optimal
    elif kind=="proof_obligations":
        optimal=ref["conditional_value"]
        result["oracle_value"]=optimal
    try:
        if kind=="adaptive_evidence":
            _keys(answer,("policy","value"))
            v=replay_policy(d,answer["policy"])
            stated=_frac(answer["value"])
            gap=optimal-v
            require(gap>=0,"candidate exceeds oracle")
            result.update(feasible=True,certificate_consistent=stated==v,
                          candidate_value=fraction(v.numerator,v.denominator),
                          gap=fraction(gap.numerator,gap.denominator),
                          optimal=gap==0 and stated==v)
        elif kind=="robust_discrimination":
            if _eq(answer,{"status":"infeasible"}):
                result.update(feasible=optimal is None,optimal=optimal is None,
                              certificate_consistent=optimal is None,gap=0 if optimal is None else None)
            else:
                _keys(answer,("status","experiments","cost"))
                require(answer["status"]=="plan","unknown decision")
                c=panel_value(d,_chosen(d["experiments"],answer["experiments"]))
                if c is not None:
                    require(optimal is not None and c>=optimal,"candidate beats oracle")
                    consistent=_int(answer["cost"])==c
                    result.update(feasible=True,certificate_consistent=consistent,candidate_value=c,
                                  gap=c-optimal,optimal=c==optimal and consistent)
        elif kind=="research_portfolio":
            _keys(answer,("jobs","worst_gain"))
            v=portfolio_value(d,_chosen(d["jobs"],answer["jobs"]))
            if v is not None:
                require(v<=optimal,"candidate beats oracle")
                consistent=_int(answer["worst_gain"])==v
                result.update(feasible=True,certificate_consistent=consistent,candidate_value=v,
                              gap=optimal-v,optimal=v==optimal and consistent)
        elif kind=="proof_obligations":
            _keys(answer,("current","jobs","conditional","conditional_value"))
            rows=_chosen(d["jobs"],answer["jobs"])
            if sum(r["cost"] for r in rows)<=d["budget"]:
                claims,v=closure(d,rows)
                require(v<=optimal,"candidate beats oracle")
                consistent=(_eq(answer["current"],ref["current"]) and
                            _eq(answer["conditional"],claims) and _int(answer["conditional_value"])==v)
                result.update(feasible=True,certificate_consistent=consistent,candidate_value=v,
                              gap=optimal-v,optimal=v==optimal and consistent)
        else:
            passed=_eq(answer,ref)
            result.update(feasible=passed,certificate_consistent=passed,optimal=passed)
    except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,RecursionError):
        return result
    return result

def grade_research(packet,response,*,adapters=None):
    _packet(packet)
    ins=packet.get("research")
    require(isinstance(ins,dict),"not a research packet")
    parent=compile_research(ins["family"],seed=ins["seed"],ordinal=ins["ordinal"],
                            difficulty=packet["difficulty"],adapters=adapters)
    require(packet==parent or any(packet==part for k in range(1,5) for part in split_quiz(parent,max_questions=k)),
            "research packet does not match reconstructed source")
    answers=validate_response(packet,response)
    metrics={}
    def checker(case,fields):
        metrics.update(assess(case,fields["answer"]))
        return metrics["optimal"]
    receipt=evaluate(packet,response,checkers={"case.solve":checker} if "case.solve" in answers else {})
    receipt["research_decision"]={"family":ins["family"],"content_sha256":ins["content_sha256"],
          "metrics":metrics if metrics else None,
          "scope":"exact finite decision or source-aggregate check; scientific interpretation and proposal review pending"}
    return receipt
