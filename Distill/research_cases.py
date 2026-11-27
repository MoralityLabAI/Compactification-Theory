"""DISTILL-05: research decisions and first-principles scientific reasoning.

All generated worlds are bounded public-development planning models. MTR_DOCKET
is a source-pinned published development result. Nothing here is a live gym
outcome, a sealed evaluation, a training export, or a model competence claim.
"""
from __future__ import annotations
import copy
from .core import (CASE, EXPLAIN, DIFFICULTIES, RESEARCH_GENERATOR, RESEARCH_VERSION,
                   _packet, compile_quiz, digest, require)
from .instances import Rng
from .registry import load_registry

LEVEL = {"foundation":0,"standard":1,"frontier":2}
FAMILIES = {
    "adaptive_evidence": "MTR-02D",
    "robust_discrimination": "BPD-DT01",
    "research_portfolio": "MOLT-GOBLIN-01",
    "paired_causal_gate": "SCM-23",
    "proof_obligations": "RH-PROOF",
}
MTR_COMMIT = "cbc028c2525b148269e14b43b5f379f5c10a9b72"
MTR_PROTOCOL = {"repository":"MoralityLabAI/DataGoblin","commit":MTR_COMMIT,
                "path":"training/rlt_gym/hypergame_taste/MTR_02D_PROTOCOL.md",
                "blob":"915e0aff1863e25c1860d50d7e0f39263dcd68f4"}
MTR_REPORT = {"repository":"MoralityLabAI/DataGoblin","commit":MTR_COMMIT,
              "path":"training/rlt_gym/hypergame_taste/MTR_02I_REPORT.md",
              "blob":"3e059a1866bed8bbad738296fe33e64a7e382d01"}
DOCKET = "mtr_negative_result"

PROMPTS = {
    "adaptive_evidence": (
        "Choose a contingent research policy maximizing expected net utility under the per-path credit budget. "
        "Hypothesis weights are prior masses. Each test reveals its bit in the matching hypothesis row, costs its listed credits, "
        "and subtracts the same number of utility units. Tests may be used at most once per path. "
        "A terminal decision A or B receives correct_reward if it matches the hypothesis class, wrong_reward otherwise; defer receives 0. "
        "A policy node is either {decision: A/B/defer} or {test: test ID, zero: policy, one: policy}. "
        "Use a test only if both outcomes remain possible. Supply answer as {policy: tree, value: reduced fraction string n/d}. "
        "Show conditional masses, one branch computation and why the whole policy is optimal; multiple optimal policies are accepted."
    ),
    "robust_discrimination": (
        "Find the cheapest controlled intervention panel able to distinguish every pair of candidate mechanisms even if up to error_bits "
        "of the measured bits are corrupted. Each experiment gives one binary prediction per mechanism; selected columns must separate "
        "each pair in at least 2*error_bits+1 positions. Select each experiment at most once and include every required control ID. "
        "The total cost must fit budget. Answer {status: plan, experiments: sorted list of IDs, cost: integer}, or {status: infeasible} "
        "if no panel fits. Explain the pair-separation certificate and the optimal cost or impossibility. Predictions are hypothetical, not data."
    ),
    "research_portfolio": (
        "Choose an admissible research portfolio maximizing the minimum of the summed gains over the supplied scenarios. "
        "Jobs run sequentially: sum CPU credits, take the maximum selected RAM, and require all dependencies. Only permitted jobs may be selected, "
        "and at most one job from each non-null group may be selected. An empty portfolio is feasible with gain zero. "
        "Answer {jobs: sorted list of IDs, worst_gain: integer}. Demonstrate cost, dependency and worst-scenario calculations "
        "and a valid optimality argument; ties may use any optimal portfolio. Scenario gains are stipulated planning utilities, not measured learning gains."
    ),
    "paired_causal_gate": (
        "Compute the frozen descriptive gate for every arm. For each root take mean(active) minus mean(sham); average these root differences "
        "with equal root weight, regardless of repeat counts. A positive root has strictly positive difference. An arm is matched only when "
        "active and sham visits and dose are equal at every root. An eligible arm must be matched, have at least min_positive positive roots "
        "and mean_delta >= min_mean (exact rational). Answer {arms: [{id: ID, mean_delta: reduced fraction n/d, positive_roots: integer, "
        "matched: boolean}, ... in given order], eligible: sorted list of IDs}. Explain any mismatch and limits of this descriptive gate. "
        "No inferential or population-wide conclusion is licensed."
    ),
    "proof_obligations": (
        "Choose audit jobs within budget to maximize total value of claims that would be releasable IF every selected job succeeds. "
        "A claim is releasable only if its own state is checked and every dependency is recursively releasable. A selected audit job "
        "can change only its listed open claims to checked under the hypothetical success assumption. Refuted claims stay refuted. "
        "Jobs are unique and costs add. Answer {current: sorted claim IDs releasable before jobs, jobs: sorted selected job IDs, "
        "conditional: sorted claim IDs releasable after hypothetical successes, conditional_value: integer}. "
        "Show closure and optimality; distinguish today's certificate from a proposed audit. Any optimal feasible job selection is accepted."
    ),
    DOCKET: (
        "From the published MTR-02I development aggregates, compute cheap_route minus forced_scored_route at the 8192 work cap "
        "for ordinary and OOD, in integer millionths of utility. The learned route choices and forced scored route receipts match. "
        "Answer {ordinary_loss_millionths: integer, ood_loss_millionths: integer, learned_promotion_supported: boolean}. "
        "Explain the evidence for selection overhead and the limits of the inference. The displayed values are rounded to six decimals, "
        "so use their displayed integer-millionth representation rather than assert greater precision."
    ),
}

INTERPRET = {
    "adaptive_evidence":"Compare information gain, feasible decision utility and opportunity cost. Explain why a high-information test can have low decision value, and give a concrete example tied to this problem.",
    "robust_discrimination":"Explain which scientific alternatives the panel discriminates, what the error model assumes, and how a shared nuisance or correlated errors could defeat its interpretation.",
    "research_portfolio":"Explain which hypotheses the selected studies would resolve, what evidence the scenario utilities omit, and what a cheap baseline or ablation would test before allocating more compute.",
    "paired_causal_gate":"Identify the estimand, independent sampling unit, confounded arms and the scope of the gate. Explain why repeated measurements do not provide independent roots.",
    "proof_obligations":"Separate a verified formal consequence, source-text evidence, an unresolved premise and a proposed audit. Explain why closing an audit plan does not by itself establish a mathematical theorem.",
    DOCKET:"Explain how a matched intervention isolates scoring overhead from choice effects. Identify the evidence that the learned route policy cannot yet earn a promotion and distinguish measured logical work from wall time.",
}
DESIGN = (
    "Propose a next research experiment grounded in this result. Name a falsifiable hypothesis and a plausible competing explanation, "
    "the intervention, strongest cheap baseline and sham or negative control, independent units and split, measured resource and information budgets, "
    "and a prospective threshold for advancing the hypothesis. Give the cheapest decisive measurement you can defend."
)
OBSTRUCTION = (
    "Construct a structural counterexample to a tempting stronger conclusion or the proposed method: change a modeling premise, "
    "identifiability condition, dependency, noise model or information boundary. Supply a concrete small witness and a check. "
    "A routine change of numerical input that leaves the same scientific argument intact is insufficient. "
    "Explain whether to repair, stop, or narrow the claim, and what evidence would justify revisiting that decision."
)

def fraction(n,d):
    from fractions import Fraction
    f=Fraction(n,d)
    return f"{f.numerator}/{f.denominator}"

def _adapter(family, adapters):
    name=FAMILIES.get(family)
    if family in ("adaptive_evidence",DOCKET):
        a=copy.deepcopy(adapters["MOLT-GOBLIN-01"])
        a.update(gym_id="MTR-02D" if family=="adaptive_evidence" else "MTR-02I",
                 gym_version="MTR-02D public protocol" if family=="adaptive_evidence" else "MTR-02I published development",
                 source=MTR_PROTOCOL if family=="adaptive_evidence" else MTR_REPORT,
                 objective="Allocate research effort to feasible decisions under a measured work budget.",
                 context="MTR studies finite evidence DAGs, exact feasible incumbent utility and optimistic bounds. "
                         "The learned scorer competes against cheap analytic routes at matched work. "
                         "The finite examination is separate from the native 128-witness campaign.")
        return a
    require(name in adapters,"unknown research source")
    return adapters[name]

def _shuffle(r, items):
    result=list(items)
    for i in range(len(result)-1,0,-1):
        j=r.below(i+1)
        result[i],result[j]=result[j],result[i]
    return result

def _data(family,seed,ordinal,difficulty):
    require(family in FAMILIES or family==DOCKET,"unknown research family")
    require(difficulty in LEVEL and type(seed) is int and 0<=seed<2**63 and
            type(ordinal) is int and 0<=ordinal<10**7,"invalid research coordinates")
    if family==DOCKET:
        require(seed==0 and ordinal==0 and difficulty=="frontier","fixed docket coordinates")
        return {"work_cap":8192,"ordinary_roots":24,"ood_roots":24,
                "cheap_route":{"ordinary":13241905,"ood":13192982},
                "learned_scored_route":{"ordinary":9983440,"ood":10088079},
                "forced_scored_route":{"ordinary":9983440,"ood":10088079},
                "all_scored_policies_selected_route":True,"paired_forced_route_replays":96,
                "units":"displayed utility millionths; forced-route logical-work replay at both budgets",
                "source_note":"public development; no promotion, no native wall-time claim"}
    l=LEVEL[difficulty]
    r=Rng([RESEARCH_GENERATOR,family,seed,ordinal,difficulty])
    if family=="adaptive_evidence":
        n=8+2*l
        m=4+l
        classes=_shuffle(r,["A"]*(n//2)+["B"]*(n//2))
        h=[{"weight":r.between(1,5),"class":c} for c in classes]
        tests=[]
        seen=set()
        while len(tests)<m:
            bits=tuple(r.below(2) for _ in h)
            if len(set(bits))==2 and bits not in seen and tuple(1-b for b in bits) not in seen:
                seen.add(bits)
                tests.append({"id":"t"+str(len(tests)),"cost":r.between(1,3),"bits":list(bits)})
        return {"hypotheses":h,"tests":tests,"budget":3+2*l,
                "correct_reward":12,"wrong_reward":-16}
    if family=="robust_discrimination":
        n=3+l
        controls=[{"id":"c0","cost":1+r.below(2),"requires":[],"predictions":[0]*n},
                  {"id":"c1","cost":1+r.below(2),"requires":[],"predictions":[0]*n}]
        experiments=controls+[
            {"id":"e"+str(i),"cost":r.between(1,4),
             "requires":["c"+str(r.below(2))] if r.below(4)!=0 else [],
             "predictions":[r.below(2) for _ in range(n)]}
            for i in range(6+2*l)]
        return {"mechanisms":["h"+str(i) for i in range(n)],"experiments":experiments,
                "error_bits":int(l==2),"budget":r.between(8+4*l,15+7*l)}
    if family=="research_portfolio":
        n=7+l*2
        jobs=[{"id":"j0","cpu":2,"ram":1,"requires":[],"group":None,"permitted":True,"gains":[0,0,0]}]
        for i in range(1,n):
            deps=["j0"] if r.below(3)==0 else []
            if i>1 and r.below(3)==0:
                deps=sorted(set(deps+["j"+str(r.below(i))]))
            jobs.append({"id":"j"+str(i),"cpu":r.between(1,5),"ram":r.between(1,6),
                         "requires":deps,"group":"g"+str(r.below(3)) if r.below(4)==0 else None,
                         "permitted":r.below(6)!=0,"gains":[r.between(-3,9) for _ in range(3)]})
        return {"jobs":jobs,"cpu_budget":8+3*l,"ram_cap":4+l,
                "scenarios":["nominal","skeptical","shifted"]}
    if family=="paired_causal_gate":
        n=5+2*l
        m=2+int(l>0)
        confounded=r.below(m)
        arms=[]
        for a in range(m):
            roots=[]
            for i in range(n):
                base=r.between(4,12)
                delta=r.between(-2,4) if a!=confounded else r.between(2,5)
                visits=r.between(4,9)
                dose=r.between(2,6)
                roots.append({"root":"r"+str(i),
                    "sham":[base+r.between(-1,1) for _ in range(1+r.below(3))],
                    "active":[base+delta+r.between(-1,1) for _ in range(1+r.below(4))],
                    "sham_visits":visits,"active_visits":visits+int(a==confounded and i==0),
                    "sham_dose":dose,"active_dose":dose})
            arms.append({"id":"a"+str(a),"roots":roots})
        return {"arms":arms,"min_positive":(2*n+2)//3,"min_mean":"1/1",
                "scope":"descriptive development gate only; no statistical promotion"}
    n=7+2*l
    claims=[]
    for i in range(n):
        state=("checked","open","refuted")[i] if i<3 else r.choice(("checked","open","open","refuted"))
        deps=[] if i<3 else sorted({"c"+str(r.below(i)) for _ in range(1+r.below(2))})
        claims.append({"id":"c"+str(i),"state":state,"requires":deps,"value":r.between(1,8)})
    opens=[c["id"] for c in claims if c["state"]=="open"]
    jobs=[]
    for i in range(4+l*2):
        jobs.append({"id":"j"+str(i),"cost":r.between(1,4),
                     "resolves":sorted({r.choice(opens) for _ in range(1+r.below(3))})})
    return {"claims":claims,"jobs":jobs,"budget":5+2*l}

def compile_research(family,*,seed,ordinal,difficulty="frontier",adapters=None):
    adapters=load_registry() if adapters is None else adapters
    a=_adapter(family,adapters)
    data=_data(family,seed,ordinal,difficulty)
    h=digest({"kind":family,"data":data})
    p=compile_quiz(a,profile="full",difficulty=difficulty)
    p.pop("packet_id")
    p["library_version"]=RESEARCH_VERSION
    p["objective"]="Make a defensible research decision, derive its value, and propose a decisive follow-up with explicit controls and falsification."
    p["context"]+=(" This packet contains a published development dossier." if family==DOCKET else
                   " This is a new synthetic planning model; its stated mechanisms and utilities are assumptions, not observed scientific results.")
    p["public_case"]={"classification":"public_development","case_id":"research-v1-"+h,"kind":family,"data":data}
    p["research"]={"generator":RESEARCH_GENERATOR,"family":family,"seed":seed,"ordinal":ordinal,"content_sha256":h}
    p["questions"]=[
        {"id":"case.solve","category":"research_decision","prompt":PROMPTS[family],"fields":list(CASE)},
        {"id":"research.interpret","category":"scientific_interpretation","prompt":INTERPRET[family],"fields":list(EXPLAIN)},
        {"id":"research.design","category":"research_design","prompt":DESIGN,"fields":list(EXPLAIN)},
        {"id":"research.obstruction","category":"falsification_and_repair","prompt":OBSTRUCTION,"fields":list(EXPLAIN)},
    ]
    p["packet_id"]=digest(p)
    _packet(p)
    return p

def generate_research(family,*,seed,count,difficulties=("standard","frontier"),adapters=None):
    require(family in FAMILIES and type(count) is int and 1<=count<=96,"invalid research count")
    require(difficulties and len(set(difficulties))==len(difficulties) and set(difficulties)<=DIFFICULTIES.keys(),"invalid research difficulties")
    adapters=load_registry() if adapters is None else adapters
    seen=set()
    result=[]
    ordinal=0
    while len(result)<count:
        require(ordinal<max(100,128*count),"unique research generation exhausted")
        p=compile_research(family,seed=seed,ordinal=ordinal,difficulty=difficulties[len(result)%len(difficulties)],adapters=adapters)
        ordinal+=1
        h=p["research"]["content_sha256"]
        if h not in seen:
            seen.add(h)
            result.append(p)
    return result
