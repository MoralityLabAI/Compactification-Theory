"""Independent forward-rule trace checker and Lean renderer.

This module consumes ONLY a model-proposed sequence of original rule IDs.
It does not search for a proof, select a rule, rewrite the sequence, or use
the reference proof stored in the dataset. An invalid trace is rejected.
The accepted proof term is a deterministic rendering of the SAME trace.
Lean compilation is a separate kernel-authoritative step in CI.
"""
from __future__ import annotations
import hashlib
import re
from typing import Any

MAX_ACTIONS=10
MAX_ATOMS=12
MAX_RULES=20
NAME=re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


class ProofError(ValueError):
    pass


def _validate_task(task:dict)->None:
    if not isinstance(task,dict):
        raise ProofError("task must be object")
    keys=("atoms","facts","rules","goal")
    if any(key not in task for key in keys):
        raise ProofError("task missing graph field")
    atoms=task["atoms"]
    facts=task["facts"]
    rules=task["rules"]
    goal=task["goal"]
    if (type(atoms)!=list or not 1<=len(atoms)<=MAX_ATOMS or
            any(type(x)!=str or NAME.fullmatch(x) is None for x in atoms) or
            len(set(atoms))!=len(atoms)):
        raise ProofError("invalid atom identifiers")
    if (type(facts)!=list or not facts or any(x not in atoms for x in facts) or
            len(set(facts))!=len(facts)):
        raise ProofError("invalid initial hypotheses")
    if goal not in atoms:
        raise ProofError("goal missing from atom set")
    if type(rules)!=list or len(rules)>MAX_RULES:
        raise ProofError("invalid rule list")
    for rule in rules:
        if type(rule)!=dict or set(rule)!={"premises","head"}:
            raise ProofError("rule must supply premises and head")
        pre=rule["premises"]
        if (type(pre)!=list or not pre or len(pre)>MAX_ATOMS or
                len(set(pre))!=len(pre) or any(x not in atoms for x in pre) or
                rule["head"] not in atoms):
            raise ProofError("invalid rule arguments")


def verify_trace(task:dict,trace:list[int],*,budget:int=MAX_ACTIONS)->dict[str,Any]:
    """Never repair a model's proposed proof; certify the submitted actions."""
    _validate_task(task)
    if type(budget)!=int or not 1<=budget<=MAX_ACTIONS:
        raise ProofError("invalid action budget")
    if type(trace)!=list or len(trace)>budget or any(type(i)!=int for i in trace):
        raise ProofError("trace must be a bounded list of integer rule indices")
    known=set(task["facts"])
    used=[]
    if task["goal"] in known:
        return {"verified":len(trace)==0,"scope":"explicit_Horn_trace",
                "steps":0,"known":sorted(known),"reason":"goal_is_hypothesis"}
    for step,index in enumerate(trace):
        if index<0 or index>=len(task["rules"]):
            return {"verified":False,"scope":"explicit_Horn_trace",
                    "steps":step,"reason":"index_out_of_bounds"}
        rule=task["rules"][index]
        if not set(rule["premises"]).issubset(known):
            return {"verified":False,"scope":"explicit_Horn_trace",
                    "steps":step,"reason":"unavailable_antecedent"}
        if rule["head"] in known:
            return {"verified":False,"scope":"explicit_Horn_trace",
                    "steps":step,"reason":"duplicate_consequent"}
        known.add(rule["head"])
        used.append(index)
        if rule["head"]==task["goal"] and step<len(trace)-1:
            return {"verified":False,"scope":"explicit_Horn_trace",
                    "steps":step+1,"reason":"actions_after_goal"}
    return {"verified":task["goal"] in known,"scope":"explicit_Horn_trace",
            "steps":len(used),"known":sorted(known),
            "reason":"goal_derived" if task["goal"] in known else "goal_unproved"}


def _and_term(exprs:list[str])->str:
    if len(exprs)==1:
        return exprs[0]
    if len(exprs)>1:
        return "(And.intro "+exprs[0]+" "+_and_term(exprs[1:])+")"
    raise ProofError("empty antecedent not supported")


def lean_from_trace(task:dict,trace:list[int],*,theorem_name:str)->str:
    """Convert a certified model trace into a fully explicit Lean4 theorem.

    The source's complete rule inventory is passed as hypotheses. No
    declared global axioms, sorry, or theorem-specific reference proof.
    """
    report=verify_trace(task,trace)
    if not report["verified"]:
        raise ProofError("cannot render unverified trace: "+report["reason"])
    if NAME.fullmatch(theorem_name) is None:
        raise ProofError("invalid theorem name")
    facts=task["facts"]
    atoms=task["atoms"]
    rules=task["rules"]
    hypotheses=["("+" ".join(atoms)+" : Prop)"]
    hypotheses.extend(f"(h{i} : {fact})" for i,fact in enumerate(facts))
    for j,rule in enumerate(rules):
        conjunction=" ∧ ".join(rule["premises"])
        hypotheses.append(f"(r{j} : ({conjunction}) → {rule['head']})")
    known={fact:f"h{i}" for i,fact in enumerate(facts)}
    lines=["theorem "+theorem_name+" "+" ".join(hypotheses)+" : "+task["goal"]+" := by"]
    for j,index in enumerate(trace):
        rule=rules[index]
        parts=[known[p] for p in rule["premises"]]
        expression="r"+str(index)+" "+_and_term(parts)
        name=f"d{j}"
        lines.append("  have "+name+" : "+rule["head"]+" := "+expression)
        known[rule["head"]]=name
    lines.append("  exact "+known[task["goal"]])
    return "\n".join(lines)


def lean_document(proofs:list[tuple[dict,list[int]]])->str:
    """Emit ONLY supplied verified traces, keeping their original rule IDs."""
    lines=[
      "-- FC-05 model-proposed, independently checked positive proofs.",
      "set_option linter.unusedVariables false",
      "set_option maxRecDepth 4096",
      "",
    ]
    for i,(task,trace) in enumerate(proofs):
        label=f"fc05_model_proof_{i:04d}"
        lines.append(lean_from_trace(task,trace,theorem_name=label))
        lines.append("")
    if proofs:
        lines.append("#print axioms fc05_model_proof_0000")
        lines.append(f"#print axioms fc05_model_proof_{len(proofs)-1:04d}")
    return "\n".join(lines)+"\n"


def trace_sha256(trace:list[int])->str:
    if type(trace)!=list or any(type(x)!=int for x in trace):
        raise ProofError("invalid trace")
    return hashlib.sha256((",".join(str(x) for x in trace)).encode()).hexdigest()
