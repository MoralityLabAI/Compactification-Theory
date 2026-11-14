"""FC-12 registered concrete Lean actions from ACTUAL printed tactic states.

An action is a family plus 0-4 concrete tactic script candidates instantiated
only from local goal-context declarations. No original reference tactic text,
theorem identifier or source file is used to instantiate an action.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from research.refutation_bench.fc11 import source_core as imported

SOURCE_COMMIT="c84e2025334e9a967c1297425a490f98d5b3d189"
TOOLCHAIN="leanprover/lean4:v4.35.0"
FAMILIES=(
    "omega","rfl","simp_all","assumption","intro","constructor",
    "exact_local","apply_local","rw_local",
    "rcases_local","induction_list","simpa_local",
)
MAX_ARGUMENTS=4
MAX_TRANSITIONS=3
MAX_TOTAL_LEAN_PROBES=12


def contextual_names(actual_goal:str)->list[tuple[str,str]]:
    """Parse only the local hypothesis telescope printed by Lean.

    Handles simple "x y : Nat" declarations and local propositional
    assumptions. It ignores variable names in the goal conclusion.
    """
    if "⊢" not in actual_goal:
        raise AssertionError("actual Lean turnstile required")
    prefix=actual_goal.split("⊢",1)[0]
    rows=[]
    for line in prefix.splitlines():
        line=line.strip()
        m=re.fullmatch(r"([A-Za-z_][A-Za-z0-9_']*(?:\s+[A-Za-z_][A-Za-z0-9_']*)*)\s*:\s*(.+)",line)
        if not m:
            continue
        identifiers=m.group(1).split()
        category=m.group(2).strip()
        for name in identifiers:
            if name not in ("case","case'"):
                rows.append((name,category))
    return rows


def generate(actual_goal:str)->dict[str,list[str]]:
    """Deterministic concrete scripts in fixed family/context order."""
    local=contextual_names(actual_goal)
    names=[n for n,_ in local if n not in ("_",)]
    out={f:[] for f in FAMILIES}
    for f in FAMILIES[:6]:
        out[f]=[f]
    for name in names:
        out["exact_local"].append("exact "+name)
        out["apply_local"].append("apply "+name)
        out["simpa_local"].append("simpa using "+name)
    for name,typ in local:
        if re.search(r"(?:[^<>!])=(?:[^=>])|↔|≃",typ):
            out["rw_local"].extend(("rw ["+name+"]","rw [← "+name+"]"))
        if any(symbol in typ for symbol in ("∃","∧","Exists","And ")):
            out["rcases_local"].append("rcases "+name+" with ⟨fc12_left, fc12_right⟩")
        if "List " in typ or "List(" in typ or typ.startswith("List "):
            out["induction_list"].append("induction "+name)
    for family in FAMILIES:
        seen=set()
        scripts=[]
        for tactic in out[family]:
            if tactic not in seen:
                seen.add(tactic)
                scripts.append(tactic)
            if len(scripts)>=MAX_ARGUMENTS:
                break
        out[family]=scripts
    if set(out)!=set(FAMILIES) or any(len(x)>MAX_ARGUMENTS for x in out.values()):
        raise AssertionError("candidate family or argument cap violated")
    return out


def canonical_rows(path:Path,rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    content="".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in rows)
    path.write_text(content,encoding="utf8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def states_from_compiler(root:Path)->list[dict]:
    rows=[json.loads(line) for line in (root/"actual_goal_states.jsonl").read_text(encoding="utf8").splitlines()]
    if len({x["stage_id"] for x in rows})!=len(rows):
        raise AssertionError("repeated imported theorem source stage")
    if any("⊢" not in x["actual_lean_goal_state"] for x in rows):
        raise AssertionError("non-Lean goal text supplied")
    return rows
