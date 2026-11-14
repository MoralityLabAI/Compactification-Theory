"""FC-10 finite *tactic* vocabulary over actual Lean 4.19 elaborated goals.

The theorem statements and intended successful tactics are source metadata.
Student features MUST come from the actual Lean 'trace_state' compiler output.
At inference, validate an actual Lean tactic candidate with the Lean compiler.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

ROOT_SEED=20261113
LEAN_VERSION="4.19.0"
TACTICS=(
    "omega",
    "rfl",
    "assumption",
    "constructor <;> assumption",
    "exact h.1",
    "exact h.2",
    "exact Or.inl h",
    "exact Or.inr h",
    "exact And.intro h.2 h.1",
    "simp [List.append_assoc]",
    "simp_all",
    "exact False.elim h",
)
TRAIN_FAMILIES=(
    "arith_shift_le",
    "arith_add_comm",
    "arith_add_assoc",
    "arith_sub_cancel",
    "defeq_refl",
    "defeq_add_zero",
    "prop_assumption",
    "prop_and_intro",
    "prop_and_left",
    "prop_and_right",
    "prop_or_left",
)
DEV_FAMILIES=("prop_or_right","prop_and_swap")
TEST_FAMILIES=("nat_le_arith","nat_sub_arith","prop_and_nested","list_assoc")


def _family(family:str,k:int)->tuple[str,str]:
    """Return theorem telescope/goal and a *reference* tactic for source CI.

    A semantically irrelevant numeric context is present in every family.
    It varies deterministic problem instances without using task IDs or
    answer labels as features. A signature may have competing valid tactics;
    source labels denote one *known* successful tactic, not a unique proof.
    """
    variants={
        "arith_shift_le":(
            f"(a b : Nat) (hab : a ≤ b) : a + {k} ≤ b + {k}","omega"),
        "arith_add_comm":(
            "(a b : Nat) : a + b = b + a","omega"),
        "arith_add_assoc":(
            "(a b c : Nat) : a + (b + c) = (a + b) + c","omega"),
        "arith_sub_cancel":(
            f"(a b : Nat) : (a + b + {k}) - (a + {k}) = b","omega"),
        "defeq_refl":(
            "(x : Nat) : x = x","rfl"),
        "defeq_add_zero":(
            "(x : Nat) : x + 0 = x","rfl"),
        "prop_assumption":(
            "(P Q : Prop) (h : P) : P","assumption"),
        "prop_and_intro":(
            "(P Q : Prop) (hp : P) (hq : Q) : P ∧ Q",
            "constructor <;> assumption"),
        "prop_and_left":(
            "(P Q : Prop) (h : P ∧ Q) : P","exact h.1"),
        "prop_and_right":(
            "(P Q : Prop) (h : P ∧ Q) : Q","exact h.2"),
        "prop_or_left":(
            "(P Q : Prop) (h : P) : P ∨ Q","exact Or.inl h"),
        "prop_or_right":(
            "(P Q : Prop) (h : Q) : P ∨ Q","exact Or.inr h"),
        "prop_and_swap":(
            "(P Q : Prop) (h : P ∧ Q) : Q ∧ P",
            "exact And.intro h.2 h.1"),
        "nat_le_arith":(
            f"(a b c : Nat) (hab : a ≤ b) : c + a + {k} ≤ b + c + {k}",
            "omega"),
        "nat_sub_arith":(
            f"(a b : Nat) : (a + b + {k}) - (a + {k}) = b",
            "omega"),
        "prop_and_nested":(
            "(P Q R : Prop) (h : P ∧ (Q ∧ R)) : P","exact h.1"),
        "list_assoc":(
            "(xs ys zs : List Nat) : xs ++ (ys ++ zs) = (xs ++ ys) ++ zs",
            "simp [List.append_assoc]"),
    }
    if family not in variants:
        raise ValueError("unknown FC-10 grammar family: "+family)
    return variants[family]


def construct(family:str,split:str,index:int)->dict:
    if split not in ("train","dev","test"):
        raise ValueError("unsupported split")
    if index<0:
        raise ValueError("negative index")
    k=3+(index*7)%997
    suffix,tactic=_family(family,k)
    if tactic not in TACTICS:
        raise AssertionError("reference tactic absent from registered action space")
    # Add identical-shaped irrelevant context to all families, with
    # deterministic varied constants that are independent of tactic kind.
    # The entire goal state is produced by Lean, not this source text.
    context=f"(aux : Nat) (aux_le : aux ≤ aux + {k})"
    signature=context+" "+suffix
    canonical=" ".join(signature.split())
    digest=hashlib.sha256(canonical.encode("utf8")).hexdigest()
    return {
      "id":f"fc10_{split}_{index:04d}",
      "split":split,
      "family":family,
      "signature":signature,
      "reference_tactic":tactic,
      "signature_sha256":digest,
      "lean_version":LEAN_VERSION,
    }


def source_rows(split:str)->list[dict]:
    families={"train":TRAIN_FAMILIES,
              "dev":DEV_FAMILIES,
              "test":TEST_FAMILIES}[split]
    result=[]
    for family in families:
        for local in range(12):
            i=len(result)+({"train":0,"dev":132,"test":156}[split])
            result.append(construct(family,split,i))
    unique={r["signature_sha256"] for r in result}
    if len(unique)!=len(result):
        raise AssertionError("duplicate source Lean telescope/goal")
    assert len(result)=={"train":132,"dev":24,"test":48}[split]
    return result


def write_jsonl(path:Path,rows:list[dict])->str:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n"
                            for r in rows),encoding="utf8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_theorem(row:dict,*,tactic:str|None=None,
                   trace:bool=True)->str:
    """A single original Lean theorem with user-supplied tactic candidate."""
    name=row["id"]
    if not re.fullmatch(r"fc10_(?:train|dev|test)_\d{4}",name):
        raise ValueError("unexpected theorem identifier")
    proof=row["reference_tactic"] if tactic is None else tactic
    if proof not in TACTICS:
        raise ValueError("candidate outside preregistered tactic registry")
    parts=[f'theorem {name} {row["signature"]} := by']
    if trace:
        parts.extend([f'  trace "FC10_BEGIN:{name}"',
                      '  trace_state',
                      f'  trace "FC10_END:{name}"'])
    parts.append("  "+proof)
    return "\n".join(parts)


def lean_document(rows:list[dict],*,trace:bool=True,
                  tactic_map:dict[str,str]|None=None)->str:
    lines=["import Lean","", "set_option autoImplicit false",""]
    for item in rows:
        chosen=tactic_map.get(item["id"]) if tactic_map else None
        lines.extend([render_theorem(item,tactic=chosen,trace=trace),""])
    if rows:
        lines += [f"#print axioms {rows[0]['id']}",
                  f"#print axioms {rows[-1]['id']}"]
    return "\n".join(lines)+"\n"


def parse_actual_goal_states(raw_lean_output:str,expected:list[dict])->list[dict]:
    """Parse actual trace output. Reject missing/doubled Lean goal prints.

    Markers are emitted from inside the tactic elaborator immediately before
    trace_state and after it. Lean emits human-readable goal state between.
    This function only extracts the printed bytes and never reconstructs a
    goal from source signatures or reference tactic labels.
    """
    starts=re.compile(r"^FC10_BEGIN:(fc10_(?:train|dev|test)_\d{4})$")
    ends=re.compile(r"^FC10_END:(fc10_(?:train|dev|test)_\d{4})$")
    opened=None
    buffer=[]
    goal_by_id={}
    for line in raw_lean_output.splitlines():
        m=starts.fullmatch(line.strip())
        if m:
            if opened is not None or m.group(1) in goal_by_id:
                raise AssertionError("unexpected duplicate nested Lean goal trace")
            opened=m.group(1)
            buffer=[]
            continue
        m=ends.fullmatch(line.strip())
        if m:
            if opened!=m.group(1):
                raise AssertionError("missing or mismatched actual Lean END marker")
            state="\n".join(buffer).strip()
            if state.count("⊢")!=1 or not state:
                raise AssertionError(f"Lean did not emit a single goal: {opened}")
            goal_by_id[opened]=state
            opened=None
            buffer=[]
            continue
        if opened is not None:
            buffer.append(line)
    if opened is not None:
        raise AssertionError("unclosed Lean tactic state marker")
    expected_ids={r["id"] for r in expected}
    if set(goal_by_id)!=expected_ids:
        raise AssertionError("missing/extra actual Lean goal traces: "+
                             repr(sorted(expected_ids.symmetric_difference(goal_by_id))))
    result=[]
    for task in expected:
        result.append({
            "id":task["id"],
            "family":task["family"],
            "split":task["split"],
            "goal_state":goal_by_id[task["id"]],
            "goal_state_sha256":hashlib.sha256(goal_by_id[task["id"]].encode()).hexdigest(),
            "signature_sha256":task["signature_sha256"],
            # Store the reference label for training/scoring, never as features.
            "reference_tactic":task["reference_tactic"],
        })
    return result
