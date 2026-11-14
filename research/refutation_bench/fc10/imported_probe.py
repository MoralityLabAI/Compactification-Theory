"""Post-hoc FC-10 probe of *real existing* MathCan Lean theorem goal states.

This is an EXPLORATORY source-instrumentation experiment committed after the
original FC-10 preregistered model result. It does not alter the train/test
corpus, frozen checkpoints or first-run claims. Exact original MathCan repo
commit and Lean4 toolchain are pinned. Only added 'trace_state' messages
and an additional Lean import differ from original source files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

MATHCAN_COMMIT="c84e2025334e9a967c1297425a490f98d5b3d189"
LEAN_TOOLCHAIN="leanprover/lean4:v4.35.0"
FILES={
    "MathCan/FiniteHorizonGap.lean":(
        "epsilon_sixth_clearance",
        "epsilon_sixth_equiv",
        "full_horizon_recovers",
    ),
    "MathCan/DeletionMemory.lean":(
        "correct_decoder_separates",
    ),
    "MathCan/TOECosmology.lean":(
        "anchored_present",
        "curvature_mode_absorbed",
        "matter_mode_absorbed",
    ),
}


def instrument(original:str,targets:tuple[str,...],tag:str)->str:
    """Add a trace before the original tactic script, preserving its body.

    Every target theorem must be found exactly once and must have ':= by'
    before its next theorem. Replacement of an inline proof suffix only
    inserts a newline, retaining the same tactic expression.
    """
    if not original.startswith("import Init"):
        raise AssertionError("unexpected source import; freeze source before reuse")
    lines=("import Lean\n"+original).splitlines()
    for name in targets:
        declaration=re.compile(r"^\s*theorem\s+"+re.escape(name)+r"\b")
        at=[i for i,l in enumerate(lines) if declaration.match(l)]
        if len(at)!=1:
            raise AssertionError(f"expected one declaration named {name}")
        idx=at[0]
        header_line=None
        for j in range(idx,min(idx+30,len(lines))):
            if ":= by" in lines[j]:
                header_line=j
                break
            if j>idx and re.match(r"^\s*(?:theorem|lemma|def|namespace)\b",lines[j]):
                break
        if header_line is None:
            raise AssertionError("theorem missing by proof: "+name)
        old=lines[header_line]
        pre,suffix=old.split(":= by",1)
        marker=f"{tag}:{name}"
        new=[
            pre+":= by",
            f'  trace "FC10_IMPORTED_BEGIN:{marker}"',
            "  trace_state",
            f'  trace "FC10_IMPORTED_END:{marker}"',
        ]
        if suffix.strip():
            new.append("  "+suffix.strip())
        lines[header_line:header_line+1]=new
    return "\n".join(lines)+"\n"


def emit(source:Path,out:Path)->dict:
    toolchain=(source/"lean-toolchain").read_text().strip()
    if toolchain!=LEAN_TOOLCHAIN:
        raise AssertionError("MathCan Lean version changed")
    out.mkdir(parents=True,exist_ok=True)
    metadata=[]
    for path,targets in FILES.items():
        input_file=source/path
        original=input_file.read_text(encoding="utf8")
        tag=Path(path).stem
        instrumented=instrument(original,targets,tag)
        output_file=out/(tag+".lean")
        output_file.write_text(instrumented,encoding="utf8")
        metadata.append({
            "source_path":path,"source_sha256":hashlib.sha256(original.encode()).hexdigest(),
            "instrumented_name":output_file.name,
            "instrumented_sha256":hashlib.sha256(instrumented.encode()).hexdigest(),
            "theorems":list(targets)
        })
    receipt={
        "status":"posthoc exploratory original research theorem goal-state probe",
        "source_repo":"patrickdugan/MathCan",
        "source_commit":MATHCAN_COMMIT,
        "source_lean_toolchain":LEAN_TOOLCHAIN,
        "source_files":metadata,
        "theorems_to_probe":sum(len(x) for x in FILES.values()),
        "model_training_or_test_access":False,
        "original_fc10_model_result_unchanged":True,
    }
    (out/"imported_probe_manifest.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def extract(out:Path)->dict:
    result=[]
    for source_name,targets in FILES.items():
        tag=Path(source_name).stem
        log=(out/(tag+".log")).read_text(encoding="utf8")
        start=re.compile(r"^FC10_IMPORTED_BEGIN:("+re.escape(tag)+r":[A-Za-z_0-9]+)$")
        end=re.compile(r"^FC10_IMPORTED_END:("+re.escape(tag)+r":[A-Za-z_0-9]+)$")
        found={}
        opened=None
        buffer=[]
        for line in log.splitlines():
            m=start.fullmatch(line.strip())
            if m:
                if opened is not None or m.group(1) in found:
                    raise AssertionError("nested/duplicate imported goal marker")
                opened=m.group(1)
                buffer=[]
                continue
            m=end.fullmatch(line.strip())
            if m:
                if opened!=m.group(1):
                    raise AssertionError("mismatched imported theorem state END")
                state="\n".join(buffer).strip()
                if state.count("⊢")!=1:
                    raise AssertionError("expected one real Lean imported goal: "+opened)
                found[opened]=state
                opened=None
                continue
            if opened is not None:
                buffer.append(line)
        if opened is not None or set(found)!={f"{tag}:{n}" for n in targets}:
            raise AssertionError("compiler did not emit all imported theorem states")
        if re.search(r"error:|sorryAx",log):
            raise AssertionError("imported proof compiler emitted an error/sorry")
        for theorem in targets:
            key=f"{tag}:{theorem}"
            s=found[key]
            result.append({
                "source_commit":MATHCAN_COMMIT,
                "source_file":source_name,
                "theorem":theorem,
                "actual_lean_goal_state":s,
                "goal_state_sha256":hashlib.sha256(s.encode()).hexdigest(),
                "not_part_of_original_fc10_test":True,
            })
    f=out/"imported_goal_states.jsonl"
    f.write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in result))
    return {
        "source_repo":"patrickdugan/MathCan",
        "source_commit":MATHCAN_COMMIT,
        "lean_toolchain":LEAN_TOOLCHAIN,
        "real_imported_goal_states":len(result),
        "state_jsonl_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),
        "proofs_compiled_in_original_source_context":True,
        "research_scope":"posthoc external theorem-family probe, not preregistered FC10 generalization",
        "first_goal_preview":result[0]["actual_lean_goal_state"][:700],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--extract",action="store_true")
    a=p.parse_args()
    if a.extract:
        output=extract(a.out)
    else:
        if a.source is None:
            raise ValueError("source is required in instrument mode")
        output=emit(a.source,a.out)
    print(json.dumps(output,indent=2,sort_keys=True,ensure_ascii=False))


if __name__=="__main__":
    main()
