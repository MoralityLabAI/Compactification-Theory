"""Lean4.35 source proof trajectory instrumentation over pinned MathCan.

Proof bodies are preserved. Added import Lean and three tracing tactics
before each original top-level tactic command. Compiler goal strings
are never reconstructed from source or reference tactic annotations.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SOURCE_COMMIT="c84e2025334e9a967c1297425a490f98d5b3d189"
TOOLCHAIN="leanprover/lean4:v4.35.0"
TACTICS=("omega","decide","rfl","simp_all",
         "aesop","assumption","intro","constructor")
MAX_TACTIC_STEPS=3
MAX_GOAL_COUNT=8


def sha(x:str)->str:
    return hashlib.sha256(x.encode("utf8")).hexdigest()


def source_lean_version(source:Path):
    actual=(source/"lean-toolchain").read_text().strip()
    if actual!=TOOLCHAIN:
        raise AssertionError("source toolchain mismatch: "+actual)


def theorem_bounds(lines:list[str],theorem:str)->tuple[int,int,int]:
    decl=re.compile(r"^theorem\s+"+re.escape(theorem)+r"\b")
    found=[i for i,s in enumerate(lines) if decl.match(s)]
    if len(found)!=1:
        raise AssertionError(f"expected exactly one theorem {theorem}: {found}")
    start=found[0]
    begin=None
    for i in range(start,min(len(lines),start+40)):
        if ":= by" in lines[i]:
            begin=i
            break
        if i>start and re.match(r"^(?:theorem|lemma|def|namespace|end)\b",lines[i]):
            break
    if begin is None:
        raise AssertionError("no source tactic proof: "+theorem)
    end=len(lines)
    for i in range(begin+1,len(lines)):
        if re.match(r"^(?:theorem|lemma|def|namespace|end|#print)\b",lines[i]) or \
           lines[i].startswith(("/--","/-!")):
            end=i
            break
    return start,begin,end


def original_steps(lines:list[str],begin:int,end:int)->list[tuple[int,str]]:
    stages=[]
    inline=lines[begin].split(":= by",1)[1].strip()
    if inline:
        stages.append((begin,inline))
    for i in range(begin+1,end):
        line=lines[i]
        if not line.startswith("  ") or line.startswith("   "):
            continue
        command=line[2:]
        if not command or command[0].isspace() or \
           command.startswith(("|","·","--","/-")):
            continue
        stages.append((i,command.rstrip()))
    if not stages:
        raise AssertionError("source proof has zero top-level tactic commands")
    return stages


def instrument(original:str,filename:str,theorems:tuple[str,...])->tuple[str,list[dict]]:
    if not original.startswith("import Init\n"):
        raise AssertionError("only original standalone import Init modules supported")
    lines=original.splitlines()
    replacements=[]
    manifest=[]
    stem=Path(filename).stem
    for theorem in theorems:
        _,begin,end=theorem_bounds(lines,theorem)
        for index,(line_no,command) in enumerate(original_steps(lines,begin,end)):
            mark=f"{stem}.{theorem}.{index}"
            manifest.append({
                "stage_id":mark,"source_file":filename,
                "theorem":theorem,"stage_index":index,
                "source_command":command,
                "source_command_sha256":sha(command),
                "source_line":line_no+1,
            })
            replacements.append((line_no,mark,command,line_no==begin))
    for line_no,mark,command,inline in sorted(replacements,key=lambda x:x[0],reverse=True):
        tokens=[
            f'  trace "FC11_BEGIN:{mark}"',
            "  trace_state",
            f'  trace "FC11_END:{mark}"',
        ]
        if inline:
            header=lines[line_no].split(":= by",1)[0]+":= by"
            lines[line_no:line_no+1]=[header,*tokens,"  "+command]
        else:
            lines[line_no:line_no]=tokens
    source="import Lean\n"+"\n".join(lines)+"\n"
    if len({r["stage_id"] for r in manifest})!=len(manifest):
        raise AssertionError("duplicated compiler marker")
    return source,manifest


def parse_compiler_trace(log:str,expected:list[dict])->list[dict]:
    starts=re.compile(r"^FC11_BEGIN:([A-Za-z_0-9.]+)$")
    ends=re.compile(r"^FC11_END:([A-Za-z_0-9.]+)$")
    opened=None
    buf=[]
    observed={}
    for line in log.splitlines():
        m=starts.fullmatch(line.strip())
        if m:
            if opened is not None or m.group(1) in observed:
                raise AssertionError("nested/duplicate trace marker")
            opened=m.group(1)
            buf=[]
            continue
        m=ends.fullmatch(line.strip())
        if m:
            if opened!=m.group(1):
                raise AssertionError("incorrect Lean compiler marker order")
            goal="\n".join(buf).strip()
            count=goal.count("⊢")
            if not goal or count<1 or count>MAX_GOAL_COUNT:
                raise AssertionError(f"wrong goal count for {opened}: {count}")
            observed[opened]=(goal,count)
            opened=None
            buf=[]
            continue
        if opened is not None:
            buf.append(line)
    if opened is not None:
        raise AssertionError("unterminated original tactic trace")
    ids={row["stage_id"] for row in expected}
    if set(observed)!=ids:
        raise AssertionError("missing original goal states: "+
                             repr(sorted(ids.symmetric_difference(observed))))
    result=[]
    for row in expected:
        goal,count=observed[row["stage_id"]]
        result.append({
            **row,"actual_lean_goal_state":goal,
            "goal_count":count,"actual_state_sha256":sha(goal),
            "original_mathcan_commit":SOURCE_COMMIT,
            "lean_toolchain":TOOLCHAIN,
            "is_original_reference_proof_prefix":True,
        })
    return result


def write_rows(path:Path,rows:list[dict])->str:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n"
                            for x in rows),encoding="utf8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit_source(source:Path,selected:dict[str,tuple[str,...]],out:Path,
                *,phase:str):
    source_lean_version(source)
    out.mkdir(parents=True,exist_ok=True)
    stages=[]
    files=[]
    for original_file,theorems in selected.items():
        original=(source/original_file).read_text(encoding="utf8")
        name=Path(original_file).stem
        instrumented,record=instrument(original,original_file,theorems)
        (out/(name+".lean")).write_text(instrumented,encoding="utf8")
        stages.extend(record)
        files.append({
            "file":original_file,"original_sha256":sha(original),
            "instrumented_sha256":sha(instrumented),
            "theorems":list(theorems),"stage_count":len(record),
            "instrumented_file":name+".lean",
        })
    digest=write_rows(out/"stage_manifest.jsonl",stages)
    return {
       "phase":phase,"source_repo":"patrickdugan/MathCan",
       "source_commit":SOURCE_COMMIT,"source_lean_version":TOOLCHAIN,
       "files":files,"theorems":sum(len(v) for v in selected.values()),
       "stages":len(stages),"stage_manifest_sha256":digest,
       "registered_tactics":list(TACTICS),
       "original_proof_bodies_modified":False,
    }


def extract_states(out:Path):
    stages=[json.loads(x) for x in (out/"stage_manifest.jsonl").read_text().splitlines()]
    by_file={}
    for row in stages:
        by_file.setdefault(row["source_file"],[]).append(row)
    states=[]
    for original_file,manifest in by_file.items():
        name=Path(original_file).stem
        log=(out/(name+".log")).read_text(encoding="utf8")
        if re.search(r"error:|sorryAx",log):
            raise AssertionError("Lean failed original proof under source instrumentation")
        states.extend(parse_compiler_trace(log,manifest))
    digest=write_rows(out/"actual_goal_states.jsonl",states)
    distribution=Counter(row["goal_count"] for row in states)
    return {
       "original_source_commit":SOURCE_COMMIT,
       "real_compiler_states":len(states),
       "goal_count_distribution":dict(distribution),
       "compiler_state_jsonl_sha256":digest,
       "compiled_original_theorem_proofs":True,
       "first_real_goal":states[0]["actual_lean_goal_state"][:600],
    }
