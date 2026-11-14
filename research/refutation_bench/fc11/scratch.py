"""FC-11 scratch tactic application and real Lean prefix-state probes.

A scratch probe may insert sorry ONLY to expose the next goal-state text
after a candidate tactic. Such a run is explicitly UNTRUSTED. A positive
certificate is released exclusively after separately compiling the
ORIGINAL theorem with the chosen action sequence and NO sorry.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time

from . import source_core as core


def make_source(original:str,theorem:str,actions:list[str],*,
                original_stage:int|None=None,probe:bool=False)->str:
    """Replace one theorem tactic script, leaving other source declarations.

    original_stage is for training-only expert-prefix probes: reproduce
    exactly the original earlier commands through the stage boundary.
    For inference, omit original_stage; the input is the model-generated
    action list alone. Original future source reference tactics never enter
    a candidate model proof.
    """
    if any(x not in core.TACTICS for x in actions):
        raise ValueError("candidate must be one of the eight preregistered scripts")
    raw=original.splitlines()
    _,begin,end=core.theorem_bounds(raw,theorem)
    prefix=[]
    if original_stage is not None:
        stages=core.original_steps(raw,begin,end)
        if not 0<=original_stage<len(stages):
            raise ValueError("invalid source-only teacher stage")
        first=stages[original_stage][0]
        if first>begin:
            prefix=raw[begin+1:first]
    header=raw[begin].split(":= by",1)[0]+":= by"
    body=[header,*prefix,*["  "+a for a in actions]]
    if probe:
        body += [
            "  all_goals",
            '    trace "FC11_AFTER_BEGIN"',
            "    trace_state",
            '    trace "FC11_AFTER_END"',
            "  all_goals sorry",
        ]
    source=raw[:begin]+body+raw[end:]
    return "import Lean\n"+"\n".join(source)+"\n"


def _extract_after_states(log:str)->list[str]:
    opened=False
    buf=[]
    goals=[]
    for line in log.splitlines():
        if line.strip()=="FC11_AFTER_BEGIN":
            if opened:
                raise AssertionError("nested scratch compiler trace")
            opened=True
            buf=[]
            continue
        if line.strip()=="FC11_AFTER_END":
            if not opened:
                raise AssertionError("mismatched scratch compiler trace")
            text="\n".join(buf).strip()
            if text.count("⊢")<1:
                raise AssertionError("scratch compiler emitted no actual goal")
            goals.append(text)
            opened=False
            continue
        if opened:
            buf.append(line)
    if opened:
        raise AssertionError("incomplete scratch compiler trace")
    return goals


def compile_candidate(original:str,theorem:str,actions:list[str],*,
                      original_stage:int|None=None,
                      probe:bool=False,
                      lean_executable:str="lean",
                      timeout_seconds:int=20)->dict:
    source=make_source(original,theorem,actions,
                       original_stage=original_stage,probe=probe)
    with tempfile.TemporaryDirectory(prefix="fc11_lean_") as temp:
        path=Path(temp)/"Candidate.lean"
        path.write_text(source,encoding="utf8")
        start=time.perf_counter()
        try:
            process=subprocess.run([lean_executable,str(path)],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,timeout=timeout_seconds)
            stdout=process.stdout
            returncode=process.returncode
            timed_out=False
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"")
            if isinstance(stdout,bytes):
                stdout=stdout.decode("utf8",errors="replace")
            returncode=124
            timed_out=True
        duration_ms=(time.perf_counter()-start)*1000
    accepted=(returncode==0 and
              (probe or ("sorryAx" not in stdout and
                         "error:" not in stdout)))
    new_states=_extract_after_states(stdout) if accepted and probe else []
    return {
        "accepted":accepted,
        "compiler_returncode":returncode,
        "timed_out":timed_out,
        "scratch_probe_with_sorry":probe,
        "closed_without_trusted_assumptions":bool(accepted and not probe),
        "no_goal_left_in_probe":bool(accepted and probe and not new_states),
        "real_remaining_goal_states":new_states,
        "real_remaining_goal_count":sum(x.count("⊢") for x in new_states),
        "lean_wall_ms":round(duration_ms,6),
        "source_sha256":hashlib.sha256(source.encode("utf8")).hexdigest(),
        "compiler_error_excerpt":"" if accepted else stdout[-500:],
    }


def action_effect(original:str,theorem:str,action:str,
                  stage_index:int,goal_before:str)->dict:
    result=compile_candidate(original,theorem,[action],
                             original_stage=stage_index,probe=True)
    after="\n".join(result["real_remaining_goal_states"])
    result["progress"]=bool(result["accepted"] and
                            (result["no_goal_left_in_probe"] or
                             after.strip()!=goal_before.strip()))
    # This is a training-only tactic-probe diagnostic, never a certified
    # proof count. A scratch compiler may contain sorryAx.
    result["training_use_only"]=True
    return result
