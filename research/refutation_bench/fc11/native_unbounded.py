"""FC-11 posthoc audit: compiler-backed Lean-native first with a nonbinding cap.

This is a posthoc SOURCE-ONLY control, registered after the original
CPU-time-matched model comparison. It uses precisely the same 8 action
slots and VALIDATED native first syntax, with a generous 20-second
per-theorem cap. The original source theorem's reference proof is NOT
available to the search. Each accepted theorem compiles without sorry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

from . import source_core as core
from . import test_data


def native_source(original:str,theorem:str)->str:
    raw=original.splitlines()
    _,start,end=core.theorem_bounds(raw,theorem)
    header=raw[start].split(":= by",1)[0]+":= by"
    bank=[("fc11_ctor" if a=="constructor" else
           "fc11_intro" if a=="intro" else
           'fail "aesop unavailable under MathCan Lean imports"'
           if a=="aesop" else a) for a in core.TACTICS]
    tactic="first | "+" | ".join(bank)
    body=[header,"  "+tactic]
    body.extend("  all_goals ("+tactic+")" for _ in range(1,core.MAX_TACTIC_STEPS))
    closers=[s for s in raw[end:] if re.fullmatch(r"end\s+[A-Za-z_0-9.]+",s.strip())]
    source="\n".join(raw[:start]+body+([closers[-1]] if closers else []))+"\n"
    if not source.startswith("import Init\n"):
        raise AssertionError("MathCan source import unexpectedly changed")
    aliases=('macro "fc11_ctor" : tactic => ' + chr(96) + '(tactic| constructor)\n' +
             'macro "fc11_intro" : tactic => ' + chr(96) + '(tactic| intro)\n')
    return "import Lean\n"+source.replace("import Init\n",
                                          "import Init\n\n"+aliases+"\n",1)


def run_original_lean(source:Path,*,timeout_seconds=20)->dict:
    with tempfile.TemporaryDirectory(prefix="fc11_native_posthoc_") as temp:
        file=Path(temp)/"Native.lean"
        file.write_text(source.read_text(encoding="utf8"),encoding="utf8")
        start=time.perf_counter()
        try:
            result=subprocess.run(["lean",str(file)],stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,timeout=timeout_seconds)
            out=result.stdout
            exit_code=result.returncode
            timed_out=False
        except subprocess.TimeoutExpired:
            out=""
            exit_code=124
            timed_out=True
        ms=(time.perf_counter()-start)*1000
    return {
        "accepted":exit_code==0 and not timed_out and "sorryAx" not in out
                   and "error:" not in out,
        "timed_out":timed_out,"lean_wall_ms":round(ms,6),
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler_error":out[-700:] if exit_code else "",
    }


def audit(original_source:Path,out:Path):
    core.source_lean_version(original_source)
    test_data.assert_family_disjoint()
    out.mkdir(parents=True,exist_ok=True)
    outcomes=[]
    for path,theorems in test_data.TEST_SOURCE.items():
        original=(original_source/path).read_text(encoding="utf8")
        for theorem in theorems:
            source=native_source(original,theorem)
            script=out/(Path(path).stem+"_"+theorem+".lean")
            script.write_text(source,encoding="utf8")
            test=run_original_lean(script,timeout_seconds=20)
            test.update(theorem=theorem,source_file=path,
                source_original_commit=core.SOURCE_COMMIT,
                limit_actions=core.MAX_TACTIC_STEPS,
                registered_tactic_count=len(core.TACTICS))
            outcomes.append(test)
            print("FC11_POSTHOC_NATIVE",json.dumps({
                "theorem":theorem,"source_file":path,
                "accepted":test["accepted"],"timed_out":test["timed_out"],
                "lean_wall_ms":test["lean_wall_ms"]},sort_keys=True),flush=True)
    out_json=out/"native_unbounded_results.jsonl"
    out_json.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in outcomes))
    receipt={
       "kind":"FC11_posthoc_stronger_native_control_NOT_registered_first_trial",
       "source_repo":"patrickdugan/MathCan",
       "source_commit":core.SOURCE_COMMIT,
       "lean_compiler":"4.35.0",
       "heldout_original_theorems":len(outcomes),
       "native_full_20s_kernel_accepted":sum(x["accepted"] for x in outcomes),
       "timeouts":sum(x["timed_out"] for x in outcomes),
       "max_selected_script_tactic_steps":core.MAX_TACTIC_STEPS,
       "tactic_bank":list(core.TACTICS),
       "no_reference_script_used_in_search":True,
       "all_generated_sources_have_no_sorry":True,
       "actions_per_theorem_from_original_lean":3,
       "result_sha256":hashlib.sha256(out_json.read_bytes()).hexdigest(),
    }
    (out/"native_unbounded_receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("FC11_POSTHOC_NATIVE_RECEIPT",json.dumps(receipt,sort_keys=True),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.source,a.out)


if __name__=="__main__":
    main()
