"""Export Algernon-02 semantics and a deterministic CPU exactness receipt.

Invoke from the repo root:
python -m research.algernon.circuit_mirror.run --out runs/algernon/circuit_mirror
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

from .spec import PointerTask, build, canonical, case, digest, emit_circom, witness_for, witness_for_trace


def run_experiment(n: int = 4, steps: int = 8, cases_per_family: int = 48,
                   seed: int = 20261103):
    if cases_per_family < 1:
        raise ValueError("cases_per_family must be >=1")
    template=PointerTask(tuple(range(n)),0,steps)
    circuit=build(template)
    rows=[]
    for family in ("random_function", "permutation", "long_cycle", "sink"):
        for idx in range(cases_per_family):
            task=case(family,seed,idx,n=n,steps=steps)
            witness,public=witness_for(task,circuit)
            check=circuit.check(witness,public)
            # Independent direct-state oracle in PointerTask.trace; valid only if every step matches.
            if not check["valid"] or task.trace()[-1] != next(i for i in range(n) if witness[circuit.indices[f"final[{i}]"]] == 1):
                raise AssertionError("exact circuit and direct oracle disagree")
            rows.append({"family": family, "case":idx, "task_sha256":digest(task.description()),
                         "trace_sha256":digest(task.trace()), "r1cs_pass":check["valid"]})
    # A negative example that ends at the correct final state but lies on a false path.
    task=PointerTask(tuple((i+1)%n for i in range(n)),0,steps)
    path=task.trace()
    if steps > 1:
        broken=path.copy()
        broken[1]=(broken[1]+1)%n
        bad_w,bad_pub=witness_for_trace(task,circuit,broken)
        bad=circuit.check(bad_w,bad_pub)
        if bad["valid"]:
            raise AssertionError("incorrect internal trace accepted")
        bad_count=bad["violations"]
    else:
        bad_count=None
    return {
        "schema":"algernon-02-cpu-reference-v1",
        "execution_class":"python_r1cs_reference_only",
        "circom_compile_status":"not_run",
        "source_role":"independent CPU reference for future neural/compiled backends",
        "n_nodes":n,"steps":steps,"seed":seed,"cases_per_family":cases_per_family,
        "field":"BN254 Fr","modulus":str(21888242871839275222246405745257275088548364400416034343698204186575808495617),
        "signals":len(circuit.names),"constraints":len(circuit.constraints),
        "exact_checks":len(rows),"exact_failures":sum(not z["r1cs_pass"] for z in rows),
        "negative_corrupt_trace_violations":bad_count,
        "rows_sha256":digest(rows),"rows":rows,
    }, circuit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--n",type=int,default=4)
    parser.add_argument("--steps",type=int,default=8)
    parser.add_argument("--cases",type=int,default=48)
    parser.add_argument("--seed",type=int,default=20261103)
    parser.add_argument("--out",type=Path,default=Path("runs/algernon/circuit_mirror"))
    parser.add_argument("--require-circom",action="store_true",help="fail when compiler is unavailable")
    args=parser.parse_args()
    result,circuit=run_experiment(args.n,args.steps,args.cases,args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    src=emit_circom(args.n,args.steps)
    constrained=circuit.export_circom_constraints()
    files={
        f"pointer_{args.n}x{args.steps}.circom":src,
        "r1cs_reference_constraints.json":json.dumps(constrained,sort_keys=True,indent=2)+"\n",
        "r1cs_reference_signals.json":json.dumps({"signals":circuit.names,"public_names":circuit.public_names},indent=2)+"\n",
    }
    result["source_hashes_sha256"]={k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}
    # No compiler or proof tool assumed installed: this is an explicit unexecuted gate.
    result["circom_compile_status"]="available_unexecuted" if shutil.which("circom") else "binary_not_found"
    if args.require_circom and not shutil.which("circom"):
        raise RuntimeError("circom binary unavailable")
    for name,body in files.items():
        (args.out/name).write_text(body,encoding="utf-8")
    (args.out/"cpu_reference.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(f"{result['exact_checks']} exact reference cases, {result['exact_failures']} failures, "
          f"{result['constraints']} R1CS-like constraints, circom={result['circom_compile_status']}")
    print(args.out)


if __name__ == '__main__':
    main()
