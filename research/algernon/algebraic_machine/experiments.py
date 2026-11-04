"""Deterministic algebraic reference experiment; no GPU, checkpoints or training."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .machine import exhaustive_word_check, lifted_actions
from .sheaf import affine_cocycle, cycle_cohomology
from .sparse import sparse_repair, step_error_bound


def run():
    maps = list(itertools.product(range(3), repeat=3))
    counts = {"target_maps": len(maps), "program_checks": 0, "program_violations": 0}
    for t in maps:
        symbolic = (t, (1, 2, 0))
        hidden, pi = lifted_actions(symbolic)
        rep = exhaustive_word_check(hidden, symbolic, pi, 5)
        counts["program_checks"] += rep["program_checks"]
        counts["program_violations"] += rep["program_violations"]
    sheaf = {"scalar_cycle_cases": 0, "monodromy_dim_mismatches": 0,
             "examples": {}}
    for p in (2, 3, 5, 7):
        for a, b, c in itertools.product(range(p), repeat=3):
            s = cycle_cohomology([[[a]], [[b]], [[c]]], p)
            sheaf["scalar_cycle_cases"] += 1
            if s['H0_dimension'] != int((a*b*c)%p==1):
                sheaf["monodromy_dim_mismatches"] += 1
    sheaf['examples']['inconsistent_affine_3cycle_F5'] = affine_cocycle([[[1]]]*3, [[1]]*3, 5)
    sheaf['examples']['consistent_affine_3cycle_F5'] = affine_cocycle([[[1]]]*3, [[1],[1],[3]], 5)
    P=np.eye(3)
    A=np.eye(3)
    M=A.copy()
    M[0,2]=3
    support_good=np.array([[1.],[0.],[0.]])
    support_bad=np.array([[0.],[1.],[0.]])
    rgood=sparse_repair(P,M,A,support_good)
    rbad=sparse_repair(P,M,A,support_bad)
    sparse={"useful_support_residual":rgood['actual_residual_norm'],
            "orthogonal_support_residual":rbad['actual_residual_norm'],
            "useful_support_exact":rgood['exactly_repairable'],
            "orthogonal_support_exact":rbad['exactly_repairable']}
    transition=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    q=np.array([1.,0.,0.]);states=[q]
    for _ in range(8):
        q = .98*(transition@q)+.02*np.ones(3)/3
        states.append(q)
    bound=step_error_bound(transition,states,np.array([1.,0.,0.]))
    return {"schema_version":"0.1.0", "execution_class":"cpu_reference_only",
            "source_note":"No learned weights or LOTUS activations evaluated",
            "monoid":counts, "cyclic_sheaf":sheaf, "sparse_intertwining":sparse,
            "depth_bound":bound}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("runs/algernon/algebraic_machine/reference.json"))
    args=p.parse_args()
    result=run()
    raw=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(raw,encoding="utf-8")
    print(json.dumps({"results":str(args.out),"sha256":hashlib.sha256(raw.encode()).hexdigest(),
           "checks":result['monoid']['program_checks'],"sheaf_cases":result['cyclic_sheaf']['scalar_cycle_cases']}))

if __name__ == '__main__':
    main()
