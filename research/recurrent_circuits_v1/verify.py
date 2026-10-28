"""Regenerate a deterministic exact-check receipt; never edits a remote repository."""
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import random
import unittest

from reference import Affine, apply_rows, finite_partition, minimal_affine_quotient, restored_nand

HERE = Path(__file__).resolve().parent
SOURCES = ('README.md','THEORY.md','EXPERIMENTS.md','reference.py','test_reference.py',
           'gpu_dram_bench.py','test_bench.py','verify.py')


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def receipt():
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.discover(str(HERE), pattern='test_*.py'))
    checked(result.wasSuccessful(), 'unit tests failed')
    rng = random.Random(20261029)
    state_checks = transition_checks = pair_checks = 0
    for trial in range(160):
        m = 1 + trial % 5
        maps = tuple(Affine(tuple(rng.randrange(1 << m) for _ in range(m)),
                            rng.randrange(1 << m)) for _ in range(trial % 4))
        outputs = tuple(rng.randrange(1 << m) for _ in range(trial % 3))
        q = minimal_affine_quotient(m, maps, outputs)
        blocks = finite_partition(m, maps, outputs)
        checked(1 << q.bits == len(set(blocks)), 'partition count mismatch')
        for x in range(1 << m):
            checked(q.readout(q.encode(x)) == apply_rows(outputs, x), 'readout mismatch')
            state_checks += 1
            for f,g in zip(maps,q.transitions):
                checked(q.encode(f(x)) == g(q.encode(x)), 'commutation mismatch')
                transition_checks += 1
            for y in range(1 << m):
                checked((q.encode(x)==q.encode(y)) == (blocks[x]==blocks[y]), 'minimality mismatch')
                pair_checks += 1
    nand_cases = 0
    for x,y in itertools.product((0,1),repeat=2):
        for ex,ey in itertools.product((Fraction(-1,8),0,Fraction(1,8)),repeat=2):
            checked(restored_nand(x+ex,y+ey) == 1-x*y, 'NAND margin failure')
            nand_cases += 1
    return dict(schema_version=1,evidence='exact_finite_reference',tests_run=result.testsRun,
                affine_systems=160,readout_checks=state_checks,transition_checks=transition_checks,
                pair_equivalence_checks=pair_checks,nand_margin_cases=nand_cases,
                nand_ring_initial_states=16,nand_ring_steps_per_state=128,
                gpu_execution=False,lean_kernel_check=False,model_training=False,
                source_sha256={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in SOURCES})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    p.add_argument('--check',type=Path)
    a=p.parse_args()
    if a.out and a.check:
        p.error('choose --out or --check')
    text=json.dumps(receipt(),indent=2,sort_keys=True)+'\n'
    if a.check:
        checked(a.check.read_text(encoding='utf-8')==text,'receipt/source mismatch')
        print('receipt verified')
    elif a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(text,encoding='utf-8')
    else:
        print(text,end='')


if __name__=='__main__': main()
