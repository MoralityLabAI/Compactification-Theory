"""QO-01 deterministic, source-hashed, compare-or-fail finite verification receipt."""
import argparse
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path

from research.ingression_dynamics_v1.goal_aware_partition import (
    fixture, deterministic_fixture, minimum_goal_partition, optimal_rate,
    pareto_frontier, partition_bits, partitions, goal_worst_error,
)

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'research/ingression_dynamics_v1/goal_aware_partition.py',
    'research/ingression_dynamics_v1/test_goal_aware_partition.py',
    'research/ingression_dynamics_v1/GOAL_AWARE_RATE_DISTORTION.md',
    'scripts/verify_goal_aware_partition.py',
)
DEFAULT = ROOT / 'results/goal_aware_quotient/receipt.json'


def build_receipt():
    states, goals, kernels = fixture()
    ds, dg, da = deterministic_fixture()
    quotient, rounds = minimum_goal_partition(ds, da, dg)
    entries = []
    for r in pareto_frontier(states, goals, kernels):
        entries.append({
            'partition': list(r['partition']),
            'abstract_states': r['classes'],
            'mutable_bits': r['bits'],
            'tv_defect': str(r['tv_error']),
            'mean_goal_error': str(r['goal_mean']),
        })
    constrained = {}
    for bits in range(4):
        sol = optimal_rate(states, goals, kernels, bits, Q(0))
        constrained[str(bits)] = None if sol is None else {
            'tv_defect': str(sol[0]), 'partition': list(sol[1])
        }
    assert [(e['mutable_bits'],e['tv_defect'],e['mean_goal_error']) for e in entries] == [
        (0,'0','1/5'), (1,'3/10','0'), (2,'1/10','0'), (3,'0','0')
    ]
    assert constrained['0'] is None
    assert [constrained[str(i)]['tv_defect'] for i in (1,2,3)] == ['3/10','1/10','0']
    assert len(list(partitions(states))) == 52
    assert sum(goal_worst_error(states,p,goals)==0 for p in partitions(states)) == 15
    assert partition_bits(quotient)==2 and rounds==2
    return {
        'contract':'QO-01 synthetic five-state stochastic and six-state deterministic quotient; CPU only',
        'evidence':'exact finite arithmetic; no biological/neural model measurements',
        'deterministic':{'micro_states':len(ds),'quotient_labels':list(quotient.values()),
                         'mutable_bits':2,'strict_refinements':rounds},
        'stochastic':{'micro_states':len(states),'partitions_enumerated':52,
                      'goal_visible_partitions':15,'pareto_frontier':entries,
                      'goal_exact_rate_curve':constrained},
        'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true',help='compare bytes to committed receipt')
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args = parser.parse_args()
    encoded = (json.dumps(build_receipt(),indent=2,sort_keys=True)+'\n').encode('utf-8')
    if args.check:
        assert args.output.exists(),f'missing receipt: {args.output}'
        assert args.output.read_bytes() == encoded, 'receipt differs from current sources/result'
        print(f'QO-01 verified: {args.output}')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists() and args.output.read_bytes() != encoded:
            raise FileExistsError('refuse to overwrite an inconsistent existing receipt')
        args.output.write_bytes(encoded)
        print(f'QO-01 source-bound receipt created: {args.output}')


if __name__ == '__main__':
    main()
