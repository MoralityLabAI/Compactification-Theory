"""Reproduce exact rational reference receipt; no GPU, seeds or external packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from fractions import Fraction as Q
from itertools import combinations, permutations
from pathlib import Path

from .model import Evidence, DiscreteOneForm, fair_bits_example, flat_one_form, matrix_rank_rational, entropy_bits


def run():
    evidence=Evidence((Q(1,10),Q(2,10),Q(3,10),Q(4,10)), (
        (Q(1,2),Q(1,4),Q(1,8),Q(1,16)),
        (Q(1,16),Q(1,4),Q(1,8),Q(1,2)),
        (Q(1,8),Q(1,8),Q(1,2),Q(1,16)),
        (Q(1,4),Q(1,16),Q(1,8),Q(1,2)),
    ))
    checks=sum(1 for n in range(5) for c in combinations(range(4),n)
               for order in permutations(c) if evidence.posterior(order)==evidence.posterior(c))
    total_orders=sum(1 for n in range(5) for c in combinations(range(4),n) for _ in permutations(c))
    assert checks==total_orders==65
    increments=((3,2,1),(-3,-1,-2),(1,1,3),(-1,-3,-2))
    g=flat_one_form(increments)
    faces=sum(1 for mask in range(1<<4) for i in range(4) for j in range(i+1,4)
              if not (mask&(1<<i) or mask&(1<<j)))
    assert faces==24 and g.max_curvature()==0
    prior,planner=fair_bits_example()
    values=planner.scores(prior,1)
    assert values=={'STOP':Q(1,2),'REVEAL_Y':Q(9,20),'NOISY_X':Q(7,10)}
    g_bad=flat_one_form(((1,), (2,), (3,), (4,)))
    edges=dict(g_bad.edges)
    edges[(1,1)]=(Q(9),)
    nonflat=DiscreteOneForm(4,1,edges)
    assert nonflat.max_curvature()==7
    random_gen=random.Random(20261104)
    n_trials=0
    for n in range(2,6):
        for case in range(20):
            d=DiscreteOneForm(n,2,{(mask,i):(Q(random_gen.randint(-4,4)),Q(random_gen.randint(-4,4)))
                for mask in range(1<<n) for i in range(n) if not mask & (1<<i)})
            perms=list(permutations(range(n)))
            for trial in range(20):
                a,b=random_gen.choice(perms),random_gen.choice(perms)
                assert d.path_bound(a,b)['valid']
                n_trials+=1
    return {
        'schema_version':'0.1.0',
        'evidence_class':'exact_rational_CPU_reference',
        'model_inference':False,
        'source_model':'research.algernon.belief_geometry',
        'evidence_orders_checked':checks,
        'square_faces_checked':faces,
        'exact_bayes_order_defects':0,
        'flat_hypercube_max_curvature':'0',
        'rank_of_log_likelihood_increments':matrix_rank_rational(increments),
        'nonflat_example_max_curvature':str(nonflat.max_curvature()),
        'random_noncircular_path_bound_checks':n_trials,
        'binary_decision_example':{
            'stop':str(values['STOP']),
            'reveal_irrelevant_Y':str(values['REVEAL_Y']),
            'noisy_relevant_X':str(values['NOISY_X']),
            'reveal_Y_mutual_information_bits':1.0,
            'noisy_X_mutual_information_bits':round(1-entropy_bits((Q(1,4),Q(3,4))),9),
            'continuation_advantage':str(planner.advantage(prior,1)),
            'decision':'NOISY_X'
        },
        'computational_limitations':[
            'No Qwen3.5-27B or other model was evaluated.',
            'No Silico or GPU resources used.',
            'Hypercube integrability applies to additive decoded edge increments over fixed evidence sets.',
            'Alpha-vector planner uses known likelihoods and payoff structure.',
            'Empirical activation geometry requires separate held-out causal tests.'
        ]
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    results=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(results,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(f"PASS: {results['evidence_orders_checked']} evidence orders; {results['square_faces_checked']} squares; "
          f"{results['random_noncircular_path_bound_checks']} path-bound checks; {args.out}")


if __name__=='__main__':
    main()
