"""COE-02 frozen evaluator-interface adapter for exact rational latent vectors.

This is an audit harness. It accepts fixed activation prototypes and evaluator
edges; it neither extracts checkpoints nor treats a synthetic prototype as
observed model behavior. Uses ell_1 distance of the supplied feature vectors.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

try:
    from .l1_cut_codes import minimum_l1_observer
    from .observer_codes import metric_quotient_observer
except ImportError:
    from l1_cut_codes import minimum_l1_observer
    from observer_codes import metric_quotient_observer


def analyse(case: dict) -> dict:
    if case.get('format') != 'coe02-activation-v1':
        raise ValueError('unsupported frozen activation case format')
    states = case['states']
    if not states or len(states) > 8:
        raise ValueError('small-n reference requires between one and eight states')
    names = [s['id'] for s in states]
    if any(not isinstance(x, str) or not x for x in names) or len(set(names)) != len(names):
        raise ValueError('state IDs must be unique nonempty strings')
    vectors = [tuple(Fraction(a) for a in s['activation']) for s in states]
    nfeatures = len(vectors[0])
    if nfeatures < 1 or any(len(x) != nfeatures for x in vectors):
        raise ValueError('all activation vectors must have a shared positive dimension')
    if len(set(vectors)) != len(vectors):
        raise ValueError('repeated activations make a genuine metric impossible')
    index = {name: i for i, name in enumerate(names)}
    try:
        edges = tuple((index[u], index[v]) for u, v in case['evaluator_edges'])
    except KeyError as e:
        raise ValueError('unknown state ID in evaluator edge') from e
    d = tuple(tuple(sum(abs(a-b) for a, b in zip(x,y)) for y in vectors)
              for x in vectors)
    L = Fraction(case['lipschitz'])
    if L <= 0:
        raise ValueError('activation comparison requires a strictly positive Lipschitz budget')
    r = case.get('hamming_margin',1)
    if not isinstance(r, int) or isinstance(r,bool) or r < 1:
        raise ValueError('positive integer Hamming margin required')
    l1 = minimum_l1_observer(d, edges, L, r)
    try:
        linf_width, linf_words, _ = metric_quotient_observer(d, edges, L, r)
        linf = {'width': linf_width, 'words': list(linf_words)}
    except ValueError as e:
        if 'forced-equivalent' not in str(e):
            raise
        linf = None
    return {
        'format':'coe02-activation-report-v1',
        'provenance': case.get('provenance','UNSPECIFIED: synthetic or user-supplied'),
        'state_count':len(states),'activation_width':nfeatures,
        'lipschitz':str(L),'margin':r,
        'metric':'input_feature_l1_exact_rational',
        'distance':[[str(v) for v in row] for row in d],
        'linf':linf,
        'l1':None if l1 is None else {'width':l1.width,
                                    'words':list(l1.words),
                                    'cut_masks':list(l1.cuts),
                                    'pair_capacities':list(l1.capacities),
                                    'pair_loads':list(l1.loads)}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    args=parser.parse_args()
    print(json.dumps(analyse(json.loads(args.case.read_text(encoding='utf-8'))),
                     indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
