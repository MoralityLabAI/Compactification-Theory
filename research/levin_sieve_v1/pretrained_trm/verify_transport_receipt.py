"""Independent exact-rational audit of the LSV-16 serialized transport verdicts.

Relies on source-controlled Arb generation for the soundness of the positive
reported bounds. This checker independently validates identities, order,
threshold inequalities and all aggregates; it cannot reconstruct intermediate
radii without recomputing the pinned model with Arb.
"""
import argparse
from fractions import Fraction as Q
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def verify(rec,source,limits,complete=False):
    if rec.get('schema') != 'LSV-16/ambient-to-gauge-certified-transport-v1':
        raise ValueError('bad receipt schema')
    if (rec['checkpoint_sha256']!=source['checkpoint_sha256'] or
        rec['checkpoint_sha256']!=limits['checkpoint_sha256'] or
        rec['source_revision']!=source['revision'] or
        rec['roster_sha256']!=limits['roster_sha256']):
        raise ValueError('source lock mismatch')
    if rec['precision_bits']<128:
        raise ValueError('insufficient Arb precision')
    if (rec['intervention_geometry']!=
        'independent ambient l-infinity perturbation on all 3888 stage1 coordinates'):
        raise ValueError('unexpected intervention geometry')
    rs=rec['radii']
    if not rs or len(set(rs))!=len(rs) or any(Q(s)<=0 for s in rs):
        raise ValueError('bad radius roster')
    cases=rec['cases']
    if (not cases or len(cases)>64 or
        (complete and len(cases)!=64)):
        raise ValueError('invalid case count')
    totals={r:{'certified_winner':0,'certified_top2':0,'could_not_bound':0} for r in rs}
    for i,c in enumerate(cases):
        if c['case_index']!=i or set(c['ambient_radii'])!=set(rs):
            raise ValueError('missing or reordered case')
        ref=limits['case_thresholds'][i]
        if ref['case_index']!=i:
            raise ValueError('case threshold reorder')
        good1=Q(ref['winner_threshold_strict'])
        good2=Q(ref['top_two_threshold_strict'])
        if not (good1>0 and good2>0):
            raise ValueError('unexpected nonpositive LSV15 threshold')
        for r in rs:
            z=c['ambient_radii'][r]
            report=z['verified_final_transverse_radius_upper']
            if report is None:
                if z['winner_certificate'] or z['top_two_certificate'] or not z.get('reason'):
                    raise ValueError('invalid non-certificate')
                totals[r]['could_not_bound']+=1
                continue
            b=Q(report)
            if b<0:
                raise ValueError('invalid negative radius')
            if Q(z['LSV15_winner_threshold'])!=good1 or Q(z['LSV15_top2_threshold'])!=good2:
                raise ValueError('threshold substitution')
            w=b<good1; two=b<good2
            if (z['winner_certificate']!=w or z['top_two_certificate']!=two or
                two and not w):
                raise ValueError('false certificate')
            totals[r]['certified_winner']+=int(w)
            totals[r]['certified_top2']+=int(two)
    if totals!=rec['counts']:
        raise ValueError('aggregate mismatch')
    return totals


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('receipt',type=Path)
    ap.add_argument('--complete',action='store_true')
    args=ap.parse_args()
    source=json.loads((HERE/'source_lock.json').read_text())
    limits=json.loads((HERE/'results/lsv15_exact_thresholds_64_20261104.json').read_text())
    rec=json.loads(args.receipt.read_text())
    print(json.dumps(verify(rec,source,limits,complete=args.complete),sort_keys=True))

if __name__=='__main__':main()
