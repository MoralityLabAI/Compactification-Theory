"""Replay SMC-04 exact six-vertex frame identification without ML dependencies.

Standard run: 3 secrets * 60 paired trajectories for each of three
independent corruption rates (0, 10%, 25%); ~40 seconds on a CPU.
CI smoke: --smoke uses three trajectories per secret and smaller prefixes.
"""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from identify_transport import (
    uniform_catalog,hidden_fixture,calibration_cohort,
    hypothesis_agreement_histogram,expected_spurious,
    noisy_ambiguity_upper,paired_campaign,negative_calibration_control,
)


def encode(value):
    if isinstance(value,Fraction):
        return str(value)
    if isinstance(value,dict):
        return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [encode(x) for x in value]
    return value


def build_report(trials_per_seed=60,prefixes=(1,2,3,4,5,6,8,10,12,16,24)):
    c=uniform_catalog()
    truth=hidden_fixture(c,43)
    train,test=calibration_cohort(c,truth)
    full=hypothesis_agreement_histogram(c,truth)
    restricted=hypothesis_agreement_histogram(c,truth,train)
    checks=tuple(sorted(set(prefixes)|{12,24,32,48}))
    report={
        'schema':'smc04_finite_frame_identification_v1',
        'hypothesis_model':'graph = sigma * permutation * reference_matching',
        'n_matching_states':15,
        'n_permutations':720,
        'n_hypotheses':10800,
        'train_matching_classes':12,
        'test_matching_classes':3,
        'train_permutations':len(train),
        'test_permutations':len(test),
        'calibration':'iid train-cohort permutations with replacement; cohort is truth-conditioned',
        'corruption':'with probability epsilon, replace observed graph by independent uniform matching among all 15',
        'seeds':[43,77,91],
        'trials_per_seed':trials_per_seed,
        'histogram_full':full,
        'histogram_train':restricted,
        'expected_false_candidates_clean':{n:expected_spurious(restricted,n) for n in checks},
        'chernoff_union_bound':{
            str(eps):{n:noisy_ambiguity_upper(restricted,n,eps) for n in checks}
            for eps in (Fraction(0),Fraction(1,10),Fraction(1,4))},
        'campaigns':{
            str(eps):paired_campaign(seeds=(43,77,91),noise=eps,prefixes=prefixes,
                                     trials_per_seed=trials_per_seed)
            for eps in (Fraction(0),Fraction(1,10),Fraction(1,4))},
        'negative_control':negative_calibration_control(
            seeds=(43,77,91),examples=12,
            trials_per_seed=max(5,trials_per_seed//3)),
        'limitations':'Known candidate family S6 x Matchings6; only hidden parameters are learned. No real pretrained activations or SAE.',
    }
    return encode(report)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,default=Path('smc04-report.json'))
    args=p.parse_args()
    report=build_report(3 if args.smoke else 60,
        (1,3,5,8,12) if args.smoke else (1,2,3,4,5,6,8,10,12,16,24))
    report['source_sha256']={
        fname:hashlib.sha256(Path(__file__).with_name(fname).read_bytes()).hexdigest()
        for fname in ('identify_transport.py','run_smc04.py')}
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'report':str(args.output),'trials_per_seed':report['trials_per_seed'],
        'max_false_agreement':report['histogram_train']['max_false_agreement'],
        'expected_false_N12':report['expected_false_candidates_clean']['12']},indent=2))


if __name__=='__main__':
    main()
