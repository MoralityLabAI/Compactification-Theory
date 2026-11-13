"""SHAFT-COMP-22 pure-Python audit of frozen neural summary and finite reference."""
from __future__ import annotations
import json
from pathlib import Path

DEFAULT='research/shaft_compactification/results/comp22_summary.json'
METHODS=('direct','canonical','behavioral','oracle')


def verify(summary_path=DEFAULT):
    summary=json.loads(Path(summary_path).read_text())
    assert summary['experiment']=='SHAFT-COMP-22'
    assert summary['registration_commit']=='f539c28e0dc4409cc76741a48c5d39ba9531f9c0'
    assert summary['test_states']==16384
    assert summary['train_states']==2304
    assert len(summary['frozen_models'])==4
    assert len(summary['heads'])==4
    for depth in ('12','40'):
        cell=summary['pooled_by_depth'][depth]
        assert cell['states']==8192
        for method in METHODS:
            m=cell['methods'][method]
            assert 0<=m['native_exact_count']<=8192
            assert 0<=m['native_correct_bits']<=8192*36
            assert abs(m['native_bit_accuracy']-m['native_correct_bits']/(8192*36))<1e-6
            assert abs(m['native_exact_accuracy']-m['native_exact_count']/8192)<1e-6
            assert 0<=m['automaton_exact_count']<=8192
        assert cell['methods']['oracle']['automaton_exact_count']==8192
        for name in ('previous_accuracy','latch_accuracy'):
            assert 0 <= cell['latent_decoding'][name] <= 1
    primary=summary['primary']
    assert primary['positive_models']==4
    assert primary['bootstrap_draws']==1000
    observed=(summary['pooled_by_depth']['40']['methods']['canonical']['native_exact_count']-
              summary['pooled_by_depth']['40']['methods']['direct']['native_exact_count'])/8192
    assert abs(observed-primary['primary_canonical_minus_direct_exact_native_depth40'])<1e-6
    assert primary['bootstrap95'][0]>0
    assert primary['support_gate']
    assert len(summary['sources']['sha256'])==64
    return {'experiment':'SHAFT-COMP-22','verified':True,
            'primary_exact_difference':round(observed,6),
            'model_count':len(summary['frozen_models']),
            'mean_native_bit_accuracy_depth40':{
                k:summary['pooled_by_depth']['40']['methods'][k]['native_bit_accuracy']
                for k in METHODS}}


if __name__=='__main__':print(json.dumps(verify(),sort_keys=True,indent=2))
