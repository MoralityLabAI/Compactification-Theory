"""KSR-01C: source-locked recurrent checkpoint, finite branching assumptions.

The original RRC25 GRU is frozen. A label is the top-ranked left template
index (argmax(logits)//4). Each edge independently draws one of eight
calibration feature rows and applies a native GRU step. This continuation
is artificial and occurs beyond the original nine-row task input.

Exact Fraction laws are conditional on portable floating-point inference
and the registered finite root cohort. No hidden-CoT claim is made.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from research.rrc_compactification import bench25, compound25 as source, model25
try:
    from .dependence import (coarse_dependence, finite_map_floor,
        iid_histogram_power, symmetric_projection, total_variation)
except ImportError:
    from dependence import (coarse_dependence, finite_map_floor,
        iid_histogram_power, symmetric_projection, total_variation)

STATUS='FROZEN_RRC25_COARSE_STATE_ASSUMPTION_AUDIT'
SCHEMA='ksr01c.rrc25_channel.v1'
Q=4
B=2
CAL_PER=12
TEST_PER=24
BANK_SIZE=8
BASE_SEED=940229237


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),
                                      allow_nan=False).encode()).hexdigest()


def gru_step(h,row,checkpoint):
    if len(h)!=model25.HIDDEN or len(row)!=8:
        raise ValueError('native GRU input/state shape mismatch')
    s=checkpoint['state_dict']
    a=model25.affine(s['rnn.weight_ih_l0'],s['rnn.bias_ih_l0'],row)
    v=model25.affine(s['rnn.weight_hh_l0'],s['rnn.bias_hh_l0'],h)
    out=[]
    for i in range(model25.HIDDEN):
        reset=model25.sigmoid(a[i]+v[i])
        update=model25.sigmoid(a[i+model25.HIDDEN]+v[i+model25.HIDDEN])
        new=math.tanh(a[i+2*model25.HIDDEN]+reset*v[i+2*model25.HIDDEN])
        out.append((1-update)*new+update*h[i])
    if not all(math.isfinite(x) for x in out):
        raise ValueError('nonfinite native hidden state')
    return tuple(out)


def logits(h,checkpoint):
    s=checkpoint['state_dict']
    return model25.affine(s['head.weight'],s['head.bias'],h)


def state_label(h,checkpoint):
    values=logits(h,checkpoint)
    if len(values)!=16 or not all(math.isfinite(x) for x in values):
        raise ValueError('native head invalid')
    return max(range(16),key=lambda j:(values[j],-j))//4


def native_root(case,checkpoint):
    headers=source.possible_headers(case)
    if not headers:
        raise ValueError('source case has no public header')
    header=headers[0]
    rows=source.features(case,header)
    if len(rows)!=9 or any(len(row)!=8 for row in rows):
        raise ValueError('native nine-row public feature shape changed')
    h=(0.,)*model25.HIDDEN
    for row in rows:
        h=gru_step(h,row,checkpoint)
    expected=model25.logits(case,header,checkpoint)
    actual=logits(h,checkpoint)
    error=max(abs(a-b) for a,b in zip(expected,actual))
    if error>1e-12:
        raise AssertionError('native final-logit replay mismatch')
    return {'case_id':case.identifier,'family':case.family,
            'hidden':h,'label':state_label(h,checkpoint),
            'rows':rows,'native_logit_max_abs_error':error,
            'public_header_count':len(headers)}


def make_cohorts(cal_per=CAL_PER,test_per=TEST_PER):
    if (type(cal_per) is not int or type(test_per) is not int or
            not 1<=cal_per<=48 or not 1<=test_per<=64):
        raise ValueError('positive bounded cohort sizes required')
    seen=set(); result={}
    for name,families,count,seed0,stride in (
        ('calibration',source.TRAIN_FAMILIES,cal_per,BASE_SEED,1009),
        ('heldout',source.TEST_FAMILIES,test_per,BASE_SEED+1,1013)):
        group=[]
        for ix,family in enumerate(families):
            rng=random.Random(seed0+stride*ix)
            found=0
            for attempt in range(count*500):
                case=source.make_case(rng,family,f'ksr01c-{name}-{family}-{attempt:05d}')
                if case.visible in seen:
                    continue
                seen.add(case.visible); group.append(case); found+=1
                if found==count:
                    break
            if found!=count:
                raise RuntimeError('source cohort unique-input generation exhausted')
        result[name]=group
    a={c.visible for c in result['calibration']}
    if any(c.visible in a for c in result['heldout']):
        raise AssertionError('visible input leakage between cohorts')
    return result


def make_bank(cal_roots,size=BANK_SIZE):
    if type(size) is not int or not 1<=size<=16:
        raise ValueError('bank size must be 1..16')
    pool=[(r['case_id'],i,tuple(row)) for r in cal_roots for i,row in enumerate(r['rows'])]
    selected=random.Random(BASE_SEED+2).sample(pool,size)
    bank=tuple(row for _,_,row in selected)
    metadata=[{'case_id':cid,'native_row':i,'values':list(row)} for cid,i,row in selected]
    return bank,metadata


def row_law(labels):
    count=Counter(labels)
    return tuple(F(count[j],len(labels)) for j in range(Q))


def unit_histograms(row):
    return {tuple(int(j==s) for j in range(Q)):p for s,p in enumerate(row) if p}


def tree_laws(h,bank,checkpoint):
    """Exact depth-one/two histogram probabilities conditional on full h."""
    K=len(bank)
    if not K:
        raise ValueError('empty intervention bank')
    children=[gru_step(h,row,checkpoint) for row in bank]
    child_labels=[state_label(x,checkpoint) for x in children]
    one=row_law(child_labels)
    p2=[F() for _ in range(Q)]
    child_subtree=defaultdict(F)
    for child in children:
        grand=[state_label(gru_step(child,row,checkpoint),checkpoint) for row in bank]
        prob=row_law(grand)
        for j in range(Q):
            p2[j]+=prob[j]/K
        for count,p in iid_histogram_power(unit_histograms(prob),B).items():
            child_subtree[count]+=p/K
    depth_one=iid_histogram_power(unit_histograms(one),B)
    depth_two=iid_histogram_power(dict(child_subtree),B)
    for L,law in ((1,depth_one),(2,depth_two)):
        if sum(law.values())!=1 or any(sum(c)!=B**L for c in law):
            raise AssertionError('native branching histogram law invalid')
    return {'one_step':one,'two_step':tuple(p2),
            'counts':{1:depth_one,2:depth_two},
            'native_gru_steps':K+K*K}


def analyze_cohort(roots,bank,checkpoint,calibrated=None):
    if not roots:
        raise ValueError('empty root cohort')
    N=len(roots)
    labels=[r['label'] for r in roots]
    trees=[tree_laws(r['hidden'],bank,checkpoint) for r in roots]
    dep=coarse_dependence(labels,[t['one_step'] for t in trees],Q)
    matrix=[r['transition'] for r in dep['by_parent_label']]
    sym=symmetric_projection(matrix)
    for entry in dep['by_parent_label']:
        if entry['witness'] is not None:
            entry['witness']['case_ids']=[roots[i]['case_id']
                                         for i in entry['witness']['parent_indices']]
    second=[]
    for a in range(Q):
        which=[i for i,c in enumerate(labels) if c==a]
        second.append(None if not which else [
            sum((trees[i]['two_step'][j] for i in which),F())/len(which)
            for j in range(Q)])
    closure={'available':False,'reason':'unobserved_parent_label'}
    if not dep['missing_parent_labels']:
        T=[[F(x) for x in row] for row in matrix]
        T2=[[sum(T[a][k]*T[k][j] for k in range(Q)) for j in range(Q)] for a in range(Q)]
        row_errors=[total_variation(T2[a],second[a]) for a in range(Q)]
        closure={'available':True,'max_row_TV':str(max(row_errors)),
                 'row_TV':[str(x) for x in row_errors],
                 'one_step_squared':[[str(x) for x in row] for row in T2]}
    calibrated_residual={'available':False}
    if calibrated and calibrated['available']:
        P=calibrated['matrix']
        present=[j for j,row in enumerate(matrix) if row is not None]
        residual=[total_variation(matrix[j],P[j]) for j in present]
        calibrated_residual={'available':True,'calibration_lambda':calibrated['lambda'],
            'occupied_rows':present,'row_TV':[str(x) for x in residual],
            'max_row_TV':str(max(residual))}
    scores=[]
    for L in (1,2):
        joint=defaultdict(F)
        for r,t in zip(roots,trees):
            for counts,p in t['counts'][L].items():
                joint[r['label'],counts]+=p/N
        metrics=finite_map_floor(dict(joint),Q)
        metrics.update(depth=L,observed_leaf_symbols=B**L,
                       live_counter_bits_upper=(Q-1)*(B**L).bit_length())
        scores.append(metrics)
    prior=[F(labels.count(j),N) for j in range(Q)]
    violations=[]
    if prior!=[F(1,Q)]*Q: violations.append('nonuniform_root_label_prior')
    if not dep['coarse_sibling_independence_exact']:
        violations.append('siblings_dependent_given_compressed_parent_label')
    if not sym['available']: violations.append('missing_parent_classes')
    elif F(sym['max_row_TV'])>0: violations.append('one_step_channel_not_symmetric')
    if closure['available'] and F(closure['max_row_TV'])>0:
        violations.append('one_step_channel_square_does_not_predict_two_steps')
    return {'root_count':N,'root_label_counts':[labels.count(j) for j in range(Q)],
            'root_prior':[str(x) for x in prior],
            'native_replay_max_abs_error':max(r['native_logit_max_abs_error'] for r in roots),
            'conditional_kernel_audit':dep,'symmetric_channel_projection':sym,
            'calibration_channel_test_residual':calibrated_residual,
            'actual_two_step_channel':[
                None if row is None else [str(x) for x in row] for row in second],
            'stationary_markov_composition_check':closure,
            'population_count_decoding':scores,
            'original_uniform_symmetric_KSR_floor_applied':False,
            'finite_assumption_violations':violations,
            'positive_asymptotic_neural_transfer_claim':False,
            'native_gru_steps_in_branch_enumeration':sum(t['native_gru_steps'] for t in trees),
            'root_witness_records':[{'case_id':r['case_id'],'family':r['family'],
                'label':r['label'],'hidden':list(r['hidden']),
                'one_step_child_law':[str(x) for x in t['one_step']],
                'two_step_child_law':[str(x) for x in t['two_step']]}
                for r,t in zip(roots,trees)]}


def run(cal_per=CAL_PER,test_per=TEST_PER,bank_size=BANK_SIZE):
    checkpoints,checkpoint_hashes=bench25.checkpoints()
    checkpoint=checkpoints['recurrent']
    if checkpoint['parameters']!=1000:
        raise AssertionError('original recurrent parameter contract changed')
    cases=make_cohorts(cal_per,test_per)
    cal=[native_root(c,checkpoint) for c in cases['calibration']]
    test=[native_root(c,checkpoint) for c in cases['heldout']]
    bank,bank_meta=make_bank(cal,bank_size)
    calibration=analyze_cohort(cal,bank,checkpoint)
    heldout=analyze_cohort(test,bank,checkpoint,calibration['symmetric_channel_projection'])
    return {'schema':SCHEMA,'status':STATUS,'checkpoint_sha256':checkpoint_hashes['recurrent'],
            'checkpoint_training_source_hashes':checkpoint['source_sha256'],
            'original_bench_source_hashes':bench25.hashes(),
            'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dependence_sha256':hashlib.sha256(Path(__file__).with_name('dependence.py').read_bytes()).hexdigest(),
            'q':Q,'b':B,'depths':[1,2],'calibration_per_family':cal_per,
            'heldout_per_family':test_per,'bank_size':bank_size,
            'input_bank':bank_meta,'input_bank_sha256':digest(bank_meta),
            'distinct_input_bank_row_values':len(set(bank)),
            'source_roster_sha256':{k:source.fingerprint(v) for k,v in cases.items()},
            'public_visible_cohorts_disjoint':True,
            'root_label_definition':'highest-ranked native left-template U = argmax(logits)//4',
            'continuation_is_native_training_distribution':False,
            'edge_inputs_independent_given_full_hidden_state':True,
            'input_bank_fit_uses_heldout_roots':False,
            'model_weights_modified':False,'original_Q5_weights_evaluated':False,
            'precision':'original portable scalar float64 replay; exact rational discrete laws',
            'calibration':calibration,'heldout':heldout}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    report=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    compact={'status':report['status'],'checkpoint_sha256':report['checkpoint_sha256'],
             'bank_sha256':report['input_bank_sha256'],'cohorts':{}}
    for name in ('calibration','heldout'):
        r=report[name]
        compact['cohorts'][name]={
            'n':r['root_count'],'root_counts':r['root_label_counts'],
            'replay_error':r['native_replay_max_abs_error'],
            'sibling_dependence':r['conditional_kernel_audit']['weighted_sibling_diagonal_excess'],
            'symmetric_fit':r['symmetric_channel_projection'],
            'closure':r['stationary_markov_composition_check'],
            'violations':r['finite_assumption_violations'],
            'root_recovery':[{'L':s['depth'],'prior':s['majority_prior'],
                'robust_floor':s['covariance_floor_float'],'MAP':s['count_MAP_float'],
                'counter_bits':s['live_counter_bits_upper']} for s in r['population_count_decoding']]}
    print(json.dumps(compact,indent=2,sort_keys=True))
    return 0


if __name__=='__main__': raise SystemExit(main())
