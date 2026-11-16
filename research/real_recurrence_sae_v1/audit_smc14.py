"""SMC-14 independent standard-library audit of heldout causal transfer.

Recomputes calibration-only global signs, each heldout puzzle-centered
effect cosine, deterministic within-puzzle shuffled nulls, all puzzle-level
bootstrap intervals, and registered pass/fail gates. It also performs an
EXPLORATORY *evaluation-only* null conditioned on hidden correct digit and
coarse row band. That last diagnostic was designed after the initial pilot
and is not allowed to alter the preregistered SMC-14 gate.

No neural checkpoint, PyTorch, model fitting or Internet is required.
"""
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

CHECKPOINT_SHA='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'
ORIGINAL_SHA='7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087'
INDEPENDENT_SHA='1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3'
CONDITIONED_SHA='3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07'
FAMILIES=('test_novel_30','test_novel_34','test_canonical_30')
ASSIGNMENTS=('matched','clue_shuffled','random')
PAIRS=(
 ('pretrainedA_71','newB_501_SAE_71'),
 ('pretrainedA_71','newB_501_SAE_89'),
 ('pretrainedA_71','newB_701_SAE_71'),
 ('pretrainedA_71','newB_701_SAE_89'),
 ('newB_501_SAE_71','newB_501_SAE_89'),
 ('newB_701_SAE_71','newB_701_SAE_89'),
 ('newB_501_SAE_71','newB_701_SAE_71'),
)
ORIGINAL_CALIB_SHA={
 'matched':(
  '1b0d9e04fb211afdc547c4d5acf1c544ca50e614a2ae9f691666626494836e26',
  '18b25d7ef79e8a53d55cc64c92445dd196bd5abc220cb199d35c1a23257d890f',
  '6ac04e14f1626e1c01c2d9ee8434cfd9f8570ecb08bbe32b1fd829499f2f4a34',
  'd2e664079da872743f2c7f8b90d4cd96abcf0aae2a9020aab15af9b9b27d9954',
  '5c7771e467a386c3e4df49c44ff73d0f01de8e14d8ce45335554ca153f02a65c',
  '5c6e67b3814fd04b2c5c1281cdfdb87835946644caed681a42b9a51509bfe623',
  '40a12fae20c30f9a43f11d631cc00c7657becd12202681f0f1c1dc14b5ab6e63',
 ),
 'clue_shuffled':(
  'b8c06d99700cdcd5bed5d4b57b642471810344cd560d9d1042bcf2fb2e27393d',
  'fcf297dd842d69e60372e7c024de3d8b79bd1c8fc597caf8c2eca5befdd9d614',
  '973f1ab7d1e43fd971cb257c59daf2ff1ff231badba1794c238e5ba4ef0ee817',
  '0c05b457ad846f63f08228bc5a052ef7cdbcecd34954ed395a733a8853e4999b',
  'beeef8c43498e9710d32535a4b8ffafd24bdd48d197edbe0cb08de13f8649583',
  '9d670432f04cbb7af25a630dd9a5e92d441cb2e970b29c79726ba1a379e7538b',
  'cac3a7391efad24024afcdb6a76a18897f88a4854032cd74424ad52c20a77da9',
 )
}


def mean(values):
    return sum(values)/len(values)


def cosine(a,b,mask=None):
    if len(a)!=len(b):
        raise ValueError('paired token dimensions mismatch')
    if mask is None:mask=[True]*len(a)
    if len(mask)!=len(a):
        raise ValueError('invalid selected-top1 mask')
    xs=[float(x) for x,valid in zip(a,mask) if valid]
    ys=[float(y) for y,valid in zip(b,mask) if valid]
    if len(xs)<3:return None
    ax=mean(xs);by=mean(ys)
    aa=[x-ax for x in xs];bb=[y-by for y in ys]
    denom=math.sqrt(sum(t*t for t in aa)*sum(t*t for t in bb))
    return sum(x*y for x,y in zip(aa,bb))/denom if denom>1e-14 else None


def calibration_sign(source,target,mask):
    if not (len(source)==len(target)==len(mask)):
        raise ValueError('calibration puzzle alignment mismatch')
    sum_inner=0.;usable=0
    for aa,bb,v in zip(source,target,mask):
        a=[float(x) for x,t in zip(aa,v) if t]
        b=[float(x) for x,t in zip(bb,v) if t]
        if len(a)<3:continue
        ac=mean(a);bc=mean(b)
        sum_inner+=sum((x-ac)*(y-bc) for x,y in zip(a,b))
        usable+=1
    return {'sign':1 if sum_inner>=0 else -1,
            'unidentified':sum_inner==0,
            'usable_calibration_puzzles':usable,
            'total_centered_inner':sum_inner}


def bootstrap(vals,seed,draws=2000):
    rng=random.Random(seed);n=len(vals)
    if n==0:return None
    values=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    return {'mean':mean(vals),'puzzle_bootstrap95':[
        values[int(.025*draws)],values[int(.975*draws)-1]],
        'n_independent_puzzles':n,'resamples':draws}


def deterministic_shuffle_null(a,b,mask,orientation,pair_index,puzzle_index,draws=8):
    rng=random.Random(202614000+pair_index*1009+puzzle_index)
    values=[]
    for _ in range(draws):
        order=list(range(len(b)));rng.shuffle(order)
        s=cosine(a,[orientation*b[j] for j in order],mask)
        if s is not None:values.append(s)
    return mean(values) if values else None


def close(x,y,tolerance=2e-9):
    if x is None or y is None:return x is None and y is None
    return abs(x-y)<=tolerance


def verify_puzzle(row,orient,pair_index):
    valid=row['source_strict_top1_selection_mask']
    if (len(valid)!=8 or len(row['selected_goal_positions'])!=8 or
        len(row['selected_goal_truth_digits'])!=8 or
        len(set(row['selected_goal_positions']))!=8):
        raise ValueError('missing original eight frozen goal tokens')
    if any(not 0<=p<81 for p in row['selected_goal_positions']):
        raise ValueError('invalid physical goal position')
    if any(not 1<=digit<=9 for digit in row['selected_goal_truth_digits']):
        raise ValueError('invalid correct Sudoku target digit')
    if row['n_valid_source_top_features']!=sum(bool(v) for v in valid):
        raise ValueError('source feature abstention count inconsistent')
    data={
      'matched':row['target_matched_causal_response'],
      'clue_shuffled':row['target_clue_shuffled_causal_response'],
      'random':row['target_random_assignment_causal_response'],
    }
    label_fields={
      'matched':'matched_effect_cosine',
      'clue_shuffled':'within_clue_shuffled_effect_cosine',
      'random':'random_feature_assignment_effect_cosine',
    }
    for label in ASSIGNMENTS:
        b=[orient[label]['sign']*v for v in data[label]]
        actual=cosine(row['source_causal_response'],b,valid)
        if not close(actual,row[label_fields[label]]):
            raise ValueError('heldout causal effect cosine differs from original')
    inside=deterministic_shuffle_null(
        row['source_causal_response'],data['matched'],valid,
        orient['matched']['sign'],pair_index,row['puzzle_index'])
    if not close(inside,row['within_puzzle_shuffled_goal_response_cosine']):
        raise ValueError('test-time puzzle-shuffled negative control mismatch')


def orientation_audit(pair):
    cal=pair['calibration']['source_causal_orientation_data']
    source=cal['source_response'];mask=cal['source_strict_mask']
    if not (len(source)==len(mask)==64):
        raise ValueError('wrong calibration puzzle count')
    for label in ASSIGNMENTS:
        computed=calibration_sign(source,cal['target_response_by_assignment'][label],mask)
        reported=pair['calibration']['orientation'][label]
        if (computed['sign']!=reported['sign']
            or computed['unidentified']!=reported['unidentified']
            or computed['usable_calibration_puzzles']!=reported['usable_calibration_puzzles']):
            raise ValueError('global orientation selected with wrong calibration')
        if not close(computed['total_centered_inner'],reported['total_centered_inner'],2e-7):
            raise ValueError('calibration orientation covariance mismatch')


def reported_summary_audit(rows,family,pair_index):
    reported=rows
    original=rows['per_puzzle']
    for row in original:
        verify_puzzle(row,rows['_orientation'],pair_index)
    valid=[x for x in original if not x['nonfinite_or_invalid'] and
           x['within_puzzle_shuffled_goal_response_cosine'] is not None]
    if len(valid)!=rows['valid_puzzles'] or len(original)!=rows['n_puzzles']:
        raise ValueError('puzzle count differs')
    keys={
       'matched':'matched_effect_cosine',
       'clue_shuffled':'within_clue_shuffled_effect_cosine',
       'random':'random_feature_assignment_effect_cosine',
       'within_goal_shuffle':'within_puzzle_shuffled_goal_response_cosine',
    }
    for name,field in keys.items():
        if not close(mean([x[field] for x in valid]),
                     rows['means_oriented_centered_causal_cosine'][name]):
            raise ValueError('mean causal response changed')
    for control in ('clue_shuffled','random','within_goal_shuffle'):
        delta=[x[keys['matched']]-x[keys[control]] for x in valid]
        computed=bootstrap(delta,20261400+pair_index*100+len(family)+len(control))
        prior=rows['paired_matched_minus_null'][control]
        if not close(computed['mean'],prior['mean']):
            raise ValueError('paired null difference mean changed')
        for a,b in zip(computed['puzzle_bootstrap95'],prior['puzzle_bootstrap95']):
            if not close(a,b):
                raise ValueError('bootstrap over puzzle clusters changed')
    gate=all(rows['paired_matched_minus_null'][control]['puzzle_bootstrap95'][0]>0
             for control in ('clue_shuffled','random'))
    if gate!=rows['primary_both_mapping_nulls_pass']:
        raise ValueError('registered primary transfer gate mismatch')


def digit_rowband_null(pair,family,pair_index,reps=16):
    """Posthoc evaluation-only null conditioned on true digit and row band.

    Preserves the target matched causal response marginal inside every
    (ground truth digit, row-band) group; breaks individual puzzle/token
    pairings by permuting target responses. This null is *not* used to
    fit a mapping, a global sign, a model, or the preregistered success gate.
    """
    group=pair['heldout'][family]
    original=group['per_puzzle']
    orientation=pair['calibration']['orientation']['matched']['sign']
    n=len(original)
    buckets=defaultdict(list)
    for p,row in enumerate(original):
        for i,(pos,digit) in enumerate(zip(
            row['selected_goal_positions'],row['selected_goal_truth_digits'])):
            buckets[(digit,pos//27)].append((p,i))
    tested=[len(v) for v in buckets.values()]
    if n!=64 or sum(tested)!=n*8:
        raise ValueError('conditioned null requires full independent test cohort')
    reference=[r['target_matched_causal_response'][:] for r in original]
    per_puzzle_values=[[] for _ in range(n)]
    rng=random.Random(202614717+pair_index*173+len(family))
    for _ in range(reps):
        shuffled=[row[:] for row in reference]
        for places in buckets.values():
            perm=places[:];rng.shuffle(perm)
            for (outp,outi),(inp,ini) in zip(places,perm):
                shuffled[outp][outi]=reference[inp][ini]
        for p,row in enumerate(original):
            v=cosine(row['source_causal_response'],
                     [orientation*z for z in shuffled[p]],
                     row['source_strict_top1_selection_mask'])
            if v is not None:per_puzzle_values[p].append(v)
    paired=[]
    null=[]
    for i,row in enumerate(original):
        if (row['matched_effect_cosine'] is None or
            len(per_puzzle_values[i])!=reps):
            continue
        shuffled_mean=mean(per_puzzle_values[i])
        paired.append(row['matched_effect_cosine']-shuffled_mean)
        null.append(shuffled_mean)
    return {
      'status':'posthoc_evaluation_only_null',
      'condition':'true Sudoku digit and row band (rows 0..2,3..5,6..8)',
      'calibration_mapping_and_sign_unchanged':True,
      'shuffle_reps_per_puzzle':reps,
      'groups':len(buckets),
      'group_size_range':[min(tested),max(tested)],
      'singleton_groups':sum(t==1 for t in tested),
      'tested_puzzles':len(paired),
      'matched_mean':mean([row['matched_effect_cosine'] for row in original
                            if row['matched_effect_cosine'] is not None]),
      'digit_rowband_shuffled_mean':mean(null) if null else None,
      'paired_matched_minus_digit_rowband_shuffle':
          bootstrap(paired,20261477+pair_index*100+len(family)) if paired else None,
      'interpretation':'Exploratory stronger confound control after SMC14 original results. This cannot alter the frozen preregistered primary gate.'
    }


def audit(receipt,posthoc=True):
    if receipt['schema']!='SMC-14/calibration-locked-cross-teacher-causal-transfer-v1':
        raise ValueError('unrecognized experiment receipt')
    if (receipt['source_teacher_sha256']!=CHECKPOINT_SHA or
        receipt['original_frozen_sae_receipt_sha256']!=ORIGINAL_SHA or
        receipt['independent_teacher_training_receipt_sha256']!=INDEPENDENT_SHA or
        receipt['conditioned_alignment_receipt_sha256']!=CONDITIONED_SHA):
        raise ValueError('source model/training/calibration provenance mismatch')
    if receipt['pilot_only'] or receipt['test_puzzles_per_cohort']!=64:
        raise ValueError('full 64-puzzle receipt needed')
    if (receipt['feature_calibration_puzzles']!=64 or
        receipt['goal_tokens_per_puzzle']!=8 or
        not close(receipt['intervention_radius_l2'],0.2)):
        raise ValueError('unregistered calibration or intervention parameters')
    if len(receipt['comparisons'])!=len(PAIRS):
        raise ValueError('comparison roster incomplete')
    total=0;posthoc_results=[]
    for pair_index,(pair,names) in enumerate(zip(receipt['comparisons'],PAIRS)):
        if (pair['source_component'],pair['target_component'])!=names:
            raise ValueError('cross-teacher pair sequence differs')
        orientation_audit(pair)
        for control in ('matched','clue_shuffled'):
            items=pair['calibration']['feature_assignments'][control]
            if sorted(items)!=list(range(96)):
                raise ValueError('96-feature assignment not a permutation')
            sha=hashlib.sha256(json.dumps(items).encode()).hexdigest()
            if sha!=ORIGINAL_CALIB_SHA[control][pair_index]:
                raise ValueError('frozen SMC13C mapping SHA mismatch')
        for family in FAMILIES:
            summary=dict(pair['heldout'][family])
            summary['_orientation']=pair['calibration']['orientation']
            if summary['status']!='measured':
                raise ValueError('no full measured result for registered family')
            reported_summary_audit(summary,family,pair_index)
            total+=len(summary['per_puzzle'])
            if posthoc:
                posthoc_results.append({
                   'source':pair['source_component'],'target':pair['target_component'],
                   'family':family,**digit_rowband_null(pair,family,pair_index)})
        primary=all(pair['heldout'][name]['primary_both_mapping_nulls_pass']
                    for name in ('test_novel_30','test_novel_34'))
        if bool(primary)!=pair['registered_success_on_both_novel_families']:
            raise ValueError('pair-level primary gate differs')
    return {
      'schema':'SMC-14/independent-replay-and-exploratory-digit-conditioned-null-v1',
      'original_mapping_provenance_verified':True,
      'calibration_only_orientation_recomputed':True,
      'per_puzzle_centered_response_and_negative_controls_recomputed':True,
      'puzzle_level_bootstrap_and_preregistered_gates_recomputed':True,
      'n_teacher_SAE_pairs':len(PAIRS),
      'n_full_heldout_puzzle_pair_family_cases':total,
      'test_families':list(FAMILIES),
      'posthoc_digit_rowband_controls':posthoc_results,
      'scope':'Independent standard-library replay of original FP64 neural intervention receipt; posthoc digit-conditioned shuffling is exploratory, not part of the predeclared primary gate.'
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/smc14/audit.json'))
    p.add_argument('--no-posthoc',action='store_true')
    a=p.parse_args()
    raw=a.receipt.read_bytes()
    result=audit(json.loads(raw),posthoc=not a.no_posthoc)
    result['original_full_receipt_sha256']=hashlib.sha256(raw).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({
      'out':str(a.out),'original_full_receipt_sha256':result['original_full_receipt_sha256'],
      'rechecked_puzzle_cases':result['n_full_heldout_puzzle_pair_family_cases'],
      'posthoc_digit_rowband':[
        {'pair':r['source']+' -> '+r['target'],
         'family':r['family'],
         'null_mean':round(r['digit_rowband_shuffled_mean'],5),
         'paired_delta':round(r['paired_matched_minus_digit_rowband_shuffle']['mean'],5),
         'bootstrap95':r['paired_matched_minus_digit_rowband_shuffle']['puzzle_bootstrap95']}
        for r in result['posthoc_digit_rowband_controls']
      ]
    },indent=2))


if __name__=='__main__':
    main()
