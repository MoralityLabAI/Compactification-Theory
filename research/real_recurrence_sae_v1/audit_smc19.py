"""SMC-19 independent standard-library auditor for refit causal stability.

Checks original model/SAE source locks, all seven original conditioned feature
assignments, four new independent SAE checkpoint *file-byte SHA256s*, original
calibration sign decisions, fresh Sudoku rosters, matched active controls,
physical dose invariance, puzzle-bootstrap statistical gates and orthogonal
gauge engineering tolerances. Neural checkpoint forward/SAE fitting is the
original source-locked PyTorch runner's trust boundary, not recomputed here.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from audit_smc14 import ORIGINAL_CALIB_SHA
from audit_smc15 import orient_replay,score_replay,validate_raw
from smc13_family import valid_solution,rectangle_count,count_solutions

PRECOMMIT='fd96967fbf251c884a8a705d6d8a68c0c2cddd2e'
FAMILIES={'smc19_novel_30':(202619030000,30),
          'smc19_novel_34':(202619040000,34)}
PAIRS=(('newB_501_SAE_71','newB_501_SAE_131'),
       ('newB_501_SAE_71','newB_501_SAE_167'),
       ('newB_701_SAE_71','newB_701_SAE_131'),
       ('newB_701_SAE_71','newB_701_SAE_167'))
SOURCES={'source_receipt_sha256':'7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087',
         'independent_training_receipt_sha256':'1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3',
         'conditioned_mapping_receipt_sha256':'3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07'}


def digest(x):
    return hashlib.sha256(x).hexdigest()


def close(a,b,tol=2e-8):
    if a is None or b is None:
        return a is None and b is None
    return (math.isfinite(a) and math.isfinite(b) and
            abs(a-b)<=tol*max(1.,abs(a),abs(b)))


def fingerprint(rows):
    return digest(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())


def compare_dict(got,expected,scope):
    if set(got)!=set(expected):
        raise ValueError(scope+' keys changed')
    for key,a in got.items():
        b=expected[key]
        if isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b) or any(not close(x,y) for x,y in zip(a,b)):
                raise ValueError(scope+'/'+key+' numeric array differs')
        elif isinstance(a,float) or a is None:
            if not close(a,b):
                raise ValueError(scope+'/'+key+' float differs')
        elif a!=b:
            raise ValueError(scope+'/'+key+' field differs')


def audit_original(frozen):
    if frozen['pre_registration_commit']!=PRECOMMIT:
        raise ValueError('original SMC19 protocol commit changed')
    if frozen['schema']!='SMC-19/four-source-trained-refit-SAE-calibration-lock-v1':
        raise ValueError('incorrect SMC19 frozen refit artifact schema')
    for key,value in SOURCES.items():
        if frozen['source_original_receipts'][key]!=value:
            raise ValueError('prior original SHA source changed')
    maps=frozen['source_original_seven_mappings']
    if len(maps)!=7:
        raise ValueError('missing original seven source-conditioned mappings')
    for i,item in enumerate(maps):
        mapping=item['matched']
        if sorted(mapping)!=list(range(96)):
            raise ValueError('original Hungarian assignment not a permutation')
        expected=ORIGINAL_CALIB_SHA['matched'][i]
        if digest(json.dumps(mapping).encode())!=expected:
            raise ValueError('original matched 96-feature assignment SHA differs')
        if item['hashes']['matched_assignment_sha256']!=expected or item['hashes']['clue_shuffled_assignment_sha256']!=ORIGINAL_CALIB_SHA['clue_shuffled'][i]:
            raise ValueError('original SMC13C conditioned/null correspondence changed')
        if not item['hashes']['matched_to_original_smc13c_receipt']:
            raise ValueError('source assignment provenance claim changed')
    new=frozen['refit_roster']
    if len(new)!=4 or len(frozen['pair_calibration'])!=4:
        raise ValueError('refit models/calibration record missing')
    found=set()
    original=set(frozen['source_old_SAE_fingerprints'].values())
    for row in new:
        pair=(row['teacher_seed'],row['sae_seed'])
        if pair not in ((501,131),(501,167),(701,131),(701,167)) or pair in found:
            raise ValueError('unexpected independent SAE refit seed/teacher')
        found.add(pair)
        if row['optimizer_steps']!=300 or not 1<=row['best_validation_step']<=300:
            raise ValueError('SAE refit optimizer/selection length changed')
        if row['sae_fp32_fingerprint'] in original:
            raise ValueError('new SAE shares old snapshot fingerprint')
    if len({r['sae_fp32_fingerprint'] for r in new})!=4:
        raise ValueError('independent SAE refit fingerprint duplicates')
    return len(found)


def audit_frozen_files(frozen,root):
    for row in frozen['refit_roster']:
        label=f"saes/teacher{row['teacher_seed']}_sae{row['sae_seed']}.pt"
        if row['snapshot_relative_path']!=label:
            raise ValueError('refit checkpoint path changed')
        path=root/label
        if not path.is_file() or digest(path.read_bytes())!=row['snapshot_file_sha256']:
            raise ValueError('SAE refit archived file-byte SHA mismatch')


def audit_signs(frozen):
    for index,panel in enumerate(frozen['pair_calibration']):
        if (panel['source'],panel['target'])!=PAIRS[index]:
            raise ValueError('registered reference/refit model ordering changed')
        maps=panel['mapping']
        if set(maps)!=set(('matched','clue_shuffled','random')):
            raise ValueError('frozen calibration controls missing')
        for arm,perm in maps.items():
            if sorted(perm)!=list(range(96)) or digest(json.dumps(perm).encode())!=panel['mapping_hashes'][arm+'_sha256']:
                raise ValueError('new SAE Hungarian assignment SHA changed')
        expected=list(range(96))
        random.Random(202619810+index).shuffle(expected)
        if maps['random']!=expected:
            raise ValueError('registered random assignment seed changed')
        cal=panel['calibration']
        if cal['puzzles']!=64 or len(cal['raw'])!=64 or cal['cohort']!='feature_calibration':
            raise ValueError('original SMC13 calibration size changed')
        for k,row in enumerate(cal['raw']):
            if row['puzzle_index']!=k or row['seed']!=202613020000+k:
                raise ValueError('source calibration puzzle ordering changed')
            for src,target in zip(row['source_selected_ids'],row['target_matched_ids']):
                if maps['matched'][src]!=target:
                    raise ValueError('calibration frozen assignment changed per goal')
        replay=orient_replay(cal['raw'])
        for arm in ('matched','other'):
            for key,value in replay[arm].items():
                if key=='calibration_products_sum':
                    if not close(value,cal['orientation'][arm][key],1e-7):
                        raise ValueError('calibration-only sign covariance changed')
                elif value!=cal['orientation'][arm][key]:
                    raise ValueError('global causal sign selected outside calibration')


def audit_rosters(receipt):
    count=receipt['puzzles_per_family']
    if count not in (4,64):
        raise ValueError('SMC19 disallowed new test size')
    seen_x=set();seen_y=set()
    for family,(base,holes) in FAMILIES.items():
        rows=receipt['fresh_rosters'][family]
        if len(rows)!=count or receipt['test_seed_bases'][family]!=base:
            raise ValueError('SMC19 registered new puzzle cohort differs')
        if fingerprint(rows)!=receipt['fresh_manifest_sha256'][family]:
            raise ValueError('new Sudoku roster SHA-256 failed')
        for i,row in enumerate(rows):
            x,y=row['x'],row['target']
            if (row['seed']!=base+i or row['holes']!=holes or len(x)!=81 or
                x.count(0)!=holes or not valid_solution(y) or
                rectangle_count(y)!=row['rectangles'] or row['rectangles']<=0):
                raise ValueError('invalid or canonical new Sudoku source roster')
            if any(d!=0 and d!=y[j] for j,d in enumerate(x)) or count_solutions(x,2)!=1:
                raise ValueError('fresh exact-unique Sudoku truth differs')
            xx,yy=tuple(x),tuple(y)
            if xx in seen_x or yy in seen_y:
                raise ValueError('new SMC19 cohorts overlap each other')
            seen_x.add(xx);seen_y.add(yy)


def audit_gauge(g):
    if (g['rotation_seed']!=20261919 or
        g['tested_puzzles']!=4 or g['tested_goal_tokens']!=32 or
        'orthogonal' not in g['gauge_type']):
        raise ValueError('orthogonal transport engineering panel altered')
    for key in ('orthogonality_max_error','feature_code_max_abs_error',
                'clean_logits_max_abs_error','native_erase_logits_max_abs_error'):
        if g[key]<0 or not math.isfinite(g[key]) or g[key]>=2e-9:
            raise ValueError('exact jointly transported gauge failed tolerance')


def audit(receipt,frozen_dir):
    frozen_dir=Path(frozen_dir)
    bytes_frozen=(frozen_dir/'frozen.json').read_bytes()
    frozen=json.loads(bytes_frozen)
    if receipt['schema']!='SMC-19/orthogonal-gauge-and-independent-SAE-refit-causal-v1':
        raise ValueError('unregistered SMC19 source receipt schema')
    if (receipt['pre_registration_commit']!=PRECOMMIT or
        receipt['frozen_refit_json_sha256']!=digest(bytes_frozen) or
        receipt['frozen_calibration_source']!=frozen):
        raise ValueError('refit/orientation source lock mismatch')
    if receipt['source_original_receipts']!=frozen['source_original_receipts']:
        raise ValueError('original checkpoint SHA differs between phases')
    if not receipt['source_original_seven_assignments_hash_verified'] or not receipt['calibration_frozen_before_new_test']:
        raise ValueError('source assignment or calibration not frozen before heldout')
    if receipt['source_original_seven_assignments_hash_verified'] is not True or receipt['trained_models_and_SAEs_unchanged'] is not True:
        raise ValueError('source run provenance flag differs')
    audit_original(frozen)
    audit_frozen_files(frozen,frozen_dir)
    audit_signs(frozen)
    audit_rosters(receipt)
    audit_gauge(receipt['orthogonal_gauge_transport_control'])
    full=receipt['puzzles_per_family']==64 and len(receipt['pairs'])==4
    if receipt['full_protocol']!=full:
        raise ValueError('SMC19 pilot/full protocol label changed')
    cases=0;outcomes=[]
    for index,pair in enumerate(receipt['pairs']):
        if index>=4 or (pair['source'],pair['target'])!=PAIRS[index]:
            raise ValueError('unregistered receiving SAE fit pair')
        cal=frozen['pair_calibration'][index]
        if (pair['frozen_mapping_hashes']!=cal['mapping_hashes'] or
            pair['calibration_orientation']!=cal['calibration']['orientation']):
            raise ValueError('original mapping/sign differs from locked source')
        mapping=cal['mapping']['matched']
        for family in FAMILIES:
            group=pair['heldout'][family]
            raw=group['per_puzzle']
            cases+=len(raw)
            if len(raw)!=receipt['puzzles_per_family']:
                raise ValueError('incomplete fresh puzzle panel')
            validate_raw(raw,receipt['fresh_rosters'][family],True)
            for row in raw:
                for src,target in zip(row['source_selected_ids'],row['target_matched_ids']):
                    if mapping[src]!=target:
                        raise ValueError('heldout selected refit feature not frozen match')
            rebuilt=score_replay(raw,cal['calibration']['orientation'],
                                 202619130+index*100+len(family))
            compare_dict(rebuilt,group['statistics'],family+'/causal')
            if group['coverage']['eligible_goals']!=rebuilt['eligible_goals'] or group['coverage']['selected_goals']!=8*len(raw):
                raise ValueError('eligible causal subset count differs')
            for arm in ('matched','clue_shuffled','random'):
                overlaps=group['activation_overlap'][arm]
                if len(overlaps['per_puzzle'])!=len(raw) or not close(
                    overlaps['mean'],sum(overlaps['per_puzzle'])/len(raw)):
                    raise ValueError('new refit heldout source activity overlap mean changed')
                if any(v<0 or v>1 for v in overlaps['per_puzzle']):
                    raise ValueError('heldout coactivity fraction outside [0,1]')
            outcomes.append({'source':pair['source'],'target':pair['target'],
                             'family':family,'eligible_goals':rebuilt['eligible_goals'],
                             'paired_signed_concordance_advantage':rebuilt.get('matched_minus_other_active'),
                             'registered_family_gate':rebuilt['registered_success']})
        both=bool(full and all(pair['heldout'][fam]['statistics']['registered_success']
                               for fam in FAMILIES))
        if both!=pair['registered_success_both_families']:
            raise ValueError('pair-level both-family registered gate differs')
    return {
       'schema':'SMC-19/independent-stdlib-replay-v1',
       'n_new_SAE_snapshots':len(frozen['refit_roster']),
       'n_pair_family_puzzle_cases':cases,
       'n_test_goal_records':cases*8,
       'new_checkpoint_file_SHA256_verified':True,
       'original_seven_SMC13C_mapping_hashes_verified':True,
       'calibration_only_orientations_recomputed':True,
       'fresh_Sudoku_rosters_checked':True,
       'paired_native_erasure_and_puzzle_bootstrap_gate_recomputed':True,
       'gauge_transport_numerical_tolerances_checked':True,
       'full_protocol':full,
       'outcomes':outcomes,
       'scope':'Stdlib replay checks saved numerical data and checkpoint file hashes; PyTorch original neural inference/SAE retraining is the source-locked experimental trust boundary.'
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--frozen-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/smc19/audit.json'))
    a=p.parse_args()
    raw=a.receipt.read_bytes()
    report=audit(json.loads(raw),a.frozen_dir)
    report['original_full_receipt_sha256']=digest(raw)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'out':str(a.out),'full_protocol':report['full_protocol'],
         'n_puzzle_cases':report['n_pair_family_puzzle_cases'],
         'pair_outcomes':report['outcomes']},indent=2))


if __name__=='__main__':
    main()
