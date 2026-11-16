"""SMC-18 independent standard-library replay of task-gradient receipts.

Audits original seven SMC-13C mappings, fresh exact-unique Sudoku rosters,
SMC-17 pre-outcome tangent sham geometry, gradient inner products, the entire
inactive-row dual, finite directional curvature algebra, all puzzle-cluster
bootstraps and every registered decision. Does not import Torch or reexecute
the frozen recurrent teachers.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from audit_smc14 import ORIGINAL_CALIB_SHA
from audit_smc17 import verify_candidate
from smc13_family import valid_solution, rectangle_count, count_solutions

FAMILIES={'smc18_novel_30':(202618030000,30),
          'smc18_novel_34':(202618040000,34)}
PAIRS=(('pretrainedA_71','newB_501_SAE_71'),
       ('pretrainedA_71','newB_501_SAE_89'),
       ('pretrainedA_71','newB_701_SAE_71'),
       ('pretrainedA_71','newB_701_SAE_89'))
SOURCE_SHA='7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087'
TRAIN_SHA='1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3'
CALIB_SHA='3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07'
PRECOMMIT='70dc6a3e48097aeed02ccf1a10623a0cbbd624e5'


def close(a,b,tol=2e-7):
    if a is None or b is None:return a is None and b is None
    return (math.isfinite(a) and math.isfinite(b) and
            abs(a-b)<=tol*max(1,abs(a),abs(b)))


def dot(a,b):
    if len(a)!=len(b):raise ValueError('independent vector mismatch')
    return sum(x*y for x,y in zip(a,b))


def norm(x):
    return math.sqrt(dot(x,x))


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def audit_original_mappings(r):
    original=r['all_seven_original_assignment_receipts']
    if len(original)!=7 or r['original_conditioned_assignments_frozen_before_heldout'] is not True:
        raise ValueError('missing frozen original SMC13C mapping roster')
    for i,item in enumerate(original):
        mapping=item['matched']
        if sorted(mapping)!=list(range(96)):
            raise ValueError('original feature assignment is not a bijection')
        if hashlib.sha256(json.dumps(mapping).encode()).hexdigest()!=ORIGINAL_CALIB_SHA['matched'][i]:
            raise ValueError('original source matched assignment SHA differs')
        if item['hashes']['matched_assignment_sha256']!=ORIGINAL_CALIB_SHA['matched'][i]:
            raise ValueError('original matched source mapping hash changed')
        if item['hashes']['clue_shuffled_assignment_sha256']!=ORIGINAL_CALIB_SHA['clue_shuffled'][i]:
            raise ValueError('original conditioned shuffled mapping hash changed')
        if item['hashes']['matched_to_original_smc13c_receipt'] is not True:
            raise ValueError('unverified original SMC13 source alignment')


def audit_rosters(r):
    seen_x=set();seen_y=set()
    for name,(base,holes) in FAMILIES.items():
        rows=r['fresh_rosters'][name]
        if len(rows)!=r['puzzles_per_family'] or r['test_seed_bases'][name]!=base:
            raise ValueError('fresh SMC18 puzzle roster size or base differs')
        if fingerprint(rows)!=r['fresh_manifest_sha256'][name]:
            raise ValueError('new puzzle roster SHA-256 mismatch')
        for j,row in enumerate(rows):
            x,y=row['x'],row['target']
            if (row['seed']!=base+j or row['holes']!=holes or len(x)!=81 or
                x.count(0)!=holes or not valid_solution(y) or
                rectangle_count(y)!=row['rectangles'] or row['rectangles']<=0):
                raise ValueError('source Sudoku family changed')
            if any(v and v!=y[i] for i,v in enumerate(x)) or count_solutions(x,2)!=1:
                raise ValueError('Sudoku puzzle is invalid or not uniquely solved')
            tx,ty=tuple(x),tuple(y)
            if tx in seen_x or ty in seen_y:
                raise ValueError('duplicate new Sudoku puzzle/complete target')
            seen_x.add(tx);seen_y.add(ty)


def solve_cholesky(A,g):
    """Independent FP64 Cholesky for positive definite 48x48 Gram systems."""
    n=len(g)
    if len(A)!=n or any(len(row)!=n for row in A):
        raise ValueError('expected square full-rank Gram matrix')
    L=[[0.]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            val=A[i][j]-sum(L[i][q]*L[j][q] for q in range(j))
            if i==j:
                if val<=1e-16 or not math.isfinite(val):
                    raise ValueError('inactive physical Gram not full rank')
                L[i][j]=math.sqrt(val)
            else:L[i][j]=val/L[j][j]
    y=[]
    for i in range(n):
        y.append((g[i]-sum(L[i][j]*y[j] for j in range(i)))/L[i][i])
    x=[0.]*n
    for i in range(n-1,-1,-1):
        x[i]=(y[i]-sum(L[j][i]*x[j] for j in range(i+1,n)))/L[i][i]
    return x


def audit_inactive_dual(item,cov):
    gradient=item['gradient_48']
    kill=item['kill_delta_48'];sham=item['sham_delta_48']
    active=item['active_feature_ids']
    d=item['inactive_gradient_dual']
    if (len(gradient)!=48 or len(kill)!=48 or len(sham)!=48 or
        len(cov)!=96 or any(len(x)!=48 for x in cov) or
        len(active)!=4 or len(set(active))!=4 or
        any(not 0<=j<96 for j in active)):
        raise ValueError('invalid SAE gradient and encoder physical geometry')
    if d['full_inactive_rank']!=48 or d['minimum_inactive_singular_value']<=1e-10:
        raise ValueError('inactive full-rank guard lost')
    inactive=[j for j in range(96) if j not in active]
    B=[cov[j] for j in inactive]
    lam=d['lambda_minimum_norm']
    rec=d['gradient_reconstructed_from_inactive']
    if len(lam)!=92 or len(rec)!=48:
        raise ValueError('inactive minimum dual dimension differs')
    btlam=[sum(B[p][j]*lam[p] for p in range(92)) for j in range(48)]
    if any(not close(btlam[j],gradient[j],1e-7) or not close(rec[j],btlam[j],1e-7)
           for j in range(48)):
        raise ValueError('inactive dual does not reconstruct clean task gradient')
    # Independently check minimum-norm representation by solving B^T B w=g.
    columns=[[B[p][j] for p in range(92)] for j in range(48)]
    gram=[[dot(columns[i],columns[j]) for j in range(48)] for i in range(48)]
    w=solve_cholesky(gram,gradient)
    wanted=[dot(B[p],w) for p in range(92)]
    if any(not close(wanted[p],lam[p],1e-6) for p in range(92)):
        raise ValueError('inactive dual coefficients are not minimum-norm')
    u=[kill[j]-sham[j] for j in range(48)]
    shifts=[dot(row,u) for row in B]
    target=dot(gradient,u)
    reconstructed=dot(lam,shifts)
    ceiling=norm(lam)*norm(shifts)
    if (not close(target,d['task_gradient_dual_dot']) or
        not close(reconstructed,target,2e-7) or
        not close(ceiling,d['task_gradient_inactive_cs_ceiling']) or
        abs(target)>ceiling+1e-7*(1+abs(target)) or
        not close(norm(shifts),d['inactive_shift_l2']) or
        not close(norm([gradient[i]-btlam[i] for i in range(48)]),
                  d['reconstruction_l2_error'])):
        raise ValueError('inactive gradient task attribution or Cauchy ceiling changed')


def audit_metric(metric,diagnostic,cov):
    if metric['selected_feature']!=diagnostic['candidate_geometry']['matched_id']:
        raise ValueError('gradient target feature differs from prior selected atom')
    verify_candidate(diagnostic)
    if metric['active_feature_ids']!=diagnostic['candidate_geometry']['active_ids']:
        raise ValueError('gradient SAE clean active atom set differs')
    k=metric['kill_delta_48'];s=metric['sham_delta_48']
    if len(k)!=48 or len(s)!=48:
        raise ValueError('missing per-goal physical edits')
    d=norm(k)
    if not close(d,norm(s),1e-8):
        raise ValueError('kill and sham have differing physical doses')
    if abs(dot(k,s))>1e-7*(1+d*d):
        raise ValueError('original selective kill/tangent sham lost orthogonality')
    allkill=[dot(row,k) for row in cov]
    allsham=[dot(row,s) for row in cov]
    geo=diagnostic['candidate_geometry']
    if any(not close(x,y,2e-6) for x,y in zip(allkill,geo['all96_kill_preactivation_delta'])):
        raise ValueError('registered 96 encoder kill shift changed')
    if any(not close(x,y,2e-6) for x,y in zip(allsham,geo['all96_best_preactivation_delta'])):
        raise ValueError('registered 96 encoder sham shift changed')
    finite=metric['kill_margin_drop']-metric['sham_margin_drop']
    g=metric['gradient_48']
    linear_k=dot(g,k);linear_s=dot(g,s)
    linear=linear_k-linear_s
    qk=metric['directional_curvature_kill']
    qs=metric['directional_curvature_sham']
    second=linear-.5*(qk-qs)
    for field,value in (
        ('finite_kill_minus_sham',finite),
        ('clean_linear_kill',linear_k),
        ('clean_linear_sham',linear_s),
        ('clean_linear_prediction',linear),
        ('finite_minus_linear',finite-linear),
        ('relative_abs_linear_remainder',abs(finite-linear)/(abs(finite)+1e-8)),
        ('second_order_local_prediction',second),
        ('finite_minus_second_order',finite-second)
    ):
        if not close(value,metric[field],2e-7):
            raise ValueError('paired clean gradient / curvature scalar changed: '+field)
    if metric['kill_correct_to_wrong'] and not metric['clean_correct']:
        raise ValueError('kill falsely marks initially incorrect goal as flip')
    if metric['sham_correct_to_wrong'] and not metric['clean_correct']:
        raise ValueError('sham falsely marks initially incorrect goal as flip')
    audit_inactive_dual(metric,cov)


def audit_goals(comp,rosters,fam):
    rows=comp['heldout'][fam]['per_puzzle']
    roster=rosters[fam]
    if len(rows)!=len(roster):
        raise ValueError('heldout puzzle count differs')
    mapping=comp['frozen_matched_feature_mapping']
    cov=comp['heldout'][fam]['coverage']['target_physical_encoder_covectors_96x48']
    reasons={}
    eligible=0
    for index,(row,puzzle) in enumerate(zip(rows,roster)):
        if row['puzzle_index']!=index or row['seed']!=puzzle['seed']:
            raise ValueError('new hidden state wrong puzzle index')
        blanks=[i for i,v in enumerate(puzzle['x']) if v==0]
        positions=[blanks[k*len(blanks)//8] for k in range(8)]
        if (row['goal_positions']!=positions or
            row['goal_truth_digits']!=[puzzle['target'][i] for i in positions]):
            raise ValueError('goal masks/targets differ from frozen Sudoku roster')
        for key in ('source_top_feature_ids','target_matched_feature_ids','eligible_mask',
                    'edit_diagnostics','metrics'):
            if len(row[key])!=8:raise ValueError('per-goal evidence incomplete')
        for i in range(8):
            source=row['source_top_feature_ids'][i]
            target=row['target_matched_feature_ids'][i]
            if not 0<=source<96 or target!=mapping[source]:
                raise ValueError('SMC13C matched feature mapping drifted')
            info=row['edit_diagnostics'][i]
            reasons[info['status']]=reasons.get(info['status'],0)+1
            ok=info['status']=='eligible'
            if bool(row['eligible_mask'][i])!=ok or (row['metrics'][i] is None)==ok:
                raise ValueError('gradient data eligibility mask drifted')
            if ok:
                eligible+=1
                audit_metric(row['metrics'][i],info,cov)
    coverage=comp['heldout'][fam]['coverage']
    if (eligible!=coverage['eligible_goals'] or reasons!=coverage['ineligibility_counts'] or
        coverage['selected_goal_tokens']!=8*len(rows)):
        raise ValueError('reported gradient eligibility count differs')
    return eligible


def independent_score(rows,seed):
    source=[]
    for row in rows:
        n=0;sumD=0.;sumL=0.;ar=0.;ad=0.;c=0;agreements=0.
        for eligible,metric in zip(row['eligible_mask'],row['metrics']):
            if not eligible:continue
            n+=1
            d=metric['finite_kill_minus_sham'];l=metric['clean_linear_prediction']
            sumD+=d;sumL+=l;ar+=abs(d-l);ad+=abs(d)
            if abs(d)>1e-8 and abs(l)>1e-8:
                c+=1
                agreements+=1. if d*l>0 else 0.
        source.append((n,sumD,sumL,ar,ad,c,agreements))
    N=sum(x[0] for x in source)
    puzzles=sum(x[0]>0 for x in source)
    comparable=sum(x[5] for x in source)
    if N==0:
        return {'status':'no_eligible_goals','eligible_goals':0,
                'eligible_puzzles':0,'comparable_goals':0,'gate':False}
    r=random.Random(seed);bootD=[];bootS=[]
    for _ in range(2000):
        ids=[r.randrange(len(rows)) for i in rows]
        count=sum(source[i][0] for i in ids)
        numcomp=sum(source[i][5] for i in ids)
        if count:bootD.append(sum(source[i][1] for i in ids)/count)
        if numcomp:bootS.append(sum(source[i][6] for i in ids)/numcomp-.5)
    bootD.sort();bootS.sort()
    vd=len(bootD)==2000;vs=len(bootS)==2000
    ciD=[bootD[50],bootD[1949]] if vd else None
    ciS=[bootS[50],bootS[1949]] if vs else None
    absD=sum(x[4] for x in source)
    ratio=sum(x[3] for x in source)/absD if absD>1e-8 else None
    first=N>=24 and puzzles>=20 and vd and ciD[0]>0
    second=comparable>=24 and vs and ciS[0]>0
    third=ratio is not None and ratio<=.5
    return {
      'status':'measured' if vd and vs else 'bootstrap_abstention',
      'eligible_goals':N,'eligible_puzzles':puzzles,'comparable_goals':comparable,
      'finite_effect_mean':sum(x[1] for x in source)/N,
      'linear_prediction_mean':sum(x[2] for x in source)/N,
      'finite_minus_linear_mean':sum(x[1]-x[2] for x in source)/N,
      'absolute_remainder_mean':sum(x[3] for x in source)/N,
      'absolute_finite_effect_mean':absD/N,
      'relative_absolute_remainder':ratio,
      'sign_agreement_rate':sum(x[6] for x in source)/comparable if comparable else None,
      'unweighted_mean_puzzle_finite':sum(x[1]/x[0] for x in source if x[0])/puzzles,
      'puzzle_bootstrap95_finite':ciD,
      'puzzle_bootstrap95_sign_agreement_minus_half':ciS,
      'valid_finite_draws':len(bootD),'valid_sign_draws':len(bootS),
      'requested_bootstrap_draws':2000,
      'registered_coverage_and_finite_gate':first,
      'registered_gradient_sign_gate':second,
      'registered_linear_remainder_gate':third,
      'gate':bool(first and second and third)
    }


def audit(receipt):
    if receipt['schema']!='SMC-18/task-gradient-inactive-dual-v1':
        raise ValueError('unexpected SMC18 evidence format')
    if receipt['pre_registration_commit']!=PRECOMMIT:
        raise ValueError('original preregistration SHA changed')
    for key,value in (
      ('source_receipt_sha256',SOURCE_SHA),
      ('independent_training_receipt_sha256',TRAIN_SHA),
      ('conditioned_mapping_receipt_sha256',CALIB_SHA)
    ):
        if receipt[key]!=value:
            raise ValueError('source-frozen model/SAE/calibration SHA changed')
    settings=receipt['fixed_parameters']
    if settings!={'curvature_symmetric_fraction':.05,'bootstrap_draws':2000,
                  'sign_and_effect_cutoff':1e-8,
                  'inactive_minimum_singular_value':1e-10,
                  'linear_remainder_ratio_max':.5,
                  'sign_agreement_vs_half_ci_lower_strict':0}:
        raise ValueError('relabelled SMC18 fixed primary protocol')
    audit_original_mappings(receipt)
    full=(receipt['puzzles_per_family']==64 and len(receipt['comparison_roster'])==4)
    if receipt['full_protocol']!=full or receipt['puzzles_per_family'] not in (4,64):
        raise ValueError('pilot/full experimental label differs')
    audit_rosters(receipt)
    outcomes=[]
    for i,comp in enumerate(receipt['comparison_roster']):
        if i>=4 or (comp['source'],comp['target'])!=PAIRS[i]:
            raise ValueError('registered target model/SAE roster changed')
        mapping=comp['frozen_matched_feature_mapping']
        if (mapping!=receipt['all_seven_original_assignment_receipts'][i]['matched']
            or comp['original_calibration_assignment_sha256']!=receipt['all_seven_original_assignment_receipts'][i]['hashes']):
            raise ValueError('source-trained feature mapping changed')
        for fam in FAMILIES:
            coverage=audit_goals(comp,receipt['fresh_rosters'],fam)
            rows=comp['heldout'][fam]['per_puzzle']
            recalculated=independent_score(rows,202618130+100*i+len(fam))
            reported=comp['heldout'][fam]['statistics']
            if set(recalculated)!=set(reported):
                raise ValueError('gradient scorer schema changed')
            for key,value in recalculated.items():
                z=reported[key]
                if isinstance(value,list):
                    if not isinstance(z,list) or len(z)!=len(value) or any(not close(a,b) for a,b in zip(value,z)):
                        raise ValueError('gradient puzzle cluster interval differs: '+key)
                elif isinstance(value,float) or value is None:
                    if not close(value,z):
                        raise ValueError('gradient mean/difference differs: '+key)
                elif value!=z:
                    raise ValueError('registered gradient gate/count changed: '+key)
            if recalculated['eligible_goals']!=coverage:
                raise ValueError('SMC18 eligible denominator changed')
            outcomes.append({'target':comp['target'],'family':fam,
                             'eligible':coverage,
                             'finite_effect':recalculated.get('finite_effect_mean'),
                             'clean_gradient_prediction':recalculated.get('linear_prediction_mean'),
                             'relative_abs_remainder':recalculated.get('relative_absolute_remainder'),
                             'gate':recalculated['gate']})
        success=bool(full and all(comp['heldout'][fam]['statistics']['gate'] for fam in FAMILIES))
        if success!=comp['registered_gradient_support_both_families']:
            raise ValueError('SMC18 two-family success misreported')
    return {
        'schema':'SMC-18/independent-stdlib-gradient-inactive-dual-audit-v1',
        'n_pair_family_puzzle_cases':sum(len(c['heldout'][f]['per_puzzle']) for c in receipt['comparison_roster'] for f in FAMILIES),
        'original_seven_calibration_assignment_sha256_verified':True,
        'fresh_unique_noncanonical_sudoku_rosters_verified':True,
        'smc17_encoder_geometry_and_131_candidate_controls_replayed':True,
        'physical_task_gradient_and_minimum_norm_inactive_duals_replayed':True,
        'paired_finite_curvature_and_puzzle_bootstrap_gates_replayed':True,
        'full_protocol':full,'outcomes':outcomes,
        'scope':'Recomputes saved numerical gradients/duals and task response receipts; neural FP64 autograd and recurrent forward passes are trusted as source experimental observations.'
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('runs/smc18/audit.json'))
    args=parser.parse_args()
    raw=args.receipt.read_bytes()
    result=audit(json.loads(raw))
    result['source_full_json_sha256']=hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'audit':str(args.out),'full':result['full_protocol'],
                      'cases':result['n_pair_family_puzzle_cases'],
                      'pair_families':result['outcomes']},indent=2))


if __name__=='__main__':
    main()
