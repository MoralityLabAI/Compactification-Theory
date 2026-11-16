"""SMC-15: fresh-cohort native SAE erasure for active cross-model features.

All training, model selection, SMC-13C feature assignment and sign calibration
come from source-pinned earlier cohorts. The fresh test families are constructed
only after the calibration mappings and erasure orientations have been frozen.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

FAMILIES = {"smc15_novel_30": (202615030000, 30),
            "smc15_novel_34": (202615040000, 34)}
PAIR_COUNT = 4
PUZZLES = 64
GOALS = 8
EPS = 1e-8
BOOTSTRAPS = 2000


def fresh_rosters(old, limit=PUZZLES):
    """Generate exact-unique new families; refuse any old/new solution collision."""
    if not isinstance(limit, int) or limit not in (4, 64):
        raise ValueError('only preregistered 4-puzzle pilot or 64-puzzle full run')
    from smc13_family import (new_full_solution, dig_unique, valid_solution,
                              count_solutions, rectangle_count, fingerprint)
    seen_solutions = set()
    seen_inputs = set()
    for rows in old.values():
        for row in rows:
            target = tuple(row['target'])
            puzzle = tuple(row['x'])
            if target in seen_solutions or puzzle in seen_inputs:
                raise ValueError('source SMC13 cohorts overlap')
            seen_solutions.add(target)
            seen_inputs.add(puzzle)
    out = {}
    for name, (base, holes) in FAMILIES.items():
        rows = []
        for i in range(limit):
            seed = base + i
            solution = new_full_solution(seed)
            if not valid_solution(solution) or rectangle_count(solution) <= 0:
                raise ValueError('new target invalid or not out of canonical orbit')
            puzzle = dig_unique(solution, seed+1000000, holes)
            if puzzle.count(0) != holes or count_solutions(puzzle, 2) != 1:
                raise ValueError('new puzzle not exactly uniquely solvable')
            target = tuple(solution)
            inp = tuple(puzzle)
            if target in seen_solutions or inp in seen_inputs:
                raise ValueError('SMC15 overlaps frozen source or another fresh cohort')
            seen_solutions.add(target)
            seen_inputs.add(inp)
            rows.append({'seed':seed, 'x':puzzle, 'target':solution,
                         'holes':holes, 'rectangles':rectangle_count(solution)})
        out[name] = rows
    hashes = {name:fingerprint(rows) for name,rows in out.items()}
    return out, hashes


def encoded_goals(stage, pos, sae, mean, scale):
    """Freeze full original FP32 code before changing any hidden state."""
    if stage.ndim != 3 or stage.shape[1:] != (81,48) or pos.shape[1] != GOALS:
        raise ValueError('expected stage1 Nx81x48 and eight goal positions')
    n = len(stage)
    with torch.no_grad():
        h = stage[torch.arange(n)[:,None], pos.long()]
        z = sae.encode((h-mean)/scale).detach()
        if z.shape != (n,GOALS,96):
            raise ValueError('expected source top-four 96-way SAE code')
        top = z.argmax(-1)
        values = z.topk(2,dim=-1).values
        strict = (values[...,0]>0)&(values[...,0]>values[...,1])
    return z, top, strict


def physical_columns(sae, scale):
    with torch.no_grad():
        d=(sae.decoder.weight.detach()*scale[:,None]).T.double()
        if d.shape != (96,48) or not bool((d.norm(dim=-1)>1e-12).all()):
            raise ValueError('invalid physical SAE columns')
        return d


def feature_deltas(code, ids, columns):
    if code.ndim != 3 or code.shape[-1] != 96 or ids.shape != code.shape[:2]:
        raise ValueError('feature coefficient shape mismatch')
    if columns.shape != (96,48):
        raise ValueError('expected decoder columns 96x48')
    coeff=code.gather(-1,ids[:,:,None]).squeeze(-1).double()
    return coeff[...,None]*columns[ids],coeff


def other_active_ids(code, matched, pair_index, split, seeds):
    """Choose only a distinct positive source-blind target atom per goal."""
    n,k,w=code.shape
    if k!=GOALS or w!=96 or matched.shape!=(n,k) or len(seeds)!=n:
        raise ValueError('source-independent active-control selection shape')
    out=torch.zeros((n,k),dtype=torch.long)
    eligible=torch.zeros((n,k),dtype=torch.bool)
    for p in range(n):
        for j in range(k):
            selected=int(matched[p,j]); active=(code[p,j]>0).nonzero().flatten().tolist()
            choices=[int(c) for c in active if c!=selected]
            if float(code[p,j,selected])>0 and choices:
                token=(f'SMC15|{pair_index}|{split}|{seeds[p]}|{j}|{selected}')
                pick=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')%len(choices)
                out[p,j]=choices[pick]
                eligible[p,j]=True
    return out,eligible


def dose_matched_other(matched_delta, other_ids, columns):
    """A different *active* direction gets the identical original physical dose."""
    d=columns[other_ids]
    norms=d.norm(dim=-1,keepdim=True)
    if not bool((norms>1e-12).all()):
        raise ValueError('zero alternative SAE decoder')
    return (d/norms)*matched_delta.norm(dim=-1,keepdim=True)


def ablate_margin(model, stage, clean, truth, pos, delta):
    """FP64 continuation after one native stage1 goal-token subtraction."""
    if stage.ndim!=3 or stage.shape[1:]!=(81,48):
        raise ValueError('expected frozen stage1')
    n=len(stage)
    if clean.shape!=(n,81,10) or truth.shape!=(n,GOALS) or pos.shape!=(n,GOALS) or delta.shape!=(n,GOALS,48):
        raise ValueError('incorrect ablation input shape')
    if delta.dtype!=torch.float64:
        raise ValueError('physical erasure must use FP64 continuation')
    jj=torch.arange(GOALS)
    result=[]; correct=[]; flips=[]
    with torch.no_grad():
        for p in range(n):
            if not bool(((pos[p]>=0)&(pos[p]<81)).all()):
                raise ValueError('goal positions outside grid')
            h=stage[p:p+1].double().repeat(GOALS,1,1)
            h[jj,pos[p]]-=delta[p]
            for _ in range(3):
                h=model.block(h)
            out=model.head(h)[jj,pos[p]]
            base=clean[p,pos[p]]
            y=truth[p].long()
            rival=base.clone()
            rival[jj,y]=-float('inf')
            c=rival.argmax(-1)
            initial=base[jj,y]-base[jj,c]
            after=out[jj,y]-out[jj,c]
            loss=initial-after
            if not bool(torch.isfinite(loss).all()):
                raise ArithmeticError('nonfinite native erasure margin loss')
            source_correct=base.argmax(-1)==y
            newly_wrong=source_correct&(out.argmax(-1)!=y)
            result.append(loss.tolist())
            correct.append(source_correct.tolist())
            flips.append(newly_wrong.tolist())
    return torch.tensor(result,dtype=torch.float64),torch.tensor(correct,dtype=torch.bool),torch.tensor(flips,dtype=torch.bool)


def reencoded_matched(stage,pos,sae,mean,scale,deltas,matched):
    """Post-erasure feature is a diagnostic, not a selection filter."""
    n=len(stage)
    with torch.no_grad():
        base=stage[torch.arange(n)[:,None],pos]
        erased=(base.double()-deltas).float()
        code=sae.encode((erased-mean)/scale)
        return code.gather(-1,matched[:,:,None]).squeeze(-1)


def orient(source,matched,other,eligible):
    if not (source.shape==matched.shape==other.shape==eligible.shape):
        raise ValueError('misaligned causal calibration effects')
    count=int(eligible.sum())
    out={}
    for key,t in [('matched',matched),('other',other)]:
        product=float((source[eligible]*t[eligible]).sum()) if count else 0.
        out[key]={'sign':(1 if product>=0 else -1),
                  'identified':count>0 and abs(product)>1e-14,
                  'calibration_products_sum':product,
                  'eligible_calibration_goals':count}
    return out


def sign_eps(v,eps=EPS):
    if abs(v)<=eps:return 0
    return 1 if v>0 else -1


def score_rows(rows,orientation,seed,draws=BOOTSTRAPS):
    if not rows:return {'status':'empty'}
    signs={k:orientation[k]['sign'] for k in ('matched','other')}
    sums=[];n_tok=[];m_sums=[];o_sums=[]
    n_usable=0
    for row in rows:
        eligible=row['eligible_mask']
        if not all(len(row[k])==GOALS for k in ('eligible_mask','source_drop','matched_drop','other_active_drop')):
            raise ValueError('wrong puzzle raw response count')
        vals_m=[];vals_o=[]
        for i,active in enumerate(eligible):
            if not active:continue
            a=sign_eps(row['source_drop'][i])
            vals_m.append(a*sign_eps(row['matched_drop'][i])*signs['matched'])
            vals_o.append(a*sign_eps(row['other_active_drop'][i])*signs['other'])
        if vals_m:n_usable+=1
        n_tok.append(len(vals_m))
        m_sums.append(sum(vals_m));o_sums.append(sum(vals_o))
        sums.append(sum(vals_m)-sum(vals_o))
    total=sum(n_tok)
    if total==0 or not all(orientation[k]['identified'] for k in signs):
        return {'status':'insufficient_coverage_or_unidentified_calibration',
                'eligible_goals':total,'nonempty_puzzle_clusters':n_usable,
                'registered_success':False}
    rate=sum(sums)/total
    rng=random.Random(seed)
    resampled=[]
    for _ in range(draws):
        draw=[rng.randrange(len(rows)) for _ in rows]
        denom=sum(n_tok[i] for i in draw)
        if denom:
            resampled.append(sum(sums[i] for i in draw)/denom)
    resampled.sort()
    all_valid=len(resampled)==draws
    interval=([resampled[int(.025*draws)],resampled[int(.975*draws)-1]]
              if all_valid else None)
    good=(total>=24 and n_usable>=20 and all_valid and interval[0]>0)
    unweighted=(sum(sums[i]/n_tok[i] for i in range(len(rows)) if n_tok[i]) / n_usable)
    return {
      'status':'measured' if all_valid else 'bootstrap_abstained',
      'eligible_goals':total,'nonempty_puzzle_clusters':n_usable,
      'mean_matched_signed_concordance':sum(m_sums)/total,
      'mean_other_active_signed_concordance':sum(o_sums)/total,
      'matched_minus_other_active':rate,
      'unweighted_mean_puzzle_difference':unweighted,
      'puzzle_cluster_bootstrap95':interval,
      'bootstrap_valid_resamples':len(resampled),
      'bootstrap_requested_resamples':draws,
      'registered_success':good}


def evaluate_pair_split(ctx,components,a,b,assign,split,rows,pair_index):
    tc_a=ctx[components[a][0]]['cohorts'][split]
    tc_b=ctx[components[b][0]]['cohorts'][split]
    x=tc_a['inputs'];pos=tc_a['positions'];truth=tc_a['truth_at_goals']
    if not bool((x==tc_b['inputs']).all()):
        raise ValueError('input roster not equal across teachers')
    za,top,strict=encoded_goals(tc_a['stage1'],pos,*components[a][1:])
    zb,_,_=encoded_goals(tc_b['stage1'],pos,*components[b][1:])
    mapping=torch.tensor(assign,dtype=torch.long)
    matched=mapping[top]
    alternative,available=other_active_ids(zb,matched,pair_index,split,[r['seed'] for r in rows])
    eligible=(strict&available)
    da=physical_columns(components[a][1],components[a][3])
    db=physical_columns(components[b][1],components[b][3])
    src_delta,_=feature_deltas(za,top,da)
    matched_delta,matched_z=feature_deltas(zb,matched,db)
    ctrl_delta=dose_matched_other(matched_delta,alternative,db)
    if not bool(torch.allclose(matched_delta.norm(dim=-1),ctrl_delta.norm(dim=-1),atol=1e-12,rtol=1e-12)):
        raise AssertionError('active control physical dose differs')
    result_a,correct_a,flip_a=ablate_margin(ctx[components[a][0]]['model64'],tc_a['stage1'],tc_a['clean_logits'],truth,pos,src_delta)
    result_b,correct_b,flip_b=ablate_margin(ctx[components[b][0]]['model64'],tc_b['stage1'],tc_b['clean_logits'],truth,pos,matched_delta)
    result_c,_,flip_c=ablate_margin(ctx[components[b][0]]['model64'],tc_b['stage1'],tc_b['clean_logits'],truth,pos,ctrl_delta)
    after=reencoded_matched(tc_b['stage1'],pos,components[b][1],components[b][2],components[b][3],matched_delta,matched)
    raw=[]
    for p in range(len(rows)):
        raw.append({
          'puzzle_index':p,'seed':rows[p]['seed'],
          'goal_positions':pos[p].tolist(),
          'eligible_mask':eligible[p].tolist(),
          'source_selected_ids':top[p].tolist(),
          'target_matched_ids':matched[p].tolist(),
          'target_other_active_ids':alternative[p].tolist(),
          'target_other_active_coefficients':zb[p].gather(-1,alternative[p,:,None]).squeeze(-1).tolist(),
          'target_match_clean_coefficients':matched_z[p].tolist(),
          'target_match_after_erase_coefficients':after[p].tolist(),
          'matched_physical_dose':matched_delta[p].norm(dim=-1).tolist(),
          'other_physical_dose':ctrl_delta[p].norm(dim=-1).tolist(),
          'source_drop':result_a[p].tolist(),
          'matched_drop':result_b[p].tolist(),
          'other_active_drop':result_c[p].tolist(),
          'source_clean_correct':correct_a[p].tolist(),
          'target_clean_correct':correct_b[p].tolist(),
          'source_correct_to_wrong_erase':flip_a[p].tolist(),
          'target_matched_correct_to_wrong_erase':flip_b[p].tolist(),
          'target_other_correct_to_wrong_erase':flip_c[p].tolist(),
        })
    selected=int(eligible.sum())
    return raw, {'eligible_goals':selected, 'selected_goals':len(rows)*GOALS,
                 'target_match_active_rate':float((matched_z>0).float().mean()),
                 'target_clean_accuracy':float(correct_b.float().mean()),
                 'source_clean_accuracy':float(correct_a.float().mean()),
                 'eligible_target_clean_accuracy':float(correct_b[eligible].float().mean()) if selected else None,
                 'eligible_target_matched_clean_correct_to_wrong':float(flip_b[eligible].float().mean()) if selected else None,
                 'eligible_target_other_clean_correct_to_wrong':float(flip_c[eligible].float().mean()) if selected else None,
                 'eligible_post_erasure_reencoded_ratio':float((after[eligible].double()/matched_z[eligible]).mean()) if selected else None}


def run(original_artifact, independent_artifact, conditioned_receipt, allow_fetch=False,
        source_checkpoint=None, pilot=0, pair_limit=PAIR_COUNT):
    """Model and calibration locks precede any new heldout solution or state."""
    from smc14_causal_transfer import (load_frozen_sources,source_model_from_lock,
         load_independent_models,load_conditioned,source_group_context,
         exact_frozen_assignments,PAIRS,ORIGINAL_SOURCE_RECEIPT_SHA256,
         NEW_TEACHER_RECEIPT_SHA256,CONDITIONED_JSON_SHA256)
    torch.set_num_threads(2)
    root_orig=Path(original_artifact)
    root_ind=Path(independent_artifact)
    if hashlib.sha256((root_orig/'joint-full.json').read_bytes()).hexdigest()!=ORIGINAL_SOURCE_RECEIPT_SHA256:
        raise ValueError('original full SAE source receipt changed')
    if hashlib.sha256((root_ind/'full.json').read_bytes()).hexdigest()!=NEW_TEACHER_RECEIPT_SHA256:
        raise ValueError('new teacher training receipt changed')
    original,osaes,_=load_frozen_sources(extracted_dir=original_artifact)
    teacherA,_=source_model_from_lock(allow_fetch,source_checkpoint)
    independent,original_rosters,new=load_independent_models(independent_artifact)
    conditioned=load_conditioned(conditioned_receipt)
    teachers={'A':teacherA}
    components={'pretrainedA_71':('A',*osaes[71])}
    for seed,group in new.items():
        teacher_label=f'B{seed}'
        teachers[teacher_label]=group['teacher']
        for sae_seed,params in group['saes'].items():
            components[f'newB_{seed}_SAE_{sae_seed}']=(teacher_label,*params)
    compare=PAIRS[:pair_limit]
    if pair_limit<1 or pair_limit>PAIR_COUNT or not all(a=='pretrainedA_71' for a,b in compare):
        raise ValueError('exact four source-to-new primary comparisons required')
    cal_ctx,cal_support=source_group_context(teachers,components,original_rosters,('feature_calibration',))
    frozen={}
    # Audit all seven original assignment/checkpoint pairs even in pilot mode.
    for i,(a,b) in enumerate(PAIRS):
        receipt=conditioned['results'][i]
        if (receipt['source'],receipt['target'])!=(a,b):
            raise ValueError('SMC13 original comparison order changed')
        observed=cal_ctx[components[a][0]]['cohorts']['feature_calibration']['inputs']
        assignments,sha=exact_frozen_assignments(
            cal_support[a]['feature_calibration']['support'],
            cal_support[b]['feature_calibration']['support'],observed,receipt,i)
        frozen[i]=(assignments['matched'],sha)
    print('SMC15_FROZEN_ALL_SEVEN_SMC13C_ASSIGNMENTS',flush=True)
    calibration={}
    for i,(a,b) in enumerate(compare):
        rows,_=evaluate_pair_split(cal_ctx,components,a,b,frozen[i][0],
                                   'feature_calibration',original_rosters['feature_calibration'],i)
        mask=torch.tensor([r['eligible_mask'] for r in rows],dtype=torch.bool)
        aa=torch.tensor([r['source_drop'] for r in rows],dtype=torch.float64)
        bb=torch.tensor([r['matched_drop'] for r in rows],dtype=torch.float64)
        cc=torch.tensor([r['other_active_drop'] for r in rows],dtype=torch.float64)
        orientation=orient(aa,bb,cc,mask)
        calibration[i]={'orientation':orientation,'raw':rows,
                        'feature_assignment_sha256':frozen[i][1]}
        if not all(orientation[k]['identified'] for k in ('matched','other')):
            print('SMC15_CALIBRATION_ABSTENTION',i,orientation,flush=True)
    print('SMC15_CALIBRATION_ORIENTATIONS_FROZEN',
          {i:calibration[i]['orientation'] for i in calibration},flush=True)
    new_rosters,manifest=fresh_rosters(original_rosters,limit=pilot or PUZZLES)
    test_ctx,_=source_group_context(teachers,components,new_rosters,tuple(FAMILIES))
    results=[]
    for i,(a,b) in enumerate(compare):
        heldout={}
        for fam in FAMILIES:
            rows,coverage=evaluate_pair_split(test_ctx,components,a,b,frozen[i][0],fam,new_rosters[fam],i)
            stats=score_rows(rows,calibration[i]['orientation'],
                             seed=202615130+100*i+len(fam))
            heldout[fam]={'statistics':stats,'coverage':coverage,'per_puzzle':rows}
        results.append({'source':a,'target':b,'calibration':calibration[i],
                        'heldout':heldout,
                        'registered_success_on_both_fresh_families':
                        bool(not pilot and all(heldout[f]['statistics'].get('registered_success',False) for f in FAMILIES))})
        print('SMC15_NATIVE_ERASURE',a,b,
              {f:heldout[f]['statistics'].get('matched_minus_other_active') for f in FAMILIES},flush=True)
    return {
      'schema':'SMC-15/fresh-cohort-active-native-erasure-v1',
      'preregistered_test_bases':{k:v[0] for k,v in FAMILIES.items()},
      'source_receipt_sha256':ORIGINAL_SOURCE_RECEIPT_SHA256,
      'independent_training_receipt_sha256':NEW_TEACHER_RECEIPT_SHA256,
      'conditioned_mapping_receipt_sha256':CONDITIONED_JSON_SHA256,
      'fresh_cohort_manifest_sha256':manifest,'fresh_cohort_rosters':new_rosters,
      'fresh_puzzles_per_family':pilot or PUZZLES,
      'full_protocol':not bool(pilot or pair_limit<PAIR_COUNT),
      'calibration_complete_before_heldout':True,
      'compare':results,
      'limitations':'Operational SAE decoder-contribution erasure and numerical FP64 margin responses; not a uniquely identified mechanism, a Lean certificate or cross-teacher capability equivalence.'
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original-artifact',type=Path,required=True)
    p.add_argument('--independent-artifact',type=Path,required=True)
    p.add_argument('--conditioned-receipt',type=Path,required=True)
    p.add_argument('--fetch-source',action='store_true')
    p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--pilot-puzzles',type=int,choices=(4,))
    p.add_argument('--pair-limit',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--out',type=Path,default=Path('runs/smc15/full.json'))
    a=p.parse_args()
    result=run(a.original_artifact,a.independent_artifact,a.conditioned_receipt,
               a.fetch_source,a.source_checkpoint,a.pilot_puzzles or 0,a.pair_limit)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(a.out),'full_protocol':result['full_protocol'],
        'cohort_manifest':result['fresh_cohort_manifest_sha256'],
        'pairs':[{ 'pair':r['source']+' -> '+r['target'],
          'pass':r['registered_success_on_both_fresh_families']} for r in result['compare']]},indent=2))


if __name__=='__main__':
    main()
