"""SMC-18 source-locked task-gradient attribution under inactive SAE leakage.

Reuses SMC-16 source-matched active-feature selective kill and SMC-17
outcome-blind 131-candidate inactive-matched tangent sham, with new disjoint
Sudoku families. Computes FP64 input gradients and directional curvatures
after original stage-one FP32 anchors. Never changes pretrained weights.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from smc13_family import (
    new_full_solution, dig_unique, valid_solution, rectangle_count,
    count_solutions, fingerprint,
)
from smc15_active_necessity import encoded_goals, physical_columns, ablate_margin
from smc16_selective_kill import selective_vectors, physical_covectors
from smc17_inactive_control import (
    fresh_rosters as original_smc17_rosters,
    best_inactive_matched_sham,
)
from smc17_causal_inactive import source_models

FAMILIES = {'smc18_novel_30': (202618030000, 30),
            'smc18_novel_34': (202618040000, 34)}
PAIRS = (('pretrainedA_71', 'newB_501_SAE_71'),
         ('pretrainedA_71', 'newB_501_SAE_89'),
         ('pretrainedA_71', 'newB_701_SAE_71'),
         ('pretrainedA_71', 'newB_701_SAE_89'))
PRECOMMIT = '70dc6a3e48097aeed02ccf1a10623a0cbbd624e5'
PZ = 64
GOALS = 8
DIFF = .05
ZERO = 1e-8
BOOTS = 2000


def fresh_rosters(old, limit=PZ):
    """Fresh exact-unique cohorts, disjoint from every SMC13-17 puzzle."""
    if limit not in (4, 64):
        raise ValueError('only preregistered four-puzzle pilot or 64-puzzle full')
    prev17, _ = original_smc17_rosters(old, limit=64)
    ys = {tuple(r['target']) for rows in list(old.values())+list(prev17.values())
          for r in rows}
    xs = {tuple(r['x']) for rows in list(old.values())+list(prev17.values())
          for r in rows}
    out = {}
    for name,(base,holes) in FAMILIES.items():
        rows=[]
        for i in range(limit):
            seed=base+i
            y=new_full_solution(seed)
            if not valid_solution(y) or rectangle_count(y)<=0:
                raise ValueError('new Sudoku solution must be noncanonical')
            x=dig_unique(y,seed+1000000,holes)
            if x.count(0)!=holes or count_solutions(x,2)!=1:
                raise ValueError('SMC18 puzzle not exact uniquely solvable')
            if any(v and v!=y[j] for j,v in enumerate(x)):
                raise ValueError('SMC18 clues incompatible with target')
            if tuple(y) in ys or tuple(x) in xs:
                raise ValueError('fresh Sudoku overlaps prior SMC13-17 cohorts')
            ys.add(tuple(y));xs.add(tuple(x))
            rows.append({'seed':seed,'x':x,'target':y,'holes':holes,
                         'rectangles':rectangle_count(y)})
        out[name]=rows
    return out,{name:fingerprint(rows) for name,rows in out.items()}


def competitors(clean,truth):
    """Highest clean competing digit, frozen before all edits or gradients."""
    if clean.ndim!=2 or clean.shape[1]!=10 or truth.shape!=(len(clean),):
        raise ValueError('unexpected clean logits and truth shape')
    other=clean.clone()
    other[torch.arange(len(clean)),truth]=-float('inf')
    return other.argmax(-1)


def task_gradients(model,stage,clean,truth,pos,eligible):
    """Independent replicated clean anchors, per-goal diagonal Jacobian blocks.

    The gradients are with respect to physical stage-one hidden states, not
    model weights. Every replicated clean puzzle has its own input leaf.
    """
    if stage.shape!=(81,48) or clean.shape!=(81,10) or pos.shape!=(GOALS,):
        raise ValueError('single-puzzle stage/logit/goal shape changed')
    if truth.shape!=(GOALS,) or eligible.shape!=(GOALS,):
        raise ValueError('selected goal labels/masks must be aligned')
    indices=torch.nonzero(eligible).view(-1).long()
    out=torch.zeros((GOALS,48),dtype=torch.float64)
    if not len(indices):
        return out
    goals=pos[indices].long()
    y=truth[indices].long()
    rival=competitors(clean[goals],y)
    for param in model.parameters():
        param.requires_grad_(False)
    with torch.enable_grad():
        h=stage.double().unsqueeze(0).repeat(len(indices),1,1).detach().requires_grad_(True)
        z=h
        for _ in range(3):
            z=model.block(z)
        logits=model.head(z)[torch.arange(len(indices)),goals]
        margin=logits[torch.arange(len(indices)),y]-logits[torch.arange(len(indices)),rival]
        base=clean[goals]
        old=base[torch.arange(len(indices)),y]-base[torch.arange(len(indices)),rival]
        if not bool(torch.isfinite(margin).all()) or float((margin-old).abs().max())>2e-8:
            raise ArithmeticError('differentiable clean FP64 rollout disagrees with frozen clean logit')
        grad=torch.autograd.grad(margin.sum(),h,create_graph=False,retain_graph=False)[0]
    selected=grad[torch.arange(len(indices)),goals]
    if selected.shape!=(len(indices),48) or not bool(torch.isfinite(selected).all()):
        raise ArithmeticError('nonfinite true-task goal gradient')
    out[indices]=selected.detach()
    return out


def curvature_probe(model,stage,clean,truth,pos,eligible,kill,sham,eps=DIFF):
    """Four evaluation-only +/- .05 directional samples per eligible goal."""
    indices=torch.nonzero(eligible).view(-1).long()
    qk=torch.zeros(GOALS,dtype=torch.float64)
    qs=torch.zeros(GOALS,dtype=torch.float64)
    if not len(indices):
        return qk,qs
    if eps<=0 or stage.shape!=(81,48):
        raise ValueError('invalid directional curvature geometry')
    with torch.no_grad():
        goals=pos[indices].long()
        t=len(indices)
        jj=torch.arange(t)
        h=stage.double().unsqueeze(0).repeat(4*t,1,1)
        kd=kill[indices];sd=sham[indices]
        h[jj,goals]+=eps*kd
        h[t+jj,goals]-=eps*kd
        h[2*t+jj,goals]+=eps*sd
        h[3*t+jj,goals]-=eps*sd
        for _ in range(3):
            h=model.block(h)
        logits=model.head(h)
        yy=truth[indices].long()
        rival=competitors(clean[goals],yy)
        values=[]
        for offset in (0,t,2*t,3*t):
            oo=logits[offset+jj,goals]
            values.append(oo[jj,yy]-oo[jj,rival])
        base=clean[goals]
        fm=base[jj,yy]-base[jj,rival]
        a=(values[0]+values[1]-2*fm)/(eps**2)
        b=(values[2]+values[3]-2*fm)/(eps**2)
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ArithmeticError('nonfinite directional curvature')
        qk[indices]=a
        qs[indices]=b
    return qk,qs


def inactive_gradient_dual(cov,active,gradient,u):
    """Minimal L2 inactive encoder dual and conditional sharp CS ceiling."""
    if cov.shape!=(96,48) or gradient.shape!=(48,) or u.shape!=(48,):
        raise ValueError('physical encoder and gradient dimensions differ')
    inactive=[i for i in range(96) if i not in active]
    if len(active)!=4 or len(inactive)!=92:
        raise ValueError('exactly four active SAE atoms required')
    B=cov[inactive]
    sv=torch.linalg.svdvals(B)
    sigma_min=float(sv[-1])
    if sigma_min<=1e-10:
        raise ValueError('inactive encoder block lacks full column rank')
    bt=B.T
    lam=B@torch.linalg.solve(bt@B,gradient)
    reconstructed=bt@lam
    residual=float((reconstructed-gradient).norm())
    if residual>1e-7*(1+float(gradient.norm())):
        raise ArithmeticError('inactive gradient dual reconstruction exceeds tolerance')
    shift=B@u
    dot=float(gradient@u)
    dual=float(lam@shift)
    bound=float(lam.norm()*shift.norm())
    if abs(dot-dual)>1e-7*(1+abs(dot)) or abs(dot)>bound+1e-7*(1+abs(dot)):
        raise ArithmeticError('inactive dual / CS ceiling inconsistent')
    return {'minimum_inactive_singular_value':sigma_min,
            'full_inactive_rank':48,
            'lambda_minimum_norm':lam.tolist(),
            'gradient_reconstructed_from_inactive':reconstructed.tolist(),
            'reconstruction_l2_error':residual,
            'inactive_shift_l2':float(shift.norm()),
            'task_gradient_dual_dot':dual,
            'task_gradient_inactive_cs_ceiling':bound}


def score_family(rows,seed,draws=BOOTS):
    """Puzzle-level bootstrap, preserving all goal slots in each resample."""
    if not rows:return {'status':'empty','gate':False}
    clusters=[]
    for row in rows:
        if len(row['eligible_mask'])!=GOALS or len(row['metrics'])!=GOALS:
            raise ValueError('wrong number of goal slots')
        d=[];l=[];abse=[];absd=[];agreement=[]
        for active,item in zip(row['eligible_mask'],row['metrics']):
            if active!=(item is not None):
                raise ValueError('missing/forged gradient result under eligible mask')
            if not active:continue
            D=item['finite_kill_minus_sham']
            L=item['clean_linear_prediction']
            if not all(math.isfinite(x) for x in (D,L)):
                raise ValueError('nonfinite gradient response')
            d.append(D);l.append(L)
            abse.append(abs(D-L));absd.append(abs(D))
            if abs(D)>ZERO and abs(L)>ZERO:
                agreement.append(1. if D*L>0 else 0.)
        clusters.append((len(d),sum(d),sum(l),sum(abse),sum(absd),
                         len(agreement),sum(agreement)))
    eligible=sum(x[0] for x in clusters)
    ncluster=sum(x[0]>0 for x in clusters)
    comparable=sum(x[5] for x in clusters)
    absD=sum(x[4] for x in clusters)
    if not eligible:
        return {'status':'no_eligible_goals','eligible_goals':0,
                'eligible_puzzles':0,'comparable_goals':0,'gate':False}
    rng=random.Random(seed)
    db=[];sb=[]
    for _ in range(draws):
        ids=[rng.randrange(len(rows)) for _ in rows]
        nt=sum(clusters[j][0] for j in ids)
        nc=sum(clusters[j][5] for j in ids)
        if nt:db.append(sum(clusters[j][1] for j in ids)/nt)
        if nc:sb.append(sum(clusters[j][6] for j in ids)/nc-.5)
    db.sort();sb.sort()
    valid_d=len(db)==draws
    valid_s=len(sb)==draws
    ciD=[db[50],db[1949]] if valid_d else None
    ciS=[sb[50],sb[1949]] if valid_s else None
    ratio=sum(x[3] for x in clusters)/absD if absD>ZERO else None
    coverage=(eligible>=24 and ncluster>=20 and valid_d and ciD[0]>0)
    signs=(comparable>=24 and valid_s and ciS[0]>0)
    linear=(ratio is not None and ratio<=.50)
    return {
        'status':'measured' if valid_d and valid_s else 'bootstrap_abstention',
        'eligible_goals':eligible,'eligible_puzzles':ncluster,
        'comparable_goals':comparable,
        'finite_effect_mean':sum(x[1] for x in clusters)/eligible,
        'linear_prediction_mean':sum(x[2] for x in clusters)/eligible,
        'finite_minus_linear_mean':sum(x[1]-x[2] for x in clusters)/eligible,
        'absolute_remainder_mean':sum(x[3] for x in clusters)/eligible,
        'absolute_finite_effect_mean':absD/eligible,
        'relative_absolute_remainder':ratio,
        'sign_agreement_rate':sum(x[6] for x in clusters)/comparable if comparable else None,
        'unweighted_mean_puzzle_finite':sum(x[1]/x[0] for x in clusters if x[0])/ncluster,
        'puzzle_bootstrap95_finite':ciD,
        'puzzle_bootstrap95_sign_agreement_minus_half':ciS,
        'valid_finite_draws':len(db),'valid_sign_draws':len(sb),
        'requested_bootstrap_draws':draws,
        'registered_coverage_and_finite_gate':coverage,
        'registered_gradient_sign_gate':signs,
        'registered_linear_remainder_gate':linear,
        'gate':bool(coverage and signs and linear)
    }


def evaluate_family(ctx,comps,a,b,assignment,roster,family,pair_idx):
    source=ctx[comps[a][0]]['cohorts'][family]
    dest=ctx[comps[b][0]]['cohorts'][family]
    if not bool((source['inputs']==dest['inputs']).all()):
        raise ValueError('teacher input puzzle cohort mismatch')
    pos=dest['positions']
    _,top,strict=encoded_goals(source['stage1'],pos,*comps[a][1:])
    sae,mean,scale=comps[b][1:]
    code,_,_=encoded_goals(dest['stage1'],pos,sae,mean,scale)
    mapped=torch.tensor(assignment,dtype=torch.long)[top]
    cov=physical_covectors(sae,scale)
    dec=physical_columns(sae,scale)
    n=len(roster)
    k=torch.zeros(n,GOALS,48,dtype=torch.float64)
    s=torch.zeros_like(k)
    diagnostics=[]
    reasons={}
    for p,row in enumerate(roster):
        infos=[]
        for slot in range(GOALS):
            info={'status':'source_top_not_strict','candidate_geometry':None}
            if bool(strict[p,slot]):
                h=dest['stage1'][p,int(pos[p,slot])]
                matched=int(mapped[p,slot])
                dk,seeded,prior=selective_vectors(
                    h,code[p,slot],matched,sae,mean,scale,cov,dec,
                    row['seed'],pair_idx,family,slot)
                info=dict(prior)
                info['candidate_geometry']=None
                if prior['status']=='eligible':
                    best,geo=best_inactive_matched_sham(
                        h,code[p,slot],matched,sae,mean,scale,cov,
                        dk,seeded,pair_idx,family,row['seed'],slot)
                    if (geo['full_inactive_row_rank']!=48 or
                        geo['inactive_singular_value_min']<=1e-10):
                        info['status']='inactive_rank_obstruction'
                    else:
                        k[p,slot]=dk
                        s[p,slot]=best
                        info['candidate_geometry']=geo
            infos.append(info)
            reasons[info['status']]=reasons.get(info['status'],0)+1
        diagnostics.append(infos)
    eligible=torch.tensor([[row[j]['status']=='eligible' for j in range(GOALS)]
                           for row in diagnostics],dtype=torch.bool)
    print('SMC18_PREOUTCOME_ELIGIBILITY_LOCKED',family,pair_idx,int(eligible.sum()),flush=True)
    # Only now may the model read labels, final margin logits or task gradients.
    model=ctx[comps[b][0]]['model64']
    for param in model.parameters():param.requires_grad_(False)
    truth=dest['truth_at_goals']
    clean=dest['clean_logits']
    dk,correct,fk=ablate_margin(model,dest['stage1'],clean,truth,pos,k)
    ds,correct_s,fs=ablate_margin(model,dest['stage1'],clean,truth,pos,s)
    if not bool((correct==correct_s).all()):
        raise AssertionError('different clean target labels among paired arms')
    rows=[]
    for p,original in enumerate(roster):
        grad=task_gradients(model,dest['stage1'][p],clean[p],truth[p],pos[p],eligible[p])
        qk,qs=curvature_probe(model,dest['stage1'][p],clean[p],truth[p],pos[p],
                               eligible[p],k[p],s[p])
        metrics=[]
        for j in range(GOALS):
            if not bool(eligible[p,j]):
                metrics.append(None)
                continue
            u=k[p,j]-s[p,j]
            g=grad[j]
            finite=float(dk[p,j]-ds[p,j])
            lk=float(g@k[p,j]);ls=float(g@s[p,j])
            linear=lk-ls
            q_k=float(qk[j]);q_s=float(qs[j])
            second=linear-.5*(q_k-q_s)
            geo=diagnostics[p][j]['candidate_geometry']
            dual=inactive_gradient_dual(cov,geo['active_ids'],g,u)
            if abs(float(k[p,j].norm())-float(s[p,j].norm()))>1e-8:
                raise ArithmeticError('registered equal-physical-dose control differs')
            if abs(float(k[p,j]@s[p,j]))>1e-7*(1+float(k[p,j].norm())**2):
                raise ArithmeticError('minimal-norm selected kill and tangent sham not orthogonal')
            metrics.append({
              'gradient_48':g.tolist(),
              'kill_delta_48':k[p,j].tolist(),
              'sham_delta_48':s[p,j].tolist(),
              'finite_kill_minus_sham':finite,
              'kill_margin_drop':float(dk[p,j]),
              'sham_margin_drop':float(ds[p,j]),
              'clean_linear_kill':lk,'clean_linear_sham':ls,
              'clean_linear_prediction':linear,
              'finite_minus_linear':finite-linear,
              'relative_abs_linear_remainder':abs(finite-linear)/(abs(finite)+ZERO),
              'directional_curvature_kill':q_k,
              'directional_curvature_sham':q_s,
              'second_order_local_prediction':second,
              'finite_minus_second_order':finite-second,
              'clean_correct':bool(correct[p,j]),
              'kill_correct_to_wrong':bool(fk[p,j]),
              'sham_correct_to_wrong':bool(fs[p,j]),
              'selected_feature':int(mapped[p,j]),
              'active_feature_ids':geo['active_ids'],
              'inactive_gradient_dual':dual,
            })
        rows.append({
         'puzzle_index':p,'seed':original['seed'],
         'goal_positions':pos[p].tolist(),
         'goal_truth_digits':truth[p].tolist(),
         'source_top_feature_ids':top[p].tolist(),
         'target_matched_feature_ids':mapped[p].tolist(),
         'eligible_mask':eligible[p].tolist(),
         'edit_diagnostics':diagnostics[p],
         'metrics':metrics
        })
    return rows,{
      'selected_goal_tokens':n*GOALS,
      'eligible_goals':int(eligible.sum()),
      'ineligibility_counts':reasons,
      'target_clean_selected_digit_accuracy':float(correct.float().mean()),
      'target_physical_encoder_covectors_96x48':cov.tolist(),
    }


def run(original_artifact,independent_artifact,conditioned_receipt,
        fetch_source=False,source_checkpoint=None,pilot=0,pair_limit=4):
    if pilot not in (0,4) or pair_limit not in (1,2,3,4):
        raise ValueError('only registered SMC18 pilot/full lengths')
    torch.set_num_threads(2)
    teachers,comps,old,assignment,lock=source_models(
        original_artifact,independent_artifact,conditioned_receipt,
        fetch_source,source_checkpoint)
    from smc14_causal_transfer import source_group_context
    rosters,manifests=fresh_rosters(old,limit=pilot or PZ)
    ctx,_=source_group_context(teachers,comps,rosters,tuple(FAMILIES))
    comparisons=[]
    for index,(a,b) in enumerate(PAIRS[:pair_limit]):
        for fam in FAMILIES:
            pass
        heldout={}
        for fam in FAMILIES:
            raw,coverage=evaluate_family(ctx,comps,a,b,assignment[index]['matched'],
                                         rosters[fam],fam,index)
            stats=score_family(raw,seed=202618130+100*index+len(fam))
            heldout[fam]={'per_puzzle':raw,'coverage':coverage,'statistics':stats}
        pass_both=(not bool(pilot or pair_limit<4) and
                   all(heldout[f]['statistics']['gate'] for f in FAMILIES))
        comparisons.append({'source':a,'target':b,
                            'frozen_matched_feature_mapping':assignment[index]['matched'],
                            'original_calibration_assignment_sha256':assignment[index]['hashes'],
                            'heldout':heldout,
                            'registered_gradient_support_both_families':pass_both})
        print('SMC18_TASK_GRADIENT',b,{
          fam:{'eligible':heldout[fam]['statistics']['eligible_goals'],
               'mean_finite':heldout[fam]['statistics'].get('finite_effect_mean'),
               'mean_gradient':heldout[fam]['statistics'].get('linear_prediction_mean'),
               'remainder_ratio':heldout[fam]['statistics'].get('relative_absolute_remainder'),
               'gate':heldout[fam]['statistics']['gate']} for fam in FAMILIES},flush=True)
    return {
      'schema':'SMC-18/task-gradient-inactive-dual-v1',
      'pre_registration_commit':PRECOMMIT,
      **lock,
      'original_conditioned_assignments_frozen_before_heldout':True,
      'all_seven_original_assignment_receipts':assignment,
      'fresh_rosters':rosters,'fresh_manifest_sha256':manifests,
      'test_seed_bases':{k:v[0] for k,v in FAMILIES.items()},
      'puzzles_per_family':pilot or PZ,
      'full_protocol':not bool(pilot or pair_limit<4),
      'fixed_parameters':{'curvature_symmetric_fraction':DIFF,
        'bootstrap_draws':BOOTS,'sign_and_effect_cutoff':ZERO,
        'inactive_minimum_singular_value':1e-10,
        'linear_remainder_ratio_max':.5,
        'sign_agreement_vs_half_ci_lower_strict':0},
      'comparison_roster':comparisons,
      'limit':'FP64 numerical clean autograd and finite task margins; inactive rows may represent all gradients, not causal feature identities.'
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
    p.add_argument('--out',type=Path,default=Path('runs/smc18/full.json'))
    args=p.parse_args()
    result=run(args.original_artifact,args.independent_artifact,args.conditioned_receipt,
               args.fetch_source,args.source_checkpoint,args.pilot_puzzles or 0,args.pair_limit)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(args.out),'full':result['full_protocol'],
         'pair_gates':[{'target':c['target'],
                        'both_families_passed':c['registered_gradient_support_both_families']}
                       for c in result['comparison_roster']]},indent=2))


if __name__=='__main__':
    main()
