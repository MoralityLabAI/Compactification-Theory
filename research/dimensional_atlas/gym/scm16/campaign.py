"""SCM-16 reproducible SAE/fiber-VPD looped-math training and eval campaign.

Heldout EvalMath rosters and generated challenge gold are accessed only after
all training, peak selection and model digests are frozen. The oracle remains
external, independently pinned by SCM-15; no EvalMath source is vendored.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import io
import json
import math
import random
import time
from pathlib import Path
from typing import Sequence
import torch
from ..tasks import PROGRAMS, MathTask, load_evalmath, encode, verify_order, make_toy
from ..run import _make_pool, optimizer_steps, tensorize, model_digest
from ..model import atlas_probe
from .fiber import FiberLoopedMathTransformer
from .landscape import mine_discovery, generate_fresh_challenges

VERSION='scm16.fiber_atvpd.evalmath.v1'
POLICIES=('tied','joint','visible_only','fiber_only','shuffled_visit','zero_codes')


def canonical_sha(payload) -> str:
    return sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def base_digest(model:FiberLoopedMathTransformer)->str:
    raw={k:v.detach().cpu() for k,v in model.state_dict().items() if
         k.startswith(('embed.','core.','head.','head_norm.','positions'))}
    buf=io.BytesIO();torch.save(raw,buf)
    return sha256(buf.getvalue()).hexdigest()


def evaluate(model:FiberLoopedMathTransformer,tasks:Sequence[MathTask],*,oracle=None,
             budget:int=12,policy:str='tied') -> dict:
    if policy not in POLICIES:raise ValueError('unknown policy')
    if budget<1:raise ValueError('budget must be positive')
    model.eval()
    buckets=defaultdict(list);rows=[]
    for task in tasks:
        x=encode(task).unsqueeze(0)
        kwargs={'use_vpd':policy!='tied'}
        if policy=='visible_only':kwargs['sector']='visible_only'
        if policy=='fiber_only':kwargs['sector']='fiber_only'
        if policy=='shuffled_visit':kwargs['visit_order']=tuple(reversed(range(model.visits)))
        if policy=='zero_codes':kwargs['code_override']='zero'
        with torch.no_grad():
            logits=model(x,**kwargs).logits[0]
            ranked=logits.argsort(descending=True).tolist()
        grade=verify_order(task,ranked,budget=budget,oracle=oracle)
        split=getattr(task.oracle_case,'split',None) if task.oracle_case is not None else task.family
        item={'case_id':task.case_id,'public_signature':task.signature,
              'family':task.family,'eval_split':split,
              'future_correct':int(grade['future_correct']),
              'prefix_accepted':int(grade['accepted']),
              'false_future_acceptance':int(grade['false_future_acceptance']),
              'verifier_queries':int(grade['queries'])}
        rows.append(item);buckets[(task.family,split)].append(item)
    def agg(rs):
        if not rs:return {'n':0,'future_correct':0,'future_accuracy':None,'false_future_acceptances':0,'mean_queries':None}
        n=len(rs)
        return {'n':n,'future_correct':sum(v['future_correct'] for v in rs),
                'future_accuracy':round(sum(v['future_correct'] for v in rs)/n,6),
                'prefix_acceptances':sum(v['prefix_accepted'] for v in rs),
                'false_future_acceptances':sum(v['false_future_acceptance'] for v in rs),
                'mean_queries':round(sum(v['verifier_queries'] for v in rs)/n,6),
                'total_verifier_queries':sum(v['verifier_queries'] for v in rs)}
    return {'overall':agg(rows),'groups':{f'{f}/{s}':agg(r) for (f,s),r in sorted(buckets.items())},
            'records':rows}


def symbolic_reference(tasks:Sequence[MathTask],*,oracle=None,budget:int=12)->dict:
    """Fixed public-syntax exact solver, charged separately from neural calls."""
    counts={'n':len(tasks),'correct':0,'accepted':0,'false_future_acceptances':0,'queries':0}
    for task in tasks:
        proof=verify_order(task,list(range(len(PROGRAMS))),budget=budget,oracle=oracle)
        counts['correct']+=int(proof['future_correct'])
        counts['accepted']+=int(proof['accepted'])
        counts['false_future_acceptances']+=int(proof['false_future_acceptance'])
        counts['queries']+=proof['queries']
    counts['accuracy']=round(counts['correct']/len(tasks),6) if tasks else None
    counts['mean_queries']=round(counts['queries']/len(tasks),6) if tasks else None
    counts['budget_per_task']=budget
    return counts


def paired_counts(a,b)->dict:
    left={r['public_signature']+'|'+r['case_id']:r for r in a['records']}
    right={r['public_signature']+'|'+r['case_id']:r for r in b['records']}
    if set(left)!=set(right):raise ValueError('unmatched paired tasks')
    gain=loss=tie=0
    for key in left:
        va=left[key]['future_correct'];vb=right[key]['future_correct']
        if vb>va:gain+=1
        elif va>vb:loss+=1
        else:tie+=1
    return {'joint_only_correct':gain,'tied_only_correct':loss,'same_outcome':tie}


def load_public_evalmath(oracle):
    from ..tasks import MathTask
    roster=oracle.roster(include_test=True)
    readout=[]
    for key,items in roster.items():
        if key=='calibration':continue
        for case in items:
            graph=tuple(case.source_graph) if case.source_graph is not None else PROGRAMS[0]
            readout.append(MathTask(case.identifier,case.family,case.topology,
                tuple((tuple(a[:13]),tuple(b[:13])) for a,b in case.demos),
                tuple(case.query[:13]),tuple(case.future),graph,'evalmath_arg01',case))
    return readout


def _to_public(tasks:Sequence[MathTask]):
    return [{**t.public,'public_sha256':t.signature,'source':'scm16_discovery_local_maximum_fresh_seed'} for t in tasks]


def _to_gold(tasks:Sequence[MathTask]):
    return [{'case_id':t.case_id,'public_sha256':t.signature,
             'source_program':list(t.source_graph),'future':list(t.hidden_target)} for t in tasks]


def fiber_mediation(model:FiberLoopedMathTransformer,task:MathTask,*,strength:float=.05)->dict:
    """Fixed-direction, matched-norm intervention. Exploratory causal diagnostic.

    A fiber patch at the final visit is exactly null under the linear readout.
    Earlier patches may move into the visible subspace after recurrent updates.
    The perturbation direction is input-independent and its norm is frozen.
    """
    if strength<=0:raise ValueError('strength must be positive')
    q=model.vpd.readout_basis.detach()
    v=torch.arange(1,model.hidden+1,dtype=q.dtype)
    fib=v-(v@q)@q.T
    if float(fib.norm())<1e-6:raise ValueError('fiber probe collapsed')
    fib=fib/fib.norm();vis=q[:,0]
    x=encode(task).unsqueeze(0)
    with torch.no_grad():
        clean=model(x,use_vpd=True).logits
        trace={'fiber':[],'visible':[]}
        for t in range(model.visits):
            for label,direction in (('fiber',fib),('visible',vis)):
                post=model(x,use_vpd=True,
                      state_patch=(t,(strength*direction).unsqueeze(0),3)).logits
                trace[label].append(round(float((post-clean).norm()),10))
    return {'status':'finite_float32_counterfactual_only',
            'strength':strength,'token':3,'by_visit':trace,
            'last_visit_fiber_is_readout_null':trace['fiber'][-1]<=2e-5}


def run(args):
    if args.output.exists():raise FileExistsError('receipt would overwrite existing file')
    if args.backend=='evalmath':
        if args.evalmath_source is None:raise ValueError('pinned EvalMath source is required')
        oracle=load_evalmath(args.evalmath_source)
    else:oracle=None
    for name in ('seed_count','train_count','discovery_count','challenge_per_peak','mining_tasks','peaks','budget'):
        if getattr(args,name)<1:raise ValueError(name+' must be positive')
    torch.set_num_threads(1)
    source_files=[Path(__file__),Path(__file__).with_name('fiber.py'),
                  Path(__file__).with_name('landscape.py')]
    sources={p.name:sha256(p.read_bytes()).hexdigest() for p in source_files}
    train=_make_pool(args.train_count,110000,backend=args.backend,oracle=oracle,split='train')
    discovery=_make_pool(args.discovery_count,220000,backend=args.backend,oracle=oracle,split='discovery')
    training_seen={t.signature for t in train+discovery}
    if len(training_seen)!=len(train)+len(discovery):raise ValueError('overlap in training and discovery')
    started=time.monotonic()
    frontier=mine_discovery(discovery,oracle=oracle,max_tasks=args.mining_tasks,peaks=args.peaks)
    replay=generate_fresh_challenges(frontier['peaks'],backend=args.backend,oracle=oracle,
                                    per_peak=args.replay_per_peak,seed_base=700000,
                                    forbidden=training_seen)
    complete_seen=training_seen|{t.signature for t in replay}
    if len(complete_seen)!=len(training_seen)+len(replay):raise AssertionError('replay signature overlap')
    frozen=[]
    for si in range(args.seed_count):
        seed=args.seed+si*997
        torch.manual_seed(seed)
        model=FiberLoopedMathTransformer(hidden=args.hidden,heads=args.heads,
                visits=args.visits,sae_width=args.sae_width,sae_k=args.sae_k,
                vpd_rank=args.vpd_rank,readout_dim=args.readout_dim)
        phases={}
        phases['base']=optimizer_steps(model,train,stage='base',steps=args.base_steps,batch=args.batch,
                                      seed=seed+1,lr=args.lr)
        model.lock_fiber()
        frozen_base=base_digest(model)
        phases['sae']=optimizer_steps(model,train,stage='sae',steps=args.sae_steps,batch=args.batch,
                                      seed=seed+2,lr=args.lr)
        phases['at_vpd']=optimizer_steps(model,train,stage='vpd',steps=args.vpd_steps,batch=args.batch,
                                         seed=seed+3,lr=args.lr)
        phases['peak_replay']=optimizer_steps(model,train+replay,stage='vpd',steps=args.replay_steps,
                                              batch=args.batch,seed=seed+4,lr=args.lr)
        if base_digest(model)!=frozen_base:raise AssertionError('frozen baseline mutated during adaptation')
        frozen.append({'model':model,'seed':seed,'base_sha256':frozen_base,
                       'model_sha256':model_digest(model),'phases':phases})
    # OFFICIAL TEST AND FRESH-GENERATED CHALLENGE GOLD ENTER HERE, AFTER FREEZE.
    official=load_public_evalmath(oracle) if oracle is not None else [
        make_toy(330000+i,case_id=f'readout:{i}',family='periodic') for i in range(args.readout_count)]
    official_sig={t.signature for t in official}
    # EvalMath's intentionally paired finite-prefix collisions share visible signatures.
    if official_sig & complete_seen:raise ValueError('official test signature overlaps feedback')
    new_challenges=generate_fresh_challenges(frontier['peaks'],backend=args.backend,oracle=oracle,
                      per_peak=args.challenge_per_peak,seed_base=990000,
                      forbidden=complete_seen|official_sig)
    challenge_sig={t.signature for t in new_challenges}
    if len(challenge_sig)!=len(new_challenges):raise ValueError('new challenge signatures collide')
    symbolic_controls={'first12':symbolic_reference(official,oracle=oracle,budget=12),
                       'exhaustive64':symbolic_reference(official,oracle=oracle,budget=64)}
    summary=[]
    for item in frozen:
        model=item['model'];model.eval()
        trial={policy:evaluate(model,official,oracle=oracle,budget=args.budget,policy=policy) for policy in POLICIES}
        challenge={policy:evaluate(model,new_challenges,oracle=oracle,budget=args.budget,policy=policy)
                   for policy in ('tied','joint','visible_only','fiber_only')}
        with torch.no_grad():
            trace=model(encode(discovery[0]).unsqueeze(0),use_vpd=True)
            correction_energies={
                'visible':[round(float(t),10) for t in trace.visible_energies],
                'fiber':[round(float(t),10) for t in trace.fiber_energies]}
        q=model.vpd.readout_basis
        readout_error=float((model.head[0].weight.detach()@(torch.eye(args.hidden)-q@q.T)).abs().max())
        summary.append({'seed':item['seed'],'base_sha256':item['base_sha256'],
                        'model_sha256':item['model_sha256'],'phases':item['phases'],
                        'readout_kernel_residual':readout_error,
                        'correction_energies':correction_energies,
                        'official':{k:{'overall':v['overall'],'groups':v['groups']} for k,v in trial.items()},
                        'new_challenges':{k:{'overall':v['overall'],'groups':v['groups']} for k,v in challenge.items()},
                        'paired_official_joint_vs_tied':paired_counts(trial['tied'],trial['joint']),
                        'dimensional_diagnostic':atlas_probe(model,discovery[0]),
                        'fiber_mediation_diagnostic':fiber_mediation(model,discovery[0])})
        if model_digest(model)!=item['model_sha256']:
            raise AssertionError('evaluation altered frozen model')
    public=_to_public(new_challenges);private=_to_gold(new_challenges)
    public_content=json.dumps(public,indent=2,sort_keys=True)+'\n'
    private_content=json.dumps(private,indent=2,sort_keys=True)+'\n'
    receipt={'schema':VERSION,'backend':args.backend,
             'status':'executed_exact_synthetic_gym' if args.backend=='toy' else 'executed_pinned_evalmath_gym',
             'scientific_scope':'finite-dimensional toy trained weights; chart geometry observational',
             'source_sha256':sources,
             'oracle_sha256':sha256(args.evalmath_source.read_bytes()).hexdigest() if oracle else None,
             'frozen_protocol':{k:getattr(args,k) for k in vars(args)
                   if k not in ('output','evalmath_source')},
             'train_count':len(train),'discovery_count':len(discovery),
             'readout_count':len(official),'official_unique_public_signatures':len(official_sig),
             'mining':frontier,'replay_training_count':len(replay),
             'symbolic_controls':symbolic_controls,
             'fresh_challenge_count':len(public),
             'new_public_sha256':sha256(public_content.encode()).hexdigest(),
             'new_private_sha256':sha256(private_content.encode()).hexdigest(),
             'training_steps_per_seed':args.base_steps+args.sae_steps+args.vpd_steps+args.replay_steps,
             'forward_work_note':'Base/SAE/VPD stage forward+backward updates are separately counted; symbolic exact queries omit neural compute.',
             'results':summary,'seconds':round(time.monotonic()-started,3),
             'interpretation':'head fiber is a known linear null of the final readout; earlier fiber changes can affect later computation; one seeded task family is not an independent statistical unit',
             'budget_note':'Exact verifier queries and 64x discovery landscape searches are counted separately; extra VPD FLOPs are not compute matched.'}
    output=args.output
    public_path=output.with_suffix('.generated.public.json')
    private_path=output.with_suffix('.gold.private.json')
    for p in (output,public_path,private_path):
        if p.exists():raise FileExistsError('refuse output overwrite: '+str(p))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    public_path.write_text(public_content)
    private_path.write_text(private_content)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('toy','evalmath'),default='toy')
    p.add_argument('--evalmath-source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=160)
    p.add_argument('--seed-count',type=int,default=2)
    p.add_argument('--train-count',type=int,default=384)
    p.add_argument('--discovery-count',type=int,default=96)
    p.add_argument('--readout-count',type=int,default=96)
    p.add_argument('--mining-tasks',type=int,default=24)
    p.add_argument('--peaks',type=int,default=4)
    p.add_argument('--replay-per-peak',type=int,default=4)
    p.add_argument('--challenge-per-peak',type=int,default=4)
    p.add_argument('--base-steps',type=int,default=120)
    p.add_argument('--sae-steps',type=int,default=60)
    p.add_argument('--vpd-steps',type=int,default=180)
    p.add_argument('--replay-steps',type=int,default=60)
    p.add_argument('--batch',type=int,default=24)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--readout-dim',type=int,default=8)
    p.add_argument('--visits',type=int,default=4)
    p.add_argument('--sae-width',type=int,default=64)
    p.add_argument('--sae-k',type=int,default=6)
    p.add_argument('--vpd-rank',type=int,default=4)
    p.add_argument('--budget',type=int,default=12)
    args=p.parse_args(argv)
    if any(getattr(args,k)<0 for k in ('base_steps','sae_steps','vpd_steps','replay_steps')):
        p.error('step budgets must be nonnegative')
    if args.hidden%args.heads or not 0<args.readout_dim<args.hidden:
        p.error('invalid hidden/readout/head geometry')
    out=run(args)
    print(json.dumps({'receipt':str(args.output),'status':out['status'],
        'results':[{'seed':r['seed'],
            'official':{k:v['overall'] for k,v in r['official'].items()},
            'fresh':{k:v['overall'] for k,v in r['new_challenges'].items()}}
            for r in out['results']]},indent=2))

if __name__=='__main__':main()
