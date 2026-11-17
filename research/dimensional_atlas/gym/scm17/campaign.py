"""SCM-17 trained RMP transfer. Requires *external* genuine RMP artifacts.

Preflight is read-only. No training or results are fabricated when artifacts
are absent. Training freezes the original RMP model and D-shared dictionary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict

import torch
from torch.nn import functional as F

from .rmp_fiber import SAEVisitFiber, calibrate_decision_frame, causal_probe, run_with_adapter, tensor_digest

CELLS={'LOOP-T s1':('LOOP-T-core4-ds-s1','C3'),
       'FF-U s3':('FF-U-core4-ds-inj-s3','C3')}
CORE=('pointer_chase','gated_pointer_chase','reachability','sudoku4')
RMP_FILES=('research_gym/neural/looped_decoder.py',
 'research_gym/neural/schedule_executor.py',
 'research_gym/interp/analysis.py',
 'research_gym/interp/phase2.py',
 'research_gym/interp/dictionaries.py',
 'research_gym/interp/models.py')
SCHEMA='scm17/rmp-trained-fiber/v1'


def file_hash(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def source_manifest(args)->dict:
    """Fail-closed readiness without importing RMP or unpickling checkpoints."""
    if args.model not in CELLS:raise ValueError("unknown registered model")
    missing=[];digests={}
    required={f'RMP_source/{path}':args.rmp_root/path for path in RMP_FILES}
    own=Path(__file__).resolve().parent
    required.update({f'SCM17_source/{name}':own/name
                     for name in ('campaign.py','rmp_fiber.py')})
    required.update({'checkpoint':args.artifacts/CELLS[args.model][0]/'checkpoints'/f'{CELLS[args.model][1]}.pt',
       'training_config':args.runs/CELLS[args.model][0]/'result.json',
       'dictionary_metrics':args.metrics,
       'selected_D_shared':args.dictionary})
    ck=required['checkpoint']
    required['checkpoint_integrity_receipt']=ck.with_name(ck.name+'.receipt.json')
    for key,path in required.items():
        if not path.is_file():missing.append(key)
        else:digests[key]=file_hash(path)
    if 'checkpoint' in digests and 'checkpoint_integrity_receipt' in digests:
        try:
            receipt=json.loads(required['checkpoint_integrity_receipt'].read_text(encoding='utf-8'))
            if receipt.get('sha256')!=digests['checkpoint'] or receipt.get('bytes')!=ck.stat().st_size:
                missing.append('checkpoint_integrity_mismatch')
        except (ValueError,TypeError,OSError):
            missing.append('checkpoint_integrity_invalid')
    if not args.monitor_bundle.is_dir():missing.append('monitor_bundle')
    return {'schema':SCHEMA,'status':'ready' if not missing else 'blocked_external_artifacts',
            'model':args.model,'missing':sorted(missing),'hashes':digests,
            'source':{key:str(value) for key,value in required.items()},
            'splits':{'fit':'monitor_train','frame':'monitor_calibration/selection_half',
                      'readout':'monitor_calibration/reporting_half'},
            'scientific_status':{'trained_RMP_measured':False,
                                 'exact_nonlinear_head_kernel_claimed':False,
                                 'unopened_sealed_test':True}}


def choose_rows(data:dict, *,size:int,selection:bool|None,seed:int)->list[int]:
    """Group-disjoint balanced case selection, independent of target labels."""
    if size<4:raise ValueError('size must be at least 4')
    per,extra=divmod(size,len(CORE))
    chosen=[]
    for fam_idx,family in enumerate(CORE):
        seen=set();offers=[]
        for index,(fam,group) in enumerate(zip(data['families'],data['groups'])):
            if str(fam)!=family or group in seen:continue
            if selection is not None:
                belongs=(hashlib.sha256(str(group).encode()).digest()[0]%2==0)
                if belongs!=selection:continue
            seen.add(group)
            h=hashlib.sha256(f'{seed}/{family}/{group}/{data["ids"][index]}'.encode()).hexdigest()
            offers.append((h,index))
        quota=per+int(fam_idx<extra)
        if len(offers)<quota:raise ValueError(f'insufficient {family} groups for {quota} cases')
        chosen.extend(index for _,index in sorted(offers)[:quota])
    return chosen


def assert_source_dictionary(dictionary_path:Path,metrics:Path,model_name:str,hidden:int):
    """Strictly load state dictionaries with weights_only=True and RMP selection."""
    from research_gym.interp.dictionaries import TopKDictionary
    from research_gym.interp.phase2 import select_config
    doc=json.loads(metrics.read_text(encoding='utf-8'))
    if model_name not in doc:raise ValueError('no dictionary selection metrics for model')
    chosen,met=select_config(doc[model_name])
    expected=doc[model_name][chosen]['config']
    saved=torch.load(dictionary_path,map_location='cpu',weights_only=True)
    shape=saved['shape'];actual=saved['summary']['config']
    for field in ('kind','width','k','seed','site'):
        if actual[field]!=expected[field]:raise ValueError('D-shared selection metadata mismatch: '+field)
    if shape['dim']!=hidden or shape['width']!=expected['width'] or shape['k']!=expected['k'] or shape['skip']:
        raise ValueError('D-shared tensor metadata mismatch')
    if actual['kind']!='shared' or actual['seed']!=0 or actual['site']!=-1:
        raise ValueError('D-shared calibration provenance mismatch')
    model=TopKDictionary(hidden,shape['width'],shape['k'],skip=False)
    model.load_state_dict(saved['state'],strict=True)
    if not all(bool(torch.isfinite(w).all()) for w in model.parameters()):
        raise ValueError('nonfinite D-shared parameters')
    return model.eval(),{'config':chosen,'selection_bar_met':met}


def state_digest(model)->str:
    hasher=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode());hasher.update(tensor_digest(tensor).encode())
    return hasher.hexdigest()


def score(analysis,tokens,targets,adapter=None,*,sector='joint',zero_codes=False,
          visit_order=None,batch_size=16,device='cpu')->dict:
    """Task-correct predictions under identical RMP answer-token grading."""
    correct=0;loss=0.;n=len(targets);trace=[]
    with torch.no_grad():
        for start in range(0,n,batch_size):
            t=tokens[start:start+batch_size].to(device)
            y=targets[start:start+batch_size].long().to(device)
            out,_=run_with_adapter(analysis,t,adapter,sector=sector,
                                   zero_codes=zero_codes,visit_order=visit_order)
            logits=out.logits[:,-1,:].float()
            loss+=float(F.cross_entropy(logits,y,reduction='sum'))
            pred=logits.argmax(dim=-1)
            correct+=int((pred==y).sum().item())
            trace.extend(pred.detach().cpu().tolist())
    return {'n':n,'correct':correct,'accuracy':correct/n,'mean_ce':loss/n,
            'predictions_sha256':hashlib.sha256(json.dumps(trace).encode()).hexdigest(),
            'model_forward_visits':n*analysis.model.expanded_visits}


def train_adapter(analysis,adapter,tokens,targets,*,steps:int,seed:int,
                  batch_size:int,learning_rate:float,device:str)->dict:
    if steps<0 or batch_size<1 or learning_rate<=0:raise ValueError('invalid training controls')
    if steps==0:return {'steps':0,'optimizer_calls':0,'train_last_ce':None}
    torch.manual_seed(seed)
    optimizer=torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad),lr=learning_rate)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    losses=[]
    for step in range(steps):
        idx=torch.randint(len(targets),(batch_size,),generator=gen)
        inp=tokens[idx].to(device);gold=targets[idx].long().to(device)
        out,info=run_with_adapter(analysis,inp,adapter)
        correction=sum((entry['delta_sq'] for entry in info),inp.new_zeros(()))
        ce=F.cross_entropy(out.logits[:,-1,:].float(),gold)
        objective=ce+0.01*correction
        if not torch.isfinite(objective):raise ValueError('nonfinite training objective')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_((p for p in adapter.parameters() if p.requires_grad),1.0)
        optimizer.step()
        losses.append(float(ce.detach()))
    return {'steps':steps,'optimizer_calls':steps,
            'train_last_ce':losses[-1],
            'train_first_ce':losses[0],
            'source_model_forward_visits':steps*batch_size*analysis.model.expanded_visits}


def execute(args)->dict:
    if args.model not in CELLS:raise ValueError('unknown registered RMP model')
    if args.device!='cpu' and not args.device.startswith('cuda'):
        raise ValueError('device must be cpu or cuda')
    if args.output.exists() or args.output.with_suffix('.adapter.pt').exists():
        raise FileExistsError('output or adapter checkpoint already exists')
    source=source_manifest(args)
    if source['status']!='ready':return source
    if args.lock_file is None or not args.lock_file.is_file():
        raise ValueError('trained experiment requires frozen preflight --lock-file')
    if json.loads(args.lock_file.read_text(encoding='utf-8'))!=source:
        raise ValueError('RMP source/artifact hashes differ from frozen preflight lock')
    sys.path.insert(0,str(args.rmp_root.resolve()))
    from research_gym.interp.phase2 import core_examples,load_model
    device=torch.device(args.device)
    analysis=load_model(args.model,args.runs,args.artifacts,device)
    original=state_digest(analysis.model)
    for p in analysis.model.parameters():p.requires_grad_(False)
    analysis.model.eval()
    trained=core_examples(args.monitor_bundle,'monitor_train')
    calibration=core_examples(args.monitor_bundle,'monitor_calibration')
    train_ix=choose_rows(trained,size=args.train_cases,selection=None,seed=args.seed)
    frame_ix=choose_rows(calibration,size=args.frame_cases,selection=True,seed=args.seed)
    eval_ix=choose_rows(calibration,size=args.eval_cases,selection=False,seed=args.seed)
    train_tokens=trained['tokens'][train_ix]; train_targets=trained['targets'][train_ix]
    frame_tokens=calibration['tokens'][frame_ix]
    eval_tokens=calibration['tokens'][eval_ix]
    frame=calibrate_decision_frame(analysis,frame_tokens.to(device),
           contrasts=args.contrasts,max_rank=args.frame_rank,calibration_cap=args.frame_cases)
    if not (0<args.min_frame_energy<=1):raise ValueError('invalid frame energy threshold')
    if frame.captured_gradient_energy<args.min_frame_energy:
        raise ValueError('insufficient calibration-gradient capture: %.5f < %.5f' %
                         (frame.captured_gradient_energy,args.min_frame_energy))
    sae,sae_receipt=assert_source_dictionary(args.dictionary,args.metrics,args.model,analysis.model.hidden_size)
    sae=sae.to(device)
    torch.manual_seed(args.seed)
    adapter=SAEVisitFiber(sae,frame,rank=args.adapter_rank,
                          visits=analysis.model.expanded_visits).to(device)
    trained_report=train_adapter(analysis,adapter,train_tokens,train_targets,
                       steps=args.steps,seed=args.seed+1,batch_size=args.batch,
                       learning_rate=args.lr,device=args.device)
    # Only score reporting-half labels after all adapter optimization is complete.
    eval_targets=calibration['targets'][eval_ix]
    before=score(analysis,eval_tokens,eval_targets,device=args.device)
    arms={'tied':before}
    visits=analysis.model.expanded_visits
    for name,sector,opts in (('joint','joint',{}),('visible_only','visible_only',{}),
             ('fiber_only','fiber_only',{}),('shuffled_visits','joint',{'visit_order':tuple(reversed(range(visits)))}),
             ('zero_sae_codes','joint',{'zero_codes':True})):
        arms[name]=score(analysis,eval_tokens,eval_targets,adapter,sector=sector,
                        device=args.device,**opts)
    probe=causal_probe(analysis,frame_tokens[:1].to(device),frame,visit_count=visits)
    if original!=state_digest(analysis.model):raise AssertionError('frozen source model changed')
    out={'schema':SCHEMA,'status':'trained_RMP_adapter_completed',
         'evidence':{'trained_RMP_adapter_executed':True,
                     'floating_point_source_only':True,
                     'new_sealed_test_evaluated':False,
                     'head_readout_exact_kernel':False},
         'source_manifest':source,'adapter':{'frame_rank':int(frame.q.shape[1]),
              'gradient_energy':frame.captured_gradient_energy,
              'source_logits_sha256':frame.source_logits_sha256,
              'selected_D_shared':sae_receipt,
              'trainable_parameters':sum(p.numel() for p in adapter.parameters() if p.requires_grad)},
         'training':trained_report,'arms':arms,'causal_probe':probe,
         'design':{'readout':'nonlinear_local_margin_first_order_only',
               'reporting_half':'previously_evaluated_RMP_calibration_not_new_sealed',
               'task_group_alignment':'frozen_by_group_sha256',
               'SAE_source':'frozen_D_shared_RMP',
               'G0_unchanged':True,'exact_topology_certified':False,
               'source_parameters_changed':False,
               'full_compute_matching':False}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    torch.save({'model':adapter.state_dict(),
                'source_model_sha256':original,
                'dictionary_sha256':source['hashes']['selected_D_shared'],
                'schema':SCHEMA},args.output.with_suffix('.adapter.pt'))
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','freeze','train'),default='preflight')
    p.add_argument('--lock-file',type=Path)
    p.add_argument('--model',choices=tuple(CELLS),default='LOOP-T s1')
    p.add_argument('--rmp-root',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--monitor-bundle',type=Path,required=True)
    p.add_argument('--dictionary',type=Path,required=True)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--batch',type=int,default=12)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--train-cases',type=int,default=96)
    p.add_argument('--frame-cases',type=int,default=8)
    p.add_argument('--eval-cases',type=int,default=64)
    p.add_argument('--frame-rank',type=int,default=8)
    p.add_argument('--min-frame-energy',type=float,default=.75)
    p.add_argument('--contrasts',type=int,default=2)
    p.add_argument('--adapter-rank',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261114)
    args=p.parse_args(argv)
    if args.output.exists():p.error('refuse overwrite')
    if args.mode=='preflight':result=source_manifest(args)
    elif args.mode=='freeze':
        result=source_manifest(args)
        if result['status']!='ready':
            raise ValueError('cannot freeze incomplete external artifacts')
        if args.lock_file is None:raise ValueError('freeze needs --lock-file')
        args.lock_file.parent.mkdir(parents=True,exist_ok=True)
        with args.lock_file.open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    else:result=execute(args)
    # Preflight must itself produce a precise report without writing output.
    print(json.dumps(result if args.mode!='train' else
                     {'status':result['status'],'arms':result.get('arms'),
                      'output':str(args.output)},indent=2,sort_keys=True))
    return result

if __name__=='__main__':main()
