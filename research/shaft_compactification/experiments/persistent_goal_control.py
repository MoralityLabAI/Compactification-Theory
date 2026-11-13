"""SHAFT-COMP-20: train known-goal vs behavioral-history linear probes.

Registered protocol: SHAFT_COMP_20_PROTOCOL.md at commit
9201423bc6b5895ed1773a15b683d385b9481b08.
All models are small in-memory synthetic six-query GRUs. No network access,
replication or persistence operations. Training accuracy is never used to
exclude a model from the registered results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from research.shaft_compactification.experiments import architecture_vocab_transfer as base

V=6
GOALS=64
HIDDEN=24
STEPS=2400
BATCH=128
LENGTH=12
PROBE_STEPS=400
PROBE_CONTEXTS=12
STATE_DEPTHS=(12,40)
PER_GOAL=8
DIAGNOSTIC_STEPS=5
OUTCOME_STEPS=32
FUTURE_COUNT=2
CHECKPOINTS=(8,16,32)
REGISTRATION='9201423bc6b5895ed1773a15b683d385b9481b08'
SEEDS={
 'repeat_toggle':{'calibration':(2001,2002),'heldout':(2003,2004)},
 'successor_toggle':{'calibration':(2011,2012),'heldout':(2013,2014)}
}

def target_bits(goals,tokens,family):
    p=torch.cat((torch.zeros_like(tokens[:,:1]),tokens[:,:-1]),dim=1)
    if family=='repeat_toggle':gate=(tokens==p)
    elif family=='successor_toggle':gate=(tokens==(p+1)%V)
    else:raise ValueError(family)
    before=(gate.long().cumsum(1)-gate.long())%2
    return (((goals[:,None]>>tokens)&1)^before).float()

def train_base(family,seed,steps=STEPS):
    torch.manual_seed(seed)
    gen=torch.Generator().manual_seed(10000+seed)
    model=base.RecurrentMemory('gru',V)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    model.train()
    for _ in range(steps):
        goals=torch.randint(GOALS,(BATCH,),generator=gen)
        tokens=torch.randint(V,(BATCH,LENGTH),generator=gen)
        truth=target_bits(goals,tokens,family)
        logits=model(goals,tokens)
        loss=F.binary_cross_entropy_with_logits(logits,truth)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
    model.eval()
    return model,float(loss.detach())

@torch.no_grad()
def test_base(model,family,seed,length):
    gen=torch.Generator().manual_seed(900000+seed+43*length)
    goals=torch.randint(GOALS,(512,),generator=gen)
    tokens=torch.randint(V,(512,length),generator=gen)
    y=target_bits(goals,tokens,family)
    return float(((model(goals,tokens)>=0)==y.bool()).float().mean())

def prefixes(model,goal_ids,depth,seed,family):
    gen=torch.Generator().manual_seed(seed)
    tokens=torch.randint(V,(len(goal_ids),depth),generator=gen)
    with torch.no_grad():
        states=model.embedding(goal_ids)
        for t in range(depth):
            states=model.step(states,F.one_hot(tokens[:,t],V).float())
    prev=torch.cat((torch.zeros_like(tokens[:,:1]),tokens[:,:-1]),dim=1)
    gate=(tokens==prev) if family=='repeat_toggle' else (tokens==(prev+1)%V)
    latch=(gate.long().sum(1)%2).cpu().numpy()
    return states,latch,tokens[:,-1].numpy()

def train_probes(model,seed,family):
    goals0=torch.arange(GOALS).repeat_interleave(PROBE_CONTEXTS)
    states=[];ys=[]
    for depth in (4,8,12):
        gen=torch.Generator().manual_seed(seed*10000+depth+991)
        goals=goals0[torch.randperm(len(goals0),generator=gen)]
        h,_,_=prefixes(model,goals,depth,seed*20000+depth+52,family)
        states.append(h);ys.append(goals)
    X=torch.cat(states);goals=torch.cat(ys)
    with torch.no_grad():
        one=(base.counterfactual(model,X,1)>=0).float()
        dup=one.repeat_interleave(V,dim=1)
        two=(base.counterfactual(model,X,2)>=0).float()
        goal_sig=(((goals[:,None]>>torch.arange(V))&1).float()).repeat_interleave(V,dim=1)
    probes={}
    for index,(key,truth) in enumerate((('goal',goal_sig),('duplicate',dup),('history',two))):
        torch.manual_seed(2000000+seed+1000*index)
        p=nn.Linear(model.dim,V*V)
        opt=torch.optim.Adam(p.parameters(),lr=.04)
        for _ in range(PROBE_STEPS):
            loss=F.binary_cross_entropy_with_logits(p(X),truth)
            opt.zero_grad();loss.backward();opt.step()
        p.eval()
        probes[key]=(p,float(loss.detach()))
    return probes,len(X)

def direction_scores(model,probes,a,b,diagnostic):
    n=len(a);contrast=a-b;length=contrast.norm(dim=1).clamp_min(1e-9)
    scores={key:[] for key in probes}
    with torch.no_grad():
        for states in (a,b):
            h=states.clone();d=contrast.clone()
            sum_energy={key:torch.zeros(n) for key in probes}
            for step in range(DIAGNOSTIC_STEPS+1):
                for key,(probe,_) in probes.items():
                    direction=F.linear(d,probe.weight)
                    sum_energy[key]+=direction.square().sum(1)
                if step<DIAGNOSTIC_STEPS:
                    x=F.one_hot(diagnostic[:,step],V).float()
                    d=model.step_jvp(h,d,x)
                    h=model.step(h,x)
            for key in probes:scores[key].append(sum_energy[key])
        result={}
        for key in probes:
            value=torch.sqrt((scores[key][0]+scores[key][1]).clamp_min(0)/(2*V*V))
            result[key]=value.numpy()
            result[key+'_norm']=(value/length).numpy()
        result['distance']=length.numpy()
    return result

def make_pair_rows(model,probes,seed,depth,family,per_goal=PER_GOAL):
    goals0=torch.arange(GOALS).repeat_interleave(per_goal)
    gen=torch.Generator().manual_seed(6000000+seed*100+depth)
    goals=goals0[torch.randperm(len(goals0),generator=gen)]
    a,latch_a,prev_a=prefixes(model,goals,depth,7000000+seed*100+depth,family)
    b,latch_b,prev_b=prefixes(model,goals,depth,8000000+seed*100+depth,family)
    diag_gen=torch.Generator().manual_seed(9000000+seed*100+depth)
    diagnostic=torch.randint(V,(len(goals),DIAGNOSTIC_STEPS),generator=diag_gen)
    dup=probes['duplicate'][0]
    with torch.no_grad():
        one_a=base.counterfactual(model,a,1)>=0
        one_b=base.counterfactual(model,b,1)>=0
        dup_a=(dup(a)>=0).reshape(len(goals),V,V)[:,:,0]
        dup_b=(dup(b)>=0).reshape(len(goals),V,V)[:,:,0]
        eligible=(one_a==one_b).all(1)&(dup_a==dup_b).all(1)
        goal_a=probes['goal'][0](a)>=0;goal_b=probes['goal'][0](b)>=0
        hist_a=probes['history'][0](a)>=0;hist_b=probes['history'][0](b)>=0
    scored=direction_scores(model,probes,a,b,diagnostic)
    idx=eligible.nonzero().flatten()
    flags=torch.zeros(len(goals),dtype=torch.bool)
    true_flags=torch.zeros(len(goals),dtype=torch.bool)
    checks={t:0 for t in CHECKPOINTS}
    if len(idx):
        future_gen=torch.Generator().manual_seed(10000000+seed*100+depth)
        future=torch.randint(V,(FUTURE_COUNT,len(goals),OUTCOME_STEPS),generator=future_gen)
        la=latch_a[idx.numpy()].copy()
        lb=latch_b[idx.numpy()].copy()
        pa=prev_a[idx.numpy()].copy()
        pb=prev_b[idx.numpy()].copy()
        with torch.no_grad():
            for k in range(FUTURE_COUNT):
                ha,hb=a[idx].clone(),b[idx].clone()
                true_l_a,true_l_b=la.copy(),lb.copy()
                true_p_a,true_p_b=pa.copy(),pb.copy()
                for step in range(1,OUTCOME_STEPS+1):
                    q=future[k,idx,step-1]
                    inp=F.one_hot(q,V).float()
                    ha=model.step(ha,inp);hb=model.step(hb,inp)
                    qn=q.numpy()
                    if family=='repeat_toggle':
                        true_l_a^=(qn==true_p_a).astype(np.int64)
                        true_l_b^=(qn==true_p_b).astype(np.int64)
                    else:
                        true_l_a^=(qn==(true_p_a+1)%V).astype(np.int64)
                        true_l_b^=(qn==(true_p_b+1)%V).astype(np.int64)
                    true_p_a=qn;true_p_b=qn
                    if step in CHECKPOINTS:
                        d=((base.counterfactual(model,ha,1)>=0)!=
                           (base.counterfactual(model,hb,1)>=0)).any(1)
                        flags[idx]|=d
                        checks[step]+=int(d.sum())
                        true_flags[idx]|=torch.from_numpy(true_l_a!=true_l_b)
    rows=[]
    for i in range(len(goals)):
        rec={
            'seed':seed,'depth':depth,'family':family,'pair':i,
            'goal_for_audit_only':int(goals[i]),
            'current_native_agreement':bool((one_a[i]==one_b[i]).all()),
            'current_duplicate_agreement':bool((dup_a[i]==dup_b[i]).all()),
            'eligible':bool(eligible[i]),
            'native_delayed_divergence':int(flags[i]) if eligible[i] else None,
            'oracle_delayed_divergence':int(true_flags[i]) if eligible[i] else None,
            'oracle_latch_same':bool(latch_a[i]==latch_b[i]),
            'oracle_previous_same':bool(prev_a[i]==prev_b[i]),
            'goal_probe_same':bool((goal_a[i]==goal_b[i]).all()),
            'history_probe_same':bool((hist_a[i]==hist_b[i]).all()),
        }
        rec.update({key:float(values[i]) for key,values in scored.items()})
        rows.append(rec)
    with torch.no_grad():
        inspect=a[:128]
        y2=base.counterfactual(model,inspect,2)>=0
        y1=base.counterfactual(model,inspect,1)>=0
        gsig=(((goals[:128,None]>>torch.arange(V))&1).bool()).repeat_interleave(V,dim=1)
        labels={'goal':gsig,'duplicate':y1.repeat_interleave(V,dim=1),'history':y2}
        probe_agreement={key:round(float(((probe(inspect)>=0)==labels[key]).float().mean()),6)
                         for key,(probe,_) in probes.items()}
    return rows,{'eligible':len(idx),'native_positives':int(flags[idx].sum()),
                 'oracle_positives':int(true_flags[idx].sum()),
                 'checkpoint_counts':checks,'probe_teacher_accuracy':probe_agreement}

def summarize(rows):
    elig=[r for r in rows if r['eligible']]
    labels=[r['native_delayed_divergence'] for r in elig]
    names=('goal','duplicate','history','goal_norm','duplicate_norm','history_norm','distance')
    return {
        'total':len(rows),'eligible':len(elig),'positives':sum(labels),
        'oracle_positives':sum(r['oracle_delayed_divergence'] for r in elig),
        'oracle_native_agreement_rate':sum(r['native_delayed_divergence']==r['oracle_delayed_divergence'] for r in elig)/max(1,len(elig)),
        'auc':{key:None if (a:=base.auc([r[key] for r in elig],labels)) is None else round(a,6) for key in names}
    }

def run_model(family,seed,phase,steps=STEPS,per_goal=PER_GOAL):
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    model,loss=train_base(family,seed,steps)
    probes,ncontexts=train_probes(model,seed,family)
    parity=base.test_jvp(model,seed)
    rows=[];depth_checks={}
    for depth in STATE_DEPTHS:
        rr,audit=make_pair_rows(model,probes,seed,depth,family,per_goal)
        rows.extend(rr);depth_checks[str(depth)]={**audit,'summary':summarize(rr)}
    return {
        'experiment':'SHAFT-COMP-20','registration_commit':REGISTRATION,
        'family':family,'phase':phase,'seed':seed,'torch':torch.__version__,'numpy':np.__version__,
        'training_steps':steps,'state_dimension':model.dim,'model_parameters':sum(p.numel() for p in model.parameters()),
        'base_model_sha256':base.stable_sha(model),'train_loss':loss,'jvp_parity_max':parity,
        'accuracy_d12':test_base(model,family,seed,12),'accuracy_d32':test_base(model,family,seed,32),
        'probes':{key:{'outputs':probe.out_features,'parameters':sum(p.numel() for p in probe.parameters()),
                        'train_loss':fitloss,'sha256':base.stable_sha(probe)} for key,(probe,fitloss) in probes.items()},
        'probe_training_contexts':ncontexts,'per_goal':per_goal,'depth_checks':depth_checks,
        'summary':summarize(rows),'rows':rows,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=tuple(SEEDS),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--phase',choices=('calibration','heldout'),required=True)
    parser.add_argument('--train-steps',type=int,default=STEPS)
    parser.add_argument('--per-goal',type=int,default=PER_GOAL)
    parser.add_argument('--threshold-file',type=Path)
    parser.add_argument('--threshold-sha256')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.seed not in SEEDS[args.family][args.phase]:
        raise ValueError('seed not registered')
    if args.phase=='heldout':
        if args.train_steps!=STEPS or args.per_goal!=PER_GOAL:
            raise ValueError('heldout requires registered sample/train budget')
        if not args.threshold_file or not args.threshold_sha256:
            raise ValueError('heldout requires prior frozen threshold receipt')
        raw=args.threshold_file.read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest()!=args.threshold_sha256:
            raise ValueError('calibration SHA256 mismatch')
        frozen=json.loads(raw)
        if frozen.get('registration_commit')!=REGISTRATION or frozen.get('phase')!='calibration_thresholds':
            raise ValueError('invalid preregistered source thresholds')
    result=run_model(args.family,args.seed,args.phase,args.train_steps,args.per_goal)
    payload=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload,end='')

if __name__=='__main__':
    main()
