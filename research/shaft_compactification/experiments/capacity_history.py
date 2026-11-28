"""SHAFT-COMP-19: preregistered equal-capacity history-probe transfer (toy only).

64 goal truth tables, six query symbols, GRU/LSTM, lag_rotate/lag_match.
Frozen output dimensions W1=6, Wdup=W2=36; calibration before holdout.
Entirely local and synthetic. No network/device/persistence operations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from research.shaft_compactification.experiments import architecture_vocab_transfer as base

V = 6
HIDDEN = 24
TRAIN_STEPS = 1600
BATCH = 128
TRAIN_LENGTH = 12
PROBE_STEPS = 400
DIAGNOSTIC_STEPS = 5
OUTCOME_STEPS = 32
CHECKPOINTS = (8, 16, 32)
FUTURE_COUNT = 2
GOALS = 64
PER_GOAL = 16
STATE_DEPTHS = (12, 40)
SEEDS = {
    'gru_rotate': {'arch':'gru', 'family':'lag_rotate', 'calibration':1901, 'heldout':1911},
    'lstm_rotate': {'arch':'lstm', 'family':'lag_rotate', 'calibration':1902, 'heldout':1912},
    'gru_match': {'arch':'gru', 'family':'lag_match', 'calibration':1903, 'heldout':1913},
    'lstm_match': {'arch':'lstm', 'family':'lag_match', 'calibration':1904, 'heldout':1914},
}
REGISTRATION_COMMIT = '6b063812cfbf913910e1eed60f33b279a64bf87a'


def targets(goals, tokens, family):
    prev = torch.cat((torch.zeros_like(tokens[:, :1]),tokens[:,:-1]),dim=1)
    if family=='lag_rotate':
        slots=(tokens+prev)%V
        return ((goals[:,None]>>slots)&1).float()
    if family=='lag_match':
        return (((goals[:,None]>>tokens)&1) ^ (tokens==prev).long()).float()
    raise ValueError('family must be lag_rotate or lag_match')


def train_model(arch, family, seed, steps=TRAIN_STEPS):
    torch.manual_seed(seed)
    g=torch.Generator().manual_seed(10000+seed)
    model=base.RecurrentMemory(arch,V)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    model.train()
    for _ in range(steps):
        goals=torch.randint(GOALS,(BATCH,),generator=g)
        tokens=torch.randint(V,(BATCH,TRAIN_LENGTH),generator=g)
        truth=targets(goals,tokens,family)
        logits=model(goals,tokens)
        loss=F.binary_cross_entropy_with_logits(logits,truth)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),2.)
        optimizer.step()
    model.eval()
    return model,round(float(loss.detach()),9)


@torch.no_grad()
def test_task(model,family,seed,length=12,n=512):
    g=torch.Generator().manual_seed(900000+seed+43*length)
    goals=torch.randint(GOALS,(n,),generator=g)
    tokens=torch.randint(V,(n,length),generator=g)
    return round(float(((model(goals,tokens)>=0)==targets(goals,tokens,family).bool()).float().mean()),6)


def train_probes(model,seed):
    # Identical states and exact native labels for all probes, 64*12*3=2304 contexts.
    basegoals=torch.arange(GOALS).repeat_interleave(12)
    blocks=[]
    for depth in (4,8,12):
        gen=torch.Generator().manual_seed(seed*10000+depth+991)
        goals=basegoals[torch.randperm(len(basegoals),generator=gen)]
        blocks.append(base.sample_prefixes(model,depth,goals,seed*20000+depth+52))
    states=torch.cat(blocks)
    with torch.no_grad():
        y1=(base.counterfactual(model,states,1)>=0).float()
        y2=(base.counterfactual(model,states,2)>=0).float()
        ydup=y1.repeat_interleave(V,dim=1)
    fits={}
    for i,(name,truth) in enumerate((('instant',y1),('duplicate',ydup),('history',y2))):
        torch.manual_seed(2000000+seed+i*1000)
        probe=nn.Linear(model.dim,truth.shape[1])
        optimizer=torch.optim.Adam(probe.parameters(),lr=.04)
        for _ in range(PROBE_STEPS):
            loss=F.binary_cross_entropy_with_logits(probe(states),truth)
            optimizer.zero_grad();loss.backward();optimizer.step()
        probe.eval()
        fits[name]=(probe,round(float(loss.detach()),9))
    return fits,len(states)


def auc(scores, labels):
    return base.auc(scores,labels)


def score_directions(model,probes,a,b,diagnostic):
    n=len(a);contrast=a-b;norm=contrast.norm(dim=1).clamp_min(1e-9)
    energies={k:[] for k in probes}
    with torch.no_grad():
        for states in (a,b):
            current=states.clone();tangent=contrast.clone()
            counts={k:torch.zeros(n) for k in probes}
            for t in range(DIAGNOSTIC_STEPS+1):
                for key,(probe,_) in probes.items():
                    logits=F.linear(tangent,probe.weight)
                    counts[key]+=logits.square().sum(dim=1)
                if t<DIAGNOSTIC_STEPS:
                    query=F.one_hot(diagnostic[:,t],V).float()
                    tangent=model.step_jvp(current,tangent,query)
                    current=model.step(current,query)
            for key in probes:energies[key].append(counts[key])
        result={}
        for key,(probe,_) in probes.items():
            score=torch.sqrt((energies[key][0]+energies[key][1]).clamp_min(0)/(2*probe.out_features))
            result[key]=score.cpu().numpy()
            result[key+'_norm']=(score/norm).cpu().numpy()
        result['distance']=norm.cpu().numpy()
    return result


@torch.no_grad()
def paired_multifuture_outcome(model,a,b,seed,depth):
    n=len(a)
    gen=torch.Generator().manual_seed(10000000+seed*100+depth)
    futures=torch.randint(V,(FUTURE_COUNT,n,OUTCOME_STEPS),generator=gen)
    result=torch.zeros(n,dtype=torch.bool)
    counts={k:0 for k in CHECKPOINTS}
    for k in range(FUTURE_COUNT):
        h1,h2=a.clone(),b.clone()
        for step in range(1,OUTCOME_STEPS+1):
            x=F.one_hot(futures[k,:,step-1],V).float()
            h1=model.step(h1,x);h2=model.step(h2,x)
            if step in CHECKPOINTS:
                different=((base.counterfactual(model,h1,1)>=0)!=
                           (base.counterfactual(model,h2,1)>=0)).any(dim=1)
                result|=different
                counts[step]+=int(different.sum())
    return result,counts


def evaluate(model,probes,seed,depth,per_goal=PER_GOAL):
    goals=torch.arange(GOALS).repeat_interleave(per_goal)
    generator=torch.Generator().manual_seed(6000000+seed*100+depth)
    goals=goals[torch.randperm(len(goals),generator=generator)]
    a=base.sample_prefixes(model,depth,goals,7000000+seed*100+depth)
    b=base.sample_prefixes(model,depth,goals,8000000+seed*100+depth)
    diag_gen=torch.Generator().manual_seed(9000000+seed*100+depth)
    diagnostic=torch.randint(V,(len(goals),DIAGNOSTIC_STEPS),generator=diag_gen)
    p1=probes['instant'][0]
    with torch.no_grad():
        native_a=base.counterfactual(model,a,1)>=0
        native_b=base.counterfactual(model,b,1)>=0
        p1a=p1(a)>=0;p1b=p1(b)>=0
        eligible=((native_a==native_b).all(dim=1)&(p1a==p1b).all(dim=1))
    # Score on complete pairs, independently of eligibility or future outcome.
    features=score_directions(model,probes,a,b,diagnostic)
    # Select eligible states only to avoid unnecessary 2*32-step rollout cost.
    idx=torch.where(eligible)[0]
    outcomes=torch.zeros(len(goals),dtype=torch.bool)
    checkpoints={j:0 for j in CHECKPOINTS}
    if len(idx):
        # Generate future schedules on ALL pairs, then select eligible indices.
        gen=torch.Generator().manual_seed(10000000+seed*100+depth)
        futures=torch.randint(V,(FUTURE_COUNT,len(goals),OUTCOME_STEPS),generator=gen)
        ha,hb=a[idx],b[idx]
        observed=torch.zeros(len(idx),dtype=torch.bool)
        for fidx in range(FUTURE_COUNT):
            x1,x2=ha.clone(),hb.clone()
            for t in range(1,OUTCOME_STEPS+1):
                tokens=F.one_hot(futures[fidx,idx,t-1],V).float()
                x1=model.step(x1,tokens)
                x2=model.step(x2,tokens)
                if t in CHECKPOINTS:
                    diff=((base.counterfactual(model,x1,1)>=0)!=
                          (base.counterfactual(model,x2,1)>=0)).any(dim=1)
                    observed|=diff;checkpoints[t]+=int(diff.sum())
        outcomes[idx]=observed
    ncheck=min(128,len(goals))
    with torch.no_grad():
        aa=a[:ncheck]
        signatures={
            'instant':(base.counterfactual(model,aa,1)>=0),
            'history':(base.counterfactual(model,aa,2)>=0),
        }
        signatures['duplicate']=signatures['instant'].repeat_interleave(V,dim=1)
        agreement={key:round(float(((probe(aa)>=0)==signatures[key]).float().mean()),6)
                   for key,(probe,_) in probes.items()}
    rows=[]
    for i in range(len(goals)):
        r={'seed':seed,'depth':depth,'pair':i,
           'goal_for_audit_only':int(goals[i]),'eligible':bool(eligible[i]),
           'future_any_8_16_32':int(outcomes[i]) if eligible[i] else None}
        r.update({key:float(value[i]) for key,value in features.items()})
        rows.append(r)
    return rows,{'eligible':int(eligible.sum()),'positives':int(outcomes[idx].sum()),
                  'per_checkpoint_difference_counts':checkpoints,
                  'probe_bit_agreement':agreement}


def summarize(rows):
    valid=[r for r in rows if r['eligible']]
    names=('instant','duplicate','history','instant_norm','duplicate_norm','history_norm','distance')
    ys=[r['future_any_8_16_32'] for r in valid]
    return {'total':len(rows),'eligible':len(valid),'positives':sum(ys),
            'auc':{name:None if (v:=auc([r[name] for r in valid],ys)) is None
                    else round(v,6) for name in names}}


def seed_map(cell):
    if cell not in SEEDS:raise ValueError('unknown cell '+cell)
    return SEEDS[cell]


def run_cell(cell,phase,steps=TRAIN_STEPS,per_goal=PER_GOAL):
    if phase not in ('calibration','heldout'):raise ValueError('bad phase')
    if per_goal<1 or per_goal>64:raise ValueError('invalid per-goal')
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    assignment=seed_map(cell)
    arch,family=assignment['arch'],assignment['family']
    seed=assignment[phase]
    model,loss=train_model(arch,family,seed,steps)
    probes,contexts=train_probes(model,seed)
    parity=base.test_jvp(model,seed)
    report={'experiment':'SHAFT-COMP-19','phase':phase,'cell':cell,'arch':arch,'family':family,
        'seed':seed,'registration_commit':REGISTRATION_COMMIT,
        'train_steps':steps,'per_goal':per_goal,'torch':torch.__version__,
        'numpy':np.__version__,
        'model':{'parameters':sum(p.numel() for p in model.parameters()),
                 'model_sha256':base.stable_sha(model),'train_loss':loss,
                 'accuracy_d12':test_task(model,family,seed,12),
                 'accuracy_d32':test_task(model,family,seed,32),'jvp_max_error':parity},
        'probes':{name:{'outputs':probe.out_features,'parameters':sum(p.numel() for p in probe.parameters()),
                         'loss':fitloss,'sha256':base.stable_sha(probe)}
                  for name,(probe,fitloss) in probes.items()},
        'probe_training_states':contexts,
        'outcome':'any native response divergence on 2 independently shared futures at steps 8,16,32',
        'diagnostic_steps':DIAGNOSTIC_STEPS,'rows':[], 'depth_checks':{}}
    for depth in STATE_DEPTHS:
        rows,audit=evaluate(model,probes,seed,depth,per_goal)
        report['rows'].extend(rows)
        report['depth_checks'][str(depth)]={**audit,'metrics':summarize(rows)}
    report['summary']=summarize(report['rows'])
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('calibration','heldout'),required=True)
    parser.add_argument('--cell',choices=tuple(SEEDS),required=True)
    parser.add_argument('--train-steps',type=int,default=TRAIN_STEPS)
    parser.add_argument('--per-goal',type=int,default=PER_GOAL)
    parser.add_argument('--threshold-file',type=Path)
    parser.add_argument('--threshold-sha256')
    parser.add_argument('--out',type=Path)
    a=parser.parse_args()
    if a.phase=='heldout':
        if a.train_steps!=TRAIN_STEPS or a.per_goal!=PER_GOAL:
            raise ValueError('heldout must retain registered full steps and pair counts')
        if not a.threshold_file or not a.threshold_sha256:
            raise ValueError('heldout requires frozen source thresholds and their SHA-256')
        data=a.threshold_file.read_bytes()
        if hashlib.sha256(data).hexdigest()!=a.threshold_sha256:
            raise ValueError('threshold receipt SHA-256 mismatch')
        frozen=json.loads(data)
        if frozen.get('registration_commit')!=REGISTRATION_COMMIT or frozen.get('phase')!='calibration_thresholds':
            raise ValueError('invalid calibration threshold receipt')
        if set(frozen.get('calibration_model_seeds',[]))!={1901,1902,1903,1904}:
            raise ValueError('unexpected calibration seeds')
    report=run_cell(a.cell,a.phase,a.train_steps,a.per_goal)
    out=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(out)
    else:print(out,end='')

if __name__=='__main__':main()
