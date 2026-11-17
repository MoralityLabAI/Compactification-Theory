"""OIC-16 learned dynamics CPU experiment with matched OIC-15 training samples."""
from __future__ import annotations
import argparse, json, math, platform, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from oic15_pilot import TRAIN_NS, OOD_NS, SEEDS, SPLITS, BATCH, STEPS, holdout, make_train_batch, oracle_batch, actions_for, balanced_panel, GenericGRU, CyclicFilter, RawFilter
from oic16_model import LearnedBelief

LR=0.035

def train_one(split:str,seed:int,steps:int=STEPS,augment:str='clean'):
    if split not in SPLITS or augment not in ('clean','raw_fault_aug'):
        raise ValueError('invalid split or augmentation')
    torch.manual_seed(160016+seed)
    rng=np.random.default_rng(151100+seed*19)  # exactly matching OIC-15 examples
    fault_rng=np.random.default_rng(161600+seed*211)
    model=LearnedBelief()
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    history=[]
    start=time.perf_counter()
    for it in range(steps):
        n=TRAIN_NS[it%len(TRAIN_NS)]
        fam,u,x=make_train_batch(n,BATCH,split,rng)
        if augment=='raw_fault_aug':
            x=x.clone()
            # independent perturbations of raw state samples, excluding neither
            # the initial nor the terminal readout, matched and family-blind.
            flip=torch.from_numpy(fault_rng.random(x.shape)<0.04)
            erase=torch.from_numpy(fault_rng.random(x.shape)<0.04)
            x[flip]^=1
            x[erase]=-1
        logits=model(n,u,x)
        loss=F.binary_cross_entropy_with_logits(logits,fam[:,None].float().expand_as(logits))
        # Discrete-transition preference is a declared regularizer, not a fixed law.
        kernel=model.kernel()
        regularizer=0.008*(1-(kernel*kernel).sum(-1)).sum()
        objective=loss+regularizer
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        opt.step()
        if it in (0,9,49,99,199,349,steps-1):
            history.append({'step':it+1,'loss':round(float(loss.item()),7),'regularizer':round(float(regularizer.item()),7)})
    wall=time.perf_counter()-start
    return model.eval(),{'history':history,'seconds':wall,'steps':steps,'batch':BATCH,
                         'parameters':sum(t.numel() for t in model.parameters()),'split':split,
                         'noise_augmentation':augment}

def panel(model,n,word,phase_marker=0,flip_pos=None,erase_pos=None):
    fam,phase,u,x=balanced_panel(n,word,marker=phase_marker)
    if flip_pos is not None or erase_pos is not None:
        x=x.clone()
        if flip_pos is not None:x[:,flip_pos]^=1
        if erase_pos is not None:x[:,erase_pos]=-1
    with torch.no_grad():
        logits,post=model(n,u,x,return_state=True)
        prob=torch.sigmoid(logits[:,-1]).numpy()
        posterior=post.numpy()
    labels=fam.numpy()
    return {
        'accuracy':float(np.mean((prob>0.5)==labels)),
        'brier':float(np.mean((prob-labels)**2)),
        'nll':float(np.mean(np.logaddexp(0,logits[:,-1].numpy())-labels*logits[:,-1].numpy())),
        'probabilities':[round(float(p),8) for p in prob],
        'mean_passive_deviation':float(np.mean(abs(prob-0.5))),
        'mean_joint_state_entropy':float(np.mean([-(np.maximum(q,1e-30)*np.log2(np.maximum(q,1e-30))).sum() for q in posterior.reshape(2*n,-1)])),
        'rotating_true_phase_mass':float(np.mean([posterior[n+j,1,(-j if model.kernel()[1,1] > model.kernel()[1,3] else j)%n,:].sum()/max(posterior[n+j,1,:,:].sum(),1e-30) for j in range(n)])),
    }

def evaluate(model,split,seed,augment):
    schedule_results={}
    for n in TRAIN_NS+OOD_NS:
        suite={}
        for name in ('passive','full_active','latch','extended_latch','alternating','random_budget'):
            suite[name]=panel(model,n,actions_for(n,name))
        # Unseen marker gauge: relabel latent phase and phase marker identically.
        word=actions_for(n,'full_active')
        offset=2 % n
        changed=panel(model,n,word,phase_marker=offset)
        standard=suite['full_active']
        # The model receives only observable histories; permutation of the
        # true latent labels cannot change the balanced panel distribution.
        gauge_max_prob_gap=max(abs(a-b) for f in (0,1) for a,b in zip(sorted(changed['probabilities'][f*n:(f+1)*n]),sorted(standard['probabilities'][f*n:(f+1)*n])))
        suite['relabel_control']={'accuracy':changed['accuracy'],'prob_set_gap':gauge_max_prob_gap}
        if n%2==0:
            word=(1,)*n+(0,)*2
            # Strict T=n+2 minimum one-flip-correcting raw-state horizon.
            suite['one_flip_by_position']=[panel(model,n,word,flip_pos=j)['accuracy'] for j in range(len(word)+1)]
            suite['one_erase_by_position']=[panel(model,n,word,erase_pos=j)['accuracy'] for j in range(len(word)+1)]
        schedule_results[str(n)]=suite
    return {'seed':seed,'split':split,'augment':augment,'panels':schedule_results,
            'kernel_facts':model.kernel_facts(8),'weights':{k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}}

def run(seeds,splits,augment,steps,out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter(); runs=[]
    for split in splits:
        for seed in seeds:
            model,train=train_one(split,seed,steps,augment)
            r=evaluate(model,split,seed,augment)
            r['training']=train
            runs.append(r)
            print(f"split={split} seed={seed} aug={augment} train_loss={train['history'][-1]['loss']}"
                  f"  n4/5/6/7/8 clean: "+str([round(r['panels'][str(n)]['full_active']['accuracy'],3) for n in TRAIN_NS+OOD_NS]),flush=True)
    meta={'name':'OIC-16-CPU-01','status':'executed_learned_circulant_belief','steps':steps,
          'batch':BATCH,'seeds':list(seeds),'splits':list(splits),
          'augmentation':augment,'lr':LR,'training_samples_matched_to_OIC15':augment=='clean',
          'train_n':list(TRAIN_NS),'ood_n':list(OOD_NS),'transition_kernel':'five_offset_row_stochastic_circulant',
          'emission':'learned_shared_Fourier_soft_phase_marker','model_receives_phase':False,
          'model_receives_family_at_inference':False,'known_n_input':True,
          'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()}
    result={'protocol':meta,'runs':runs,'wall_seconds':time.perf_counter()-start}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=1,sort_keys=True)+'\n')
    print('saved',out, 'wall',round(result['wall_seconds'],2),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    parser.add_argument('--splits',nargs='+',choices=SPLITS,default=list(SPLITS))
    parser.add_argument('--augment',choices=['clean','raw_fault_aug'],default='clean')
    parser.add_argument('--steps',type=int,default=STEPS)
    a=parser.parse_args(); run(tuple(a.seeds),tuple(a.splits),a.augment,a.steps,a.out)
if __name__=='__main__':main()
