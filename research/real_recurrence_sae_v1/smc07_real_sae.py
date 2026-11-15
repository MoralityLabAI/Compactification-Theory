"""SMC-07: source-locked recurrent Sudoku SAE causal audit.

The model, source and heldout roster come from LSV-05. The SAE trains
unsupervised on post-first-recurrence full-token states from TRAIN puzzles.
Validation controls model selection; test puzzles only appear in final metrics.
This script never assumes that Sudoku activations contain K6 matchings.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
TRM_DIR = ROOT / 'levin_sieve_v1' / 'pretrained_trm'
EXPECTED_ROSTER = 'eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611'


class TopKSAE(nn.Module):
    def __init__(self, d_input=48, width=96, k=4):
        super().__init__()
        assert 0 < k <= width
        self.encoder = nn.Linear(d_input, width)
        self.decoder = nn.Linear(width, d_input)
        self.k = k

    def encode(self, x):
        activations = F.relu(self.encoder(x))
        top = torch.topk(activations, self.k, dim=-1)
        return torch.zeros_like(activations).scatter(-1, top.indices, top.values)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


def split_ids(n):
    """Source roster order, grouped by puzzle ID; no token-level leakage."""
    if n < 8 or n % 4:
        raise ValueError('roster must contain >=8 puzzles and be divisible by 4')
    groups = {'train':[], 'val':[], 'test':[]}
    for i in range(n):
        groups['train' if i%4 in (0,1) else 'val' if i%4==2 else 'test'].append(i)
    assert len(set(groups['train']) & set(groups['test'])) == 0
    assert len(groups['train'])+len(groups['val'])+len(groups['test']) == n
    return groups


def train_norm(train):
    x = train.reshape(-1,train.shape[-1]).detach()
    mu = x.mean(dim=0)
    sig = x.std(dim=0,unbiased=False).clamp_min(1e-4)
    return mu,sig


def tensor_hash(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def fit_sae(train,val,seed=71,steps=300,width=96,k=4,lr=0.003,batch_size=256):
    torch.manual_seed(seed)
    mean,std = train_norm(train)
    x=((train.reshape(-1,train.shape[-1])-mean)/std).detach()
    y=((val.reshape(-1,val.shape[-1])-mean)/std).detach()
    sae=TopKSAE(d_input=x.shape[-1],width=width,k=k)
    initial=copy.deepcopy(sae.state_dict())
    opt=torch.optim.AdamW(sae.parameters(),lr=lr,weight_decay=1e-5)
    best=float('inf');best_state=None;best_step=0
    history=[]
    for t in range(1,steps+1):
        idx=torch.randint(len(x),(min(batch_size,len(x)),))
        xx=x[idx]
        pred,z=sae(xx)
        loss=F.mse_loss(pred,xx)+1e-4*z.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t==1 or t%25==0 or t==steps:
            with torch.no_grad():
                val_error=F.mse_loss(sae(y)[0],y).item()
                train_error=F.mse_loss(sae(x)[0],x).item()
            history.append({'step':t,'train_mse':train_error,'val_mse':val_error})
            if val_error < best:
                best=val_error; best_step=t;best_state=copy.deepcopy(sae.state_dict())
    sae.load_state_dict(best_state)
    sae.eval()
    return sae,mean,std,{'seed':seed,'steps':steps,'best_step':best_step,
                          'best_val_mse':best,'history':history,
                          'initial_decoder_sha256':tensor_hash(initial['decoder.weight'])}


def pca_basis(train,mean,std,d=4):
    x=(train.reshape(-1,train.shape[-1])-mean)/std
    cov=x.T@x/len(x)
    eigenvalues,eigenvectors=torch.linalg.eigh(cov)
    return eigenvectors[:,-d:],eigenvalues[-d:]


def reconstruction(sae,h,mu,std,basis=None):
    x=(h-mu)/std
    if basis is None:
        y=sae(x)[0]
    else:
        y=(x@basis)@basis.T
    return y*std+mu


def sparsity_and_error(sae,state,mu,std,baselines):
    with torch.no_grad():
        x=(state.reshape(-1,state.shape[-1])-mu)/std
        reconstructed,z=sae(x)
        mse=float(F.mse_loss(reconstructed,x))
        nnz=float((z>0).sum(-1).float().mean())
        err={'sae_mse_normalized':mse,'sae_positive_l0':nnz}
        for name,b in baselines.items():
            prediction=(x@b)@b.T
            err[name+'_mse_normalized']=float(F.mse_loss(prediction,x))
        return err


def continue_teacher(model,state,remaining=3):
    z=state
    for _ in range(remaining):
        z=model.block(z)
    return model.head(z)


def make_directions(sae, h, mu, std, gen):
    x=((h-mu)/std).reshape(1,-1)
    with torch.no_grad():
        activation=sae.encode(x)[0]
        top=int(activation.argmax())
        chosen=sae.decoder.weight[:,top].detach()*std
        inactive=torch.where(activation==0)[0]
        if len(inactive):
            j=int(inactive[torch.randint(len(inactive),(1,),generator=gen)])
        else:
            j=(top+1)%sae.decoder.weight.shape[1]
        unchosen=sae.decoder.weight[:,j].detach()*std
        normal=torch.randn(h.shape, generator=gen, dtype=h.dtype)
        return {'sae_active':chosen,'sae_inactive':unchosen,'isotropic':normal},top


def response_to_intervention(model,clean_stage1,goal,delta,remote=False):
    z=clean_stage1.clone()
    slot=(goal+1)%z.shape[1] if remote else goal
    z[0,slot,:] += delta
    return continue_teacher(model,z,remaining=3)[0,goal,:]


def gauge_control(model,h1,goal,amp=0.25):
    """Stage-two pre-LN2 shift and a matched centered perturbation in FP64."""
    m=copy.deepcopy(model).to(dtype=torch.float64).eval()
    h=h1.double()
    with torch.no_grad():
        t=m.block.ln1(h+m.block.tok_mix(h.transpose(1,2)).transpose(1,2))
        bus=t+m.block.chan_mix(t)
        shift=torch.zeros_like(bus)
        shift[0,goal,:]=amp
        centered=torch.linspace(-1,1,bus.shape[-1],dtype=bus.dtype)
        centered-=centered.mean()
        centered=centered/centered.norm()*shift[0,goal].norm()
        ctrl=torch.zeros_like(bus)
        ctrl[0,goal,:]=centered
        def finish(z):
            for _ in range(2): z=m.block(z)
            return m.head(z)[0,goal]
        base=finish(m.block.ln2(bus))
        gauge=finish(m.block.ln2(bus+shift))
        other=finish(m.block.ln2(bus+ctrl))
        return {'gauge_logit_linf':float((base-gauge).abs().max()),
                'transverse_logit_linf':float((base-other).abs().max()),
                'amp':amp,'site':'stage-2 pre-LayerNorm2 input'}


def goal_replay_summary(records):
    n=len(records)
    results={}
    for name in ('sae_reconstruction','pca4_reconstruction'):
        vals=[p['reconstructions'][name] for p in records]
        results[name]={'mean_logit_mse':sum(v['logit_mse'] for v in vals)/n,
                       'mean_hidden_delta_norm':sum(v['norm_hidden_delta'] for v in vals)/n,
                       'teacher_agreement':sum(v['teacher_argmax_agrees'] for v in vals)/n,
                       'target_correct_rate':sum(v['target_correct'] for v in vals)/n}
    for rad in ('0.2','0.8'):
        for name in ('sae_active','sae_inactive','isotropic','same_direction_remote_token'):
            v=[p['interventions'][rad][name] for p in records if 'abstain' not in p['interventions'][rad][name]]
            if not v: continue
            results[rad+'_'+name]={'mean_goal_logit_delta_norm':sum(x['goal_logit_delta_norm'] for x in v)/len(v),
                                    'teacher_flip_count':sum(x['teacher_label_flip'] for x in v),
                                    'n':len(v)}
            if name!='same_direction_remote_token':
                results[rad+'_'+name]['mean_margin_effect_abs']=sum(x['margin_effect_abs'] for x in v)/len(v)
    return results


def goal_replay(model,states,inputs,targets,split,sae,mu,std,pca,seed=71,radii=(0.2,0.8)):
    gen=torch.Generator().manual_seed(101031+seed)
    test=list(split['test'])
    records=[]
    with torch.no_grad():
        for i in test:
            goal=int(torch.where(inputs[i]==0)[0][0])
            base=states[1][i:i+1].detach()
            ref=model.head(states[4][i:i+1])[0,goal].detach()
            h=base[0,goal]
            choices={
                'sae_reconstruction':reconstruction(sae,h,mu,std),
                'pca4_reconstruction':reconstruction(sae,h,mu,std,pca),
            }
            alternatives={}
            for name,hnew in choices.items():
                changed=base.clone()
                changed[0,goal]=hnew
                predicted=continue_teacher(model,changed)[0,goal]
                alternatives[name]={
                    'norm_hidden_delta':float((hnew-h).norm()),
                    'logit_mse':float(F.mse_loss(predicted,ref)),
                    'teacher_argmax_agrees':int(predicted.argmax()==ref.argmax()),
                    'target_correct':int(predicted.argmax()==targets[i,goal]),
                }
            dirs,top_feature=make_directions(sae,h,mu,std,gen)
            direction_records={}
            for r in radii:
                dres={}
                for label,v in dirs.items():
                    norm=float(v.norm())
                    if norm<1e-10:
                        dres[label]={'abstain':'zero decoder direction'}
                        continue
                    delta=-r*v/v.norm()
                    predicted=response_to_intervention(model,base,goal,delta)
                    top2=ref.topk(2).indices
                    before=ref[top2[0]]-ref[top2[1]]
                    after=predicted[top2[0]]-predicted[top2[1]]
                    dres[label]={
                        'margin_effect_abs':float((after-before).abs()),
                        'goal_logit_delta_norm':float((predicted-ref).norm()),
                        'teacher_label_flip':int(predicted.argmax()!=ref.argmax()),
                    }
                    if label=='sae_active':
                        rem=response_to_intervention(model,base,goal,delta,remote=True)
                        dres['same_direction_remote_token']={
                            'goal_logit_delta_norm':float((rem-ref).norm()),
                            'teacher_label_flip':int(rem.argmax()!=ref.argmax()),
                        }
                direction_records[str(r)]=dres
            records.append({'puzzle_id':i,'goal_token':goal,
                            'clean_teacher_label':int(ref.argmax()),
                            'ground_truth_target':int(targets[i,goal]),
                            'top_active_sae_feature':top_feature,
                            'reconstructions':alternatives,
                            'interventions':direction_records})
    with torch.no_grad():
        g=goal_replay_summary(records)
        gauge=[gauge_control(model,states[1][i:i+1],int(torch.where(inputs[i]==0)[0][0])) for i in test[:2]]
    return {'n_independent_test_puzzles':len(test),'per_puzzle':records,
            'aggregates':g,'gauge_checks_fp64':gauge}


def feature_transport(sae,states,split,mu,std):
    with torch.no_grad():
        test=list(split['test'])
        a=sae.encode((states[1][test].reshape(-1,48)-mu)/std)
        b=sae.encode((states[2][test].reshape(-1,48)-mu)/std)
        i=a.topk(sae.k,dim=-1).indices
        j=b.topk(sae.k,dim=-1).indices
        overlap=((i[:,:,None]==j[:,None,:]).any(-1).float().sum(-1)/sae.k).mean()
        s2=sae.encode((states[2][test]-mu)/std)
        bnull=s2.roll(shifts=1,dims=0).reshape(-1,sae.encoder.out_features)
        jnull=bnull.topk(sae.k,dim=-1).indices
        null=((i[:,:,None]==jnull[:,None,:]).any(-1).float().sum(-1)/sae.k).mean()
        return {'depth1_to_2_mean_topk_overlap':float(overlap),
                'puzzle_shuffled_null_overlap':float(null),
                'n_token_pairs':int(i.shape[0]),'k':sae.k}


def run(checkpoint=None,allow_fetch=False,max_cases=64,seeds=(71,89,113),steps=300):
    sys.path.insert(0,str(TRM_DIR))
    from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
    lock=json.loads((TRM_DIR/'source_lock.json').read_text())
    data,digest=locked_bytes(lock,checkpoint_path=checkpoint,allow_fetch=allow_fetch)
    model=load_pretrained(lock,data)
    torch.set_num_threads(2)
    roster,roster_hash=prepare_roster(lock)
    if roster_hash!=EXPECTED_ROSTER:
        raise ValueError('source-locked roster mismatch')
    inputs,targets,holes=[x[:max_cases] for x in roster]
    split=split_ids(max_cases)
    with torch.no_grad():
        states=[h.detach() for h in model.reason_states(inputs)]
    results=[]
    for seed in seeds:
        train=states[1][split['train']]
        val=states[1][split['val']]
        sae,mu,std,fit=fit_sae(train,val,seed=seed,steps=steps)
        pc,evals=pca_basis(train,mu,std)
        rgen=torch.Generator().manual_seed(9817)
        qr,_=torch.linalg.qr(torch.randn(48,4,generator=rgen))
        bases={'pca4':pc,'random4':qr}
        reconstruction_by_depth={str(d):sparsity_and_error(sae,states[d][split['test']],mu,std,bases)
                                 for d in (1,2,3,4)}
        results.append({'seed':seed,'train':fit,'reconstruction_by_depth':reconstruction_by_depth,
            'feature_transport':feature_transport(sae,states,split,mu,std),
            'causal_goal_replay':goal_replay(model,states,inputs,targets,split,sae,mu,std,pc,seed)})
    return {'schema':'SMC-07/source-locked-real-recurrence-sae-v1',
            'checkpoint_sha256':digest,'source_commit':lock['revision'],
            'source_model':'external third-party TRM-inspired, 44,001 parameters',
            'roster_sha256':roster_hash,'n_puzzles':max_cases,'split_ids':split,
            'architecture':'top4 ReLU SAE 48->96->48',
            'sae_training_only_on_depth1_train_tokens':True,'train_seed_list':list(seeds),
            'steps':steps,'training_objective':'standardized reconstruction MSE + 1e-4 sparse activation L1',
            'torch_version':torch.__version__,'results':results,
            'scope':'source-locked SAE causal audit on actual pretrained recurrence; an interpretability advantage must be established from the metrics'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--max-cases',type=int,choices=(16,64),default=64)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--seeds',type=str,default='71,89,113')
    p.add_argument('--out',type=Path,default=Path('runs/smc07/full.json'))
    a=p.parse_args()
    seeds=tuple(map(int,a.seeds.split(',')))
    result=run(a.checkpoint,a.fetch,a.max_cases,seeds,a.steps)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    summary=[{'seed':item['seed'],
              'test_depth1':item['reconstruction_by_depth']['1'],
              'transport':item['feature_transport'],
              'causal_0.2_sae':item['causal_goal_replay']['aggregates']['0.2_sae_active'],
              'causal_0.2_random':item['causal_goal_replay']['aggregates']['0.2_isotropic'],
              'gauge':item['causal_goal_replay']['gauge_checks_fp64']}
             for item in result['results']]
    print(json.dumps({'receipt':str(a.out),'source_commit':result['source_commit'],
                      'checkpoint_sha256':result['checkpoint_sha256'],
                      'roster_sha256':result['roster_sha256'],
                      'puzzles':result['n_puzzles'],'summary':summary},indent=2))

if __name__=='__main__':main()
