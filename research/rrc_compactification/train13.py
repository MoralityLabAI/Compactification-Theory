"""RRC-13 train-only supervised program-construction CPU experiment.

Training teacher-forces canonical graph labels. After the warm-up, fixed
scheduled-sampling decisions select partial residuals from the train roster.
No test roster, withheld label, or query future is used in gradient updates.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from . import graph13 as b, policy13 as p
except ImportError:
    import graph13 as b
    import policy13 as p

HERE=Path(__file__).resolve().parent
STEPS=480
BATCH=128
LR=0.003
SEED=b.SEED


def hashes():
    return {n:sha256((HERE/n).read_bytes()).hexdigest()
            for n in ('graph11.py','graph12.py','graph13.py','policy13.py','train13.py')}


def create_model():
    import torch
    from torch import nn
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Linear(104,p.HIDDEN)
            self.cell=nn.GRUCell(p.INPUT,p.HIDDEN)
            self.head=nn.Linear(p.HIDDEN,4)
        def forward(self,raw,labels,fb1,fb2,*,use_feedback,scheduled):
            import torch.nn.functional as F
            h=torch.tanh(self.encoder(raw))
            prev=torch.zeros((raw.shape[0],4),dtype=raw.dtype,device=raw.device)
            fb=torch.zeros((raw.shape[0],b.FEEDBACK_SIZE),dtype=raw.dtype,device=raw.device)
            logits=[]
            selected=[]
            batch_index=torch.arange(raw.shape[0],device=raw.device)
            for stage in range(3):
                step=F.one_hot(torch.full((raw.shape[0],),stage,device=raw.device),3).to(raw.dtype)
                h=self.cell(torch.cat([prev,fb,step],dim=1),h)
                logit=self.head(h);logits.append(logit)
                # Train with model-sampled prior decisions on some minibatch rows.
                # Sampling mask is supplied by the caller and source-independent.
                choice=torch.where(scheduled[:,stage],logit.detach().argmax(1),labels[:,stage])
                selected.append(choice)
                prev=F.one_hot(choice,4).to(raw.dtype)
                if use_feedback and stage==0:fb=fb1[batch_index,choice]
                elif use_feedback and stage==1:fb=fb2[batch_index,selected[0],choice]
                else:fb=torch.zeros_like(fb)
            return torch.stack(logits,dim=1)
    return Model()


def arrays(cases,*,with_feedback=True):
    """Observed source/target demo prefixes; public partial-graph simulations.

    The training labels select the intermediate path for teacher forcing.
    Precomputed counterfactual feedback prevents repeated algebraic work
    during SGD; its 4+16 candidate executions per training example are a
    *training-time* resource, never a test-time policy permission.
    """
    import torch
    raw=torch.tensor([b.raw_features(c) for c in cases],dtype=torch.float32)
    labels=torch.tensor([[b.OPS.index(v) for v in c.hidden_graph] for c in cases],dtype=torch.long)
    if with_feedback:
        first=torch.tensor([[b.residual_feedback(c,a) for a in b.OPS] for c in cases],dtype=torch.float32)
        second=torch.tensor([[[b.residual_feedback(c,a,v) for v in b.OPS] for a in b.OPS]
                             for c in cases],dtype=torch.float32)
    else:
        first=torch.zeros((len(cases),4,b.FEEDBACK_SIZE),dtype=torch.float32)
        second=torch.zeros((len(cases),4,4,b.FEEDBACK_SIZE),dtype=torch.float32)
    return raw,labels,first,second


def fit(*,steps=STEPS,train_n=b.TRAIN_N,dev_n=b.DEV_N,verbose=True):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Test generation must remain unreachable during fitting.
    roster=b.dataset(include_test=False,train_count=train_n,dev_count=dev_n)
    train=roster['train'];dev=roster['dev']
    tx,ty,ta,tb=arrays(train)
    dx,dy,da,db=arrays(dev)
    rng=random.Random(SEED+151)
    batches=[tuple(rng.randrange(len(train)) for _ in range(BATCH)) for _ in range(steps)]
    # Fixed scheduled-sampling mask, shared between models.
    scheduler=random.Random(SEED+242)
    masks=[[tuple(k<2 and t>steps//3 and scheduler.random()<.35 for k in range(3))
            for _ in range(BATCH)] for t in range(steps)]
    checkpoints={}
    for offset,kind in enumerate(('active','passive')):
        torch.manual_seed(SEED+1000+offset)
        net=create_model()
        opt=torch.optim.Adam(net.parameters(),lr=LR)
        for t,indices in enumerate(batches):
            j=torch.tensor(indices,dtype=torch.long)
            scheduled=torch.tensor(masks[t],dtype=torch.bool)
            preds=net(tx[j],ty[j],ta[j],tb[j],use_feedback=kind=='active',scheduled=scheduled)
            loss=F.cross_entropy(preds.reshape(-1,4),ty[j].reshape(-1))
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            dev_logits=net(dx,dy,da,db,use_feedback=kind=='active',
                           scheduled=torch.zeros((len(dev),3),dtype=torch.bool))
            dev_loss=float(F.cross_entropy(dev_logits.reshape(-1,4),dy.reshape(-1)).item())
            token_acc=float((dev_logits.argmax(-1)==dy).float().mean().item())
        result={'schema':'rrc13.weights.v1','kind':kind,'seed':SEED,'steps':steps,
                'batch_size':BATCH,'learning_rate':LR,'train_count':len(train),'dev_count':len(dev),
                'parameter_count':sum(q.numel() for q in net.parameters()),'hidden_size':p.HIDDEN,
                'feedback_dimensions':b.FEEDBACK_SIZE,'partial_calls_train_per_case':20 if kind=='active' else 0,
                'training_teacher_labels':True,'scheduled_sampling_rate_after_warmup':0.35,
                'new_composition_classes_used_for_training':False,'train_digest':b.digest(train),
                'dev_digest':b.digest(dev),'source_hashes':hashes(),
                'torch_version':torch.__version__,
                'development_teacher_forced_loss':dev_loss,
                'development_teacher_forced_token_accuracy':token_acc,
                'state_dict':{name:value.detach().tolist() for name,value in net.state_dict().items()}}
        checkpoints[kind]=result
        if verbose:print(kind,'parameters',result['parameter_count'],
                         'dev loss',round(dev_loss,4),'teacher-forced token acc',round(token_acc,4),flush=True)
    return checkpoints


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=HERE/'results')
    parser.add_argument('--steps',type=int,default=STEPS)
    a=parser.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    for k,checkpoint in fit(steps=a.steps).items():
        (a.out_dir/f'rrc13_{k}.json').write_text(json.dumps(checkpoint,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
