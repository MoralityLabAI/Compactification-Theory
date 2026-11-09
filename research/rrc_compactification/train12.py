"""RRC-12 deterministic CPU-only neural training, no held-out roster access."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from . import graph12,model12
except ImportError:
    import graph12,model12

HERE=Path(__file__).resolve().parent
SEED=2026111012
STEPS=420
BATCH=128
LR=0.004


def input_tensor(cases):
    import torch
    return torch.tensor([graph12.features(c) for c in cases],dtype=torch.float32)


def labels(cases):
    import torch
    return torch.tensor([[graph12.OPS.index(s) for s in c.hidden_graph] for c in cases],dtype=torch.long)


def create_model(kind):
    import torch
    from torch import nn
    if kind in ('recurrent','shuffled_recurrent'):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn=nn.GRU(4,32,batch_first=True)
                self.head=nn.Linear(32,12)
            def forward(self,x):
                return self.head(self.rnn(x)[1][-1])
        return Net()
    if kind=='feedforward':
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1=nn.Linear(104,34)
                self.head=nn.Linear(34,12)
            def forward(self,x):
                return self.head(torch.tanh(self.fc1(x.reshape(x.shape[0],-1))))
        return Net()
    raise ValueError(kind)


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in ('graph12.py','model12.py','train12.py','graph11.py')}


def fit(*,steps=STEPS):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    splits=graph12.dataset(include_test=False)
    x=input_tensor(splits['train']);y=labels(splits['train'])
    devx=input_tensor(splits['dev']);devy=labels(splits['dev'])
    generator=random.Random(SEED+151)
    indices=[tuple(generator.randrange(len(y)) for _ in range(BATCH)) for _ in range(steps)]
    label_rng=random.Random(SEED+951)
    shuffled=list(range(len(y)));label_rng.shuffle(shuffled)
    results={}
    for offset,kind in enumerate(('recurrent','feedforward','shuffled_recurrent')):
        torch.manual_seed(SEED+200+offset)
        net=create_model(kind)
        opt=torch.optim.Adam(net.parameters(),lr=LR)
        target=y[shuffled] if kind=='shuffled_recurrent' else y
        for step,batch in enumerate(indices):
            j=torch.tensor(batch,dtype=torch.long)
            pred=net(x[j]).reshape(-1,4)
            loss=F.cross_entropy(pred,target[j].reshape(-1))
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            dev_logits=net(devx).reshape(-1,4)
            dev_loss=float(F.cross_entropy(dev_logits,devy.reshape(-1)).item())
            dev_token_acc=float((dev_logits.argmax(-1)==devy.reshape(-1)).float().mean().item())
        state={name:val.detach().tolist() for name,val in net.state_dict().items()}
        result={'schema':'rrc12.weights.v1','kind':kind,'seed':SEED,'steps':steps,
                'batch_size':BATCH,'learning_rate':LR,'parameter_count':sum(p.numel() for p in net.parameters()),
                'hidden_size':32 if kind!='feedforward' else 34,
                'input_dimensions':[26,4], 'output_dimensions':[3,4],
                'train_cases':len(splits['train']),'dev_cases':len(splits['dev']),
                'train_roster_sha256':graph12.roster_digest(splits['train']),
                'dev_roster_sha256':graph12.roster_digest(splits['dev']),
                'source_sha256':source_hashes(),'torch_version':torch.__version__,
                'development_token_accuracy':dev_token_acc,'development_loss':dev_loss,
                'source_dsl_test_generation_during_training':False,
                'public_graph_interpreter_during_inference':False,
                'state_dict':state}
        results[kind]=result
        print(kind,'params',result['parameter_count'],'dev loss',round(dev_loss,4),
              'dev token acc',round(dev_token_acc,4),flush=True)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,default=HERE/'results');p.add_argument('--steps',type=int,default=STEPS)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    for k,v in fit(steps=a.steps).items():
        (a.out_dir/f'rrc12_{k}.json').write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
