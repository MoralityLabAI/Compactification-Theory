"""RRC-19 deterministic, stratified CPU training of graph quotient classifiers.

All training signals come from signed, product or sparse source families. The
held-out rational and squared sources are unreachable when include_test=False.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from . import family19 as f, learn18 as l, model18 as m, train18
except ImportError:
    import family19 as f
    import learn18 as l
    import model18 as m
    import train18

HERE=Path(__file__).resolve().parent
STEPS=700
BATCH=128
LR=0.003
SEED=f.SEED
ROLES=('balanced_recurrent','skewed_recurrent','balanced_feedforward','shuffled_recurrent')
SOURCES=('family19.py','train19.py','learn18.py','model18.py','train18.py',
         'certify18.py','topology16.py','graph11.py','graph12.py',
         'graph13.py','graph14.py','quotient17.py')


def source_hashes():
    return {n:sha256((HERE/n).read_bytes()).hexdigest() for n in SOURCES}


def fit(*,steps=STEPS,train_per=f.TRAIN_PER_CELL,dev_per=f.DEV_PER_CELL,verbose=True):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    groups=f.roster(include_test=False,train_per=train_per,dev_per=dev_per)
    cases=groups['train'];dev=groups['dev']
    classes=l.class_map()
    ids={cell:[] for cell in ((fam,top) for fam in f.TRAIN_FAMILIES for top in l.TOPOLOGIES)}
    for i,c in enumerate(cases):ids[(f.source_family(c),c.topology)].append(i)
    if any(len(v)!=train_per for v in ids.values()):raise AssertionError('unbalanced training cells')
    labels=torch.tensor(l.labels(cases,classes),dtype=torch.long)
    dev_labels=torch.tensor(l.labels(dev,classes),dtype=torch.long)
    seq=torch.tensor([l.sequence(c) for c in cases],dtype=torch.float32)
    flat=torch.tensor([l.flat(c) for c in cases],dtype=torch.float32)
    dseq=torch.tensor([l.sequence(c) for c in dev],dtype=torch.float32)
    dflat=torch.tensor([l.flat(c) for c in dev],dtype=torch.float32)
    allowed={top:{g for g in classes[top]['representatives']} for top in l.TOPOLOGIES}
    mask=torch.tensor([[g in allowed[c.topology] for g in l.g.GRAPHS] for c in cases],dtype=torch.bool)
    dmask=torch.tensor([[g in allowed[c.topology] for g in l.g.GRAPHS] for c in dev],dtype=torch.bool)
    if not bool(mask[torch.arange(len(cases)),labels].all()):raise AssertionError('bad class masks')
    cells=tuple(ids)
    sampling={}
    for mode in ('balanced','skewed'):
        rng=random.Random(SEED+520+int(mode=='skewed'))
        family_weights={'signed':.8,'product':.15,'sparse':.05}
        batches=[];frequency={fam:0 for fam in f.TRAIN_FAMILIES}
        for _ in range(steps):
            row=[]
            for _ in range(BATCH):
                if mode=='balanced':cell=cells[rng.randrange(len(cells))]
                else:
                    u=rng.random()
                    fam='signed' if u<family_weights['signed'] else 'product' if u<family_weights['signed']+family_weights['product'] else 'sparse'
                    cell=(fam,l.TOPOLOGIES[rng.randrange(len(l.TOPOLOGIES))])
                frequency[cell[0]]+=1
                row.append(ids[cell][rng.randrange(len(ids[cell]))])
            batches.append(torch.tensor(row,dtype=torch.long))
        sampling[mode]=(batches,frequency)
    shuffled=labels.clone()
    rng=random.Random(SEED+710)
    for cell,indices in ids.items():
        targets=[int(labels[i]) for i in indices]
        rng.shuffle(targets)
        for i,z in zip(indices,targets):shuffled[i]=z
    constructors=train18.modules()
    checkpoints={}
    for role in ROLES:
        kind='feedforward' if role=='balanced_feedforward' else 'recurrent'
        mode='skewed' if role=='skewed_recurrent' else 'balanced'
        torch.manual_seed(SEED+3100+(1 if kind=='feedforward' else 0))
        net=constructors[kind]()
        optimizer=torch.optim.Adam(net.parameters(),lr=LR)
        x=flat if kind=='feedforward' else seq
        xd=dflat if kind=='feedforward' else dseq
        target=shuffled if role=='shuffled_recurrent' else labels
        for ix in sampling[mode][0]:
            logits=net(x[ix]).masked_fill(~mask[ix],-1e4)
            loss=F.cross_entropy(logits,target[ix])
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            z=net(xd).masked_fill(~dmask,-1e4)
            loss=float(F.cross_entropy(z,dev_labels))
            top1=float((z.argmax(1)==dev_labels).float().mean())
            top12=float((z.topk(12,dim=1).indices==dev_labels[:,None]).any(1).float().mean())
            family_dev={}
            for fam in f.TRAIN_FAMILIES:
                select=torch.tensor([f.source_family(c)==fam for c in dev],dtype=torch.bool)
                family_dev[fam]={
                    'cases':int(select.sum()),
                    'canonical_top12':float((z[select].topk(12,dim=1).indices==dev_labels[select,None]).any(1).float().mean()),
                }
        count=sum(p.numel() for p in net.parameters())
        if count!=m.parameter_count(kind):raise AssertionError('parameter-budget change')
        checkpoints[role]={
            'schema':'rrc19.weights.v1','role':role,'kind':kind,
            'seed':SEED,'optimizer_steps':steps,'batch_size':BATCH,'learning_rate':LR,
            'parameter_count':count,'train_per_cell':train_per,'dev_per_cell':dev_per,
            'training_families':list(f.TRAIN_FAMILIES),'unseen_families':list(f.OOD_FAMILIES),
            'sampling_mode':mode,'sample_counts':sampling[mode][1],
            'source_sha256':source_hashes(),
            'train_roster_sha256':f.digest(cases),'dev_roster_sha256':f.digest(dev),
            'hidden_graph_labels_training_only':True,'future_access_at_inference':False,
            'neural_candidate_previews':0,
            'label_shuffled_within_topology_and_family':role=='shuffled_recurrent',
            'torch_version':torch.__version__,'dev_cross_entropy':loss,
            'dev_canonical_top1':top1,'dev_canonical_top12':top12,'dev_by_family':family_dev,
            'state_dict':{k:v.detach().tolist() for k,v in net.state_dict().items()},
        }
        if verbose:print(role,'parameter count',count,'dev top12',round(top12,3),
                         'source sampling',sampling[mode][1],flush=True)
    return checkpoints


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    p.add_argument('--steps',type=int,default=STEPS)
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    for role,ck in fit(steps=args.steps).items():
        path=args.out_dir/f'rrc19_{role}.json'
        path.write_text(json.dumps(ck,indent=2,sort_keys=True)+'\n')
        print('wrote',path)
if __name__=='__main__':main()
