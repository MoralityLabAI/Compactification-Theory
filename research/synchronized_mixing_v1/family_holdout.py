"""SMC-03b: hold out entire matching identities from synthetic CPU training.

Class-conditional graph and permutation marginals are exactly matched
within each split. The input architecture/training budget matches SMC-03.
"""
import json
from fractions import Fraction
from relational_probe import orbit_split,pair_features,empirical_metrics

def family_split(k=3,heldout=3):
    if k!=3 or not (1<=heldout<15):
        raise ValueError('family benchmark fixes six vertices and 1<=heldout<15')
    s=orbit_split(k)
    m=len(s['states'])
    train_g=tuple(range(m-heldout))
    test_g=tuple(range(m-heldout,m))
    train_pi=tuple(i for i,x in enumerate(s['orbit_index']) if x in train_g)
    test_pi=tuple(i for i,x in enumerate(s['orbit_index']) if x in test_g)
    assert len(train_pi)==48*len(train_g) and len(test_pi)==48*len(test_g)
    assert not(set(train_pi)&set(test_pi))
    return {'split':s,'train_g':train_g,'test_g':test_g,'train_pi':train_pi,'test_pi':test_pi}

def train_family_holdout(k=3,seed=20261115,steps=300,batch_size=256):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    cohort=family_split(k)
    s=cohort['split']
    train_g=cohort['train_g']
    test_g=cohort['test_g']
    train_pi=cohort['train_pi']
    test_pi=cohort['test_pi']
    graph,perm,overlap=pair_features(s)
    graph=torch.tensor(graph,dtype=torch.float32)
    perm=torch.tensor(perm,dtype=torch.float32)
    overlap=torch.tensor(overlap,dtype=torch.float32)
    idx=torch.tensor(s['orbit_index'],dtype=torch.long)
    def data(pi_idx,gs):
        p=torch.tensor(pi_idx,dtype=torch.long)
        allowed=torch.tensor(gs,dtype=torch.long)
        positive_g=idx[p]
        xp=torch.cat((graph[positive_g],perm[p]),dim=-1)
        xn=torch.cat((graph[allowed][None,:,:].expand(len(p),-1,-1),
                      perm[p,None,:].expand(-1,len(allowed),-1)),dim=-1).reshape(-1,graph.shape[1]+perm.shape[1])
        sp=overlap[p,positive_g][:,None]
        sn=overlap[p,:][:,allowed].reshape(-1,1)
        return xp,xn,sp,sn
    rp,rn,sp,sn=data(train_pi,train_g)
    tp,tn,tsp,tsn=data(test_pi,test_g)
    model=nn.Sequential(nn.Linear(51,48),nn.GELU(),
                        nn.Linear(48,48),nn.GELU(),nn.Linear(48,1))
    probe=nn.Linear(1,1)
    with torch.no_grad():
        probe.weight.fill_(0)
        probe.bias.fill_(0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.008,weight_decay=0.0001)
    optimizer2=torch.optim.AdamW(probe.parameters(),lr=0.08,weight_decay=0)
    lossfn=nn.BCEWithLogitsLoss()
    for _ in range(steps):
        a=torch.randint(len(rp),(batch_size,))
        b=torch.randint(len(rn),(batch_size,))
        for m1,o1,P,N in ((model,optimizer,rp,rn),(probe,optimizer2,sp,sn)):
            o1.zero_grad()
            ya,yb=m1(P[a]),m1(N[b])
            loss=(lossfn(ya,torch.ones_like(ya))+lossfn(yb,torch.zeros_like(yb)))/2
            loss.backward()
            o1.step()
    with torch.no_grad():
        return {'seed':seed,'steps':steps,'train_matching_classes':len(train_g),
                'test_matching_classes':len(test_g),'train_perms':len(train_pi),
                'test_perms':len(test_pi),
                'train_bayes_balanced_accuracy':str(1-Fraction(1,2*len(train_g))),
                'test_bayes_balanced_accuracy':str(1-Fraction(1,2*len(test_g))),
                'raw_train':empirical_metrics(model(rp),model(rn)),
                'raw_test':empirical_metrics(model(tp),model(tn)),
                'structured_test':empirical_metrics(probe(tsp),probe(tsn)),
                'scope':'Hold out 3 whole matching classes; positives and negatives balanced separately within each cohort.'}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=20261115)
    args=ap.parse_args()
    print(json.dumps(train_family_holdout(seed=args.seed),indent=2))
