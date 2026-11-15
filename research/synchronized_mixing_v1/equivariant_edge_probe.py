"""SMC-03c: 34-parameter shared-edge probe with explicit group transport.

Requires PyTorch only for training. The vertex-permutation transport is
given as an inductive bias; the local shared edge interaction is learned.
"""
import json
from family_holdout import family_split
from relational_probe import pair_features


def train_equivariant(seed=20261115,steps=300,hidden=8,batch_size=256):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    fs=family_split()
    s=fs['split']
    gfeats,_,_=pair_features(s)
    graph=torch.tensor(gfeats,dtype=torch.float32)
    orbit_idx=torch.tensor(s['orbit_index'],dtype=torch.long)
    transported=graph[orbit_idx]

    def data(pis,allowed):
        p=torch.tensor(pis,dtype=torch.long)
        gs=torch.tensor(allowed,dtype=torch.long)
        positives=torch.stack((graph[orbit_idx[p]],transported[p]),dim=-1)
        a=graph[gs][None,:,:].expand(len(p),-1,-1)
        b=transported[p,None,:].expand(-1,len(gs),-1)
        negatives=torch.stack((a,b),dim=-1).reshape(-1,graph.shape[1],2)
        return positives,negatives

    xp,xn=data(fs['train_pi'],fs['train_g'])
    tp,tn=data(fs['test_pi'],fs['test_g'])

    class EdgeProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi=nn.Sequential(nn.Linear(2,hidden),nn.Tanh(),nn.Linear(hidden,1))
            self.bias=nn.Parameter(torch.zeros(()))
        def forward(self,x):
            return self.phi(x).sum(dim=1)+self.bias

    mdl=EdgeProbe()
    opt=torch.optim.AdamW(mdl.parameters(),lr=0.01,weight_decay=0.0001)
    lossfn=nn.BCEWithLogitsLoss()
    for step in range(steps):
        a=torch.randint(len(xp),(batch_size,))
        b=torch.randint(len(xn),(batch_size,))
        opt.zero_grad()
        yp=mdl(xp[a])
        yn=mdl(xn[b])
        loss=(lossfn(yp,torch.ones_like(yp))+lossfn(yn,torch.zeros_like(yn)))/2
        loss.backward()
        opt.step()

    with torch.no_grad():
        def acc(p,n):
            tpr=float((mdl(p)>0).float().mean())
            tnr=float((mdl(n)<=0).float().mean())
            return {'balanced_accuracy':(tpr+tnr)/2,'tpr':tpr,'tnr':tnr}

        # A common gauge permutation merely permutes the edge axis.
        edges=[(a,b) for a in range(6) for b in range(a+1,6)]
        id_edge={edge:i for i,edge in enumerate(edges)}
        sigma=[2,1,0,3,4,5]
        permute=[id_edge[tuple(sorted((sigma[a],sigma[b])))] for a,b in edges]
        reverse=[0]*len(permute)
        for old,new in enumerate(permute):
            reverse[new]=old
        gt=tp[:,reverse,:]
        gn=tn[:,reverse,:]
        same_positive=float(((mdl(tp)>0)==(mdl(gt)>0)).float().mean())
        same_negative=float(((mdl(tn)>0)==(mdl(gn)>0)).float().mean())
        max_score_delta=max(float((mdl(tp)-mdl(gt)).abs().max()),
                            float((mdl(tn)-mdl(gn)).abs().max()))
        return {'seed':seed,'steps':steps,'hidden_edge_width':hidden,
                'train':acc(xp,xn),'test':acc(tp,tn),
                'gauge_binary_agreement_pos':same_positive,
                'gauge_binary_agreement_neg':same_negative,
                'gauge_max_score_abs_difference':max_score_delta,
                'parameter_count':sum(p.numel() for p in mdl.parameters()),
                'scope':'Known group transport, learned shared edge interaction. No raw permutation-to-matching discovery.'}


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=20261115)
    args=ap.parse_args()
    print(json.dumps(train_equivariant(seed=args.seed),indent=2))
