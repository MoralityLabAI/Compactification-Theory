"""SMC-03: orbit-stratified holdout, exact-pair controls, optional tiny CPU learning.

Standard-library exact controls run without a GPU or external model.
Training needs PyTorch. Transported-edge readout is oracle-assisted.
"""
import hashlib
import json
import random
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, permutations
from transport_lift import all_matchings,base_matching,relabel


def orbit_split(k=3,train_numerator=3,train_denominator=4):
    base=base_matching(k)
    perms=tuple(permutations(range(2*k)))
    states=tuple(all_matchings(tuple(range(2*k))))
    state_idx={g:i for i,g in enumerate(states)}
    idx=tuple(state_idx[relabel(p,base)] for p in perms)
    fibers=defaultdict(list)
    for i,g in enumerate(idx):fibers[g].append(i)
    train=[]; test=[]
    for g in range(len(states)):
        fiber=sorted(fibers[g],key=lambda i:hashlib.sha256(bytes(perms[i])).digest())
        n=len(fiber)*train_numerator//train_denominator
        train.extend(fiber[:n]);test.extend(fiber[n:])
    assert len(set(train)&set(test))==0
    assert len(train)+len(test)==len(perms)
    assert len(set(idx[i] for i in train))==len(states)
    assert len(set(idx[i] for i in test))==len(states)
    assert len({sum(idx[i]==g for i in train) for g in range(len(states))})==1
    assert len({sum(idx[i]==g for i in test) for g in range(len(states))})==1
    return {'k':k,'base':base,'perms':perms,'states':states,'orbit_index':idx,
            'train':tuple(sorted(train)),'test':tuple(sorted(test))}


def pair_features(spl):
    """Raw matching-edge bits, permutation matrix bits, and transported overlap."""
    n=2*spl['k']; states=spl['states']; perms=spl['perms']
    edges=tuple(combinations(range(n),2))
    graph_features=[[float(edge in g) for edge in edges] for g in states]
    perm_features=[[float(p[i]==j) for i in range(n) for j in range(n)] for p in perms]
    overlap=[[len(set(g)&set(states[spl['orbit_index'][pi]])) for g in states]
             for pi in range(len(perms))]
    return graph_features,perm_features,overlap


def exact_population_baseline(k=3):
    """Exactly matched class-conditional additive feature means."""
    s=orbit_split(k)
    G,P,_=pair_features(s)
    m=len(s['states'])
    for subset in ('train','test'):
        ids=s[subset]
        pos=[G[s['orbit_index'][i]]+P[i] for i in ids]
        neg=[G[g]+P[i] for i in ids for g in range(m)]
        assert len(pos)*m==len(neg)
        total_pos=[sum(x[j] for x in pos) for j in range(len(pos[0]))]
        total_neg=[sum(x[j] for x in neg) for j in range(len(neg[0]))]
        assert all(a*len(neg)==b*len(pos) for a,b in zip(total_pos,total_neg))
    return {'k':k,'matching_states':m,'permutation_states':len(s['perms']),
            'train_perms':len(s['train']),'test_perms':len(s['test']),
            'train_matching_fiber_size':len(s['train'])//m,
            'test_matching_fiber_size':len(s['test'])//m,
            'raw_feature_dim':len(G[0])+len(P[0]),
            'additive_population_gradient_at_zero':'0',
            'additive_logloss_optimal_balanced_accuracy':'1/2'}


def empirical_metrics(pos_scores,neg_scores):
    """Balanced accuracy across the class-conditional laws; 0 is negative."""
    tpr=float((pos_scores>0).float().mean())
    tnr=float((neg_scores<=0).float().mean())
    return {'balanced_accuracy':(tpr+tnr)/2,'tpr':tpr,'tnr':tnr}


def train_cpu(k=3,seed=20261115,steps=300,batch_size=256):
    """Fixed-budget tiny CPU raw MLP versus oracle-overlap logistic readout.

    Permutation-fiber representatives are SHA-256-disjoint. Every orbit
    matching has the same number of train/test representatives.
    """
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    random.seed(seed)
    s=orbit_split(k)
    m=len(s['states'])
    graph,perms,overlaps=pair_features(s)
    graph=torch.tensor(graph,dtype=torch.float32)
    perm=torch.tensor(perms,dtype=torch.float32)
    overlap=torch.tensor(overlaps,dtype=torch.float32)
    idx=torch.tensor(s['orbit_index'],dtype=torch.long)
    train_idx=torch.tensor(s['train'],dtype=torch.long)
    test_idx=torch.tensor(s['test'],dtype=torch.long)
    g_train=idx[train_idx]
    g_test=idx[test_idx]
    raw_pos=torch.cat((graph[g_train],perm[train_idx]),dim=-1)
    raw_neg=torch.cat((graph[None,:,:].expand(len(train_idx),m,-1),
                       perm[train_idx,None,:].expand(-1,m,-1)),dim=-1).reshape(-1,graph.shape[1]+perm.shape[1])
    raw_pos_test=torch.cat((graph[g_test],perm[test_idx]),dim=-1)
    raw_neg_test=torch.cat((graph[None,:,:].expand(len(test_idx),m,-1),
                            perm[test_idx,None,:].expand(-1,m,-1)),dim=-1).reshape(-1,graph.shape[1]+perm.shape[1])
    struct_pos=overlap[train_idx,g_train][:,None]
    struct_neg=overlap[train_idx,:].reshape(-1,1)
    struct_pos_test=overlap[test_idx,g_test][:,None]
    struct_neg_test=overlap[test_idx,:].reshape(-1,1)
    raw_model=nn.Sequential(nn.Linear(raw_pos.shape[-1],48),nn.GELU(),
                            nn.Linear(48,48),nn.GELU(),nn.Linear(48,1))
    struct_model=nn.Linear(1,1)
    with torch.no_grad():
        struct_model.weight.fill_(0)
        struct_model.bias.fill_(0)
    opt_raw=torch.optim.AdamW(raw_model.parameters(),lr=0.008,weight_decay=0.0001)
    opt_struct=torch.optim.AdamW(struct_model.parameters(),lr=0.08,weight_decay=0)
    history=[]
    loss_fn=nn.BCEWithLogitsLoss()
    for step in range(steps):
        posid=torch.randint(len(raw_pos),(batch_size,))
        negid=torch.randint(len(raw_neg),(batch_size,))
        for mdl,opt,features_pos,features_neg in ((raw_model,opt_raw,raw_pos,raw_neg),
                                                   (struct_model,opt_struct,struct_pos,struct_neg)):
            opt.zero_grad()
            yp=mdl(features_pos[posid])
            yn=mdl(features_neg[negid])
            loss=(loss_fn(yp,torch.ones_like(yp))+loss_fn(yn,torch.zeros_like(yn)))/2
            loss.backward()
            opt.step()
        if step in (0,steps//2,steps-1):
            history.append({'step':step+1,
                            'raw_loss':float(loss_fn(raw_model(raw_pos[:20]),torch.ones(20,1)).detach()),
                            'structured_logit_w':float(struct_model.weight.detach()[0,0]),
                            'structured_logit_b':float(struct_model.bias.detach()[0])})
    with torch.no_grad():
        raw_metrics=empirical_metrics(raw_model(raw_pos_test),raw_model(raw_neg_test))
        struct_metrics=empirical_metrics(struct_model(struct_pos_test),struct_model(struct_neg_test))
        raw_train=empirical_metrics(raw_model(raw_pos),raw_model(raw_neg))
        struct_train=empirical_metrics(struct_model(struct_pos),struct_model(struct_neg))
        exact_opt=1-Fraction(1,2*m)
        tau=list(range(2*k))
        tau[0],tau[2]=tau[2],tau[0]
        for pi in s['test']:
            g=s['states'][s['orbit_index'][pi]]
            p=s['perms'][pi]
            assert relabel(tau,g)==relabel(tuple(tau[p[i]] for i in range(2*k)),s['base'])
        perm_lookup={p:i for i,p in enumerate(s['perms'])}
        match_lookup={g:i for i,g in enumerate(s['states'])}
        transformed_perm=torch.tensor(
            [perm_lookup[tuple(tau[s['perms'][i][j]] for j in range(2*k))]
             for i in s['test']],dtype=torch.long)
        transformed_graph=torch.tensor(
            [match_lookup[relabel(tau,g)] for g in s['states']],dtype=torch.long)
        gauge_pos=torch.cat((graph[transformed_graph[g_test]],perm[transformed_perm]),dim=-1)
        gauge_neg=torch.cat((graph[transformed_graph[None,:]].expand(len(test_idx),-1,-1),
                             perm[transformed_perm,None,:].expand(-1,m,-1)),dim=-1).reshape(-1,raw_pos.shape[-1])
        base_pos_pred=raw_model(raw_pos_test)>0
        base_neg_pred=raw_model(raw_neg_test)>0
        gauge_pos_pred=raw_model(gauge_pos)>0
        gauge_neg_pred=raw_model(gauge_neg)>0
        gauge_agreement=float(((base_pos_pred==gauge_pos_pred).float().mean()+
                               (base_neg_pred==gauge_neg_pred).float().mean())/2)
        gauge_accuracy=empirical_metrics(raw_model(gauge_pos),raw_model(gauge_neg))
    return {'seed':seed,'k':k,'steps':steps,'batch_size_per_class':batch_size,
            'train_perms':len(train_idx),'test_perms':len(test_idx),
            'raw_mlp':raw_metrics,'raw_train':raw_train,
            'structured_overlap_logistic':struct_metrics,'structured_train':struct_train,
            'theoretical_bayes_balanced_accuracy':str(exact_opt),
            'marginal_only_best_balanced_accuracy':'1/2',
            'additive_logloss_optimum_accuracy':'1/2',
            'common_vertex_relabel_invariance_oracle':'1',
            'raw_mlp_common_vertex_relabel_accuracy':gauge_accuracy,
            'raw_mlp_gauge_prediction_agreement':gauge_agreement,
            'loss_checkpoints':history,
            'scope':'toy, synthetic perfect matchings; structured features use exact oracle alignment'}


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument('--train',action='store_true')
    a.add_argument('--steps',type=int,default=300)
    a.add_argument('--seed',type=int,default=20261115)
    a.add_argument('--k',type=int,default=3)
    args=a.parse_args()
    print(json.dumps(train_cpu(k=args.k,seed=args.seed,steps=args.steps)
                     if args.train else exact_population_baseline(args.k),indent=2))
