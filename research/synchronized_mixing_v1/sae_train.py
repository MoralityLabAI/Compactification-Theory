"""SMC-06: train a real sparse autoencoder on synthetic dense mixtures.

The neural data come from a hidden 15-atom linear dictionary. A source-pinned
matching simulator generates only 12 of 15 complete matching patterns during
SAE training, along with unlabeled singleton/pair anchor exposures to all
atoms. The SAE receives ONLY dense vectors. Labels and planted dictionary
are used only by the evaluation fixture and never by gradient descent.

Requires torch and numpy. All principal geometry proofs are stdlib-only.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transport_lift import all_matchings
from approx_geometry import recover_with_one_swap,decode_soft
from identify_transport import uniform_catalog,hidden_fixture,predict,version_space,calibration_cohort
from fractions import Fraction
import random

torch.set_num_threads(2)

DIM=40
LATENT=15
STEPS=9000
TRAIN_DATA_SEED=7301
N_SAMPLES=16000


def teacher_data(seed=TRAIN_DATA_SEED,n_samples=N_SAMPLES,d=40,anchor=True):
    rng=np.random.default_rng(seed)
    edges=tuple(combinations(range(6),2))
    matchings=tuple(all_matchings(range(6)))
    masks=np.array([[int(e in g) for e in edges] for g in matchings],dtype=np.float32)
    # Random signed linear teacher features, normalized per column-atom.
    dictionary=(rng.normal(size=(LATENT,d))/np.sqrt(d)).astype('float32')
    dictionary/=np.linalg.norm(dictionary,axis=1,keepdims=True)
    z=np.zeros((n_samples,LATENT),dtype=np.float32)
    for i in range(n_samples):
        t=rng.random()
        if anchor:
            if t<.70:
                chosen=(int(rng.integers(LATENT)),)
            elif t<.94:
                chosen=rng.choice(LATENT,size=2,replace=False)
            else:
                chosen=np.flatnonzero(masks[int(rng.integers(12))])
        else:
            chosen=np.flatnonzero(masks[int(rng.integers(12))])
        z[i,chosen]=rng.uniform(.75,1.25,size=len(chosen))
    dense=z @ dictionary + rng.normal(scale=.015,size=(n_samples,d))
    return torch.tensor(dense,dtype=torch.float32),torch.tensor(dictionary),masks


class SAE(nn.Module):
    def __init__(self,d=40,width=15):
        super().__init__()
        self.encoder=nn.Linear(d,width)
        self.decoder=nn.Linear(width,d,bias=False)
    def forward(self,x):
        acts=torch.relu(self.encoder(x))
        return self.decoder(acts),acts


def train(seed=101,steps=STEPS,anchors=True):
    torch.manual_seed(seed)
    x,hidden_dictionary,masks=teacher_data(anchor=anchors)
    sae=SAE(d=x.shape[1],width=LATENT)
    optimizer=torch.optim.Adam(sae.parameters(),lr=.003)
    history=[]
    for step in range(steps):
        batch=x[torch.randint(len(x),(256,))]
        reconstruction,features=sae(batch)
        mse=((reconstruction-batch)**2).sum(-1).mean()
        objective=mse+.0015*features.sum(-1).mean()
        optimizer.zero_grad();objective.backward();optimizer.step()
        with torch.no_grad():
            sae.decoder.weight.div_(sae.decoder.weight.norm(dim=0).clamp_min(1e-6))
        if step in (0,steps//2,steps-1):
            history.append({'step':step+1,'loss':float(objective.detach()),
                            'batch_squared_l2':float(mse.detach())})
    # Matching-state measurement episodes, not inputs to geometry training
    # until after SAE weights are frozen. Gaussian noise is independent.
    with torch.no_grad():
        clean=torch.tensor(masks) @ hidden_dictionary
        scores=[]
        for i in range(15):
            observed=clean[i]+torch.randn(64,x.shape[1])*.015
            features=torch.relu(sae.encoder(observed))
            scores.append(tuple(float(a) for a in features.mean(0)))
        scores=tuple(scores)
        full_squared_l2=float(((sae(x)[0]-x)**2).sum(-1).mean())
    return {'model':sae,'scores':scores,'history':history,
            'all_matching_masks':masks,'teacher_dictionary':hidden_dictionary,
            'train_reconstruction_l2':full_squared_l2,
            'train_data_stats':{'size':len(x),'dim':x.shape[1],
                                'singleton_anchor_rate':'.70' if anchors else '0',
                                'pair_anchor_rate':'.24' if anchors else '0',
                                'matching_train_rate':'.06' if anchors else '1',
                                'heldout_complete_matching_classes':[12,13,14]}}


def evaluate(seed=101,steps=STEPS,anchors=True,frame_seeds=(43,77,91),
             save_checkpoint=None):
    raw=train(seed,steps,anchors)
    scores=raw['scores']
    out={'seed':seed,'steps':steps,'anchors':anchors,
         'train_reconstruction_squared_l2':raw['train_reconstruction_l2'],
         'loss_history':raw['history'],
         'train_matching_class_count':12,'heldout_matching_class_count':3,
         'raw_support_source':'top3(mean_64_ReLU_encoder_activations)',
         'repair_budget':'at most one 1->0 and 0->1 swap across all twelve training rows',
         'geometry_status':'abstain',
         'model_parameter_count':sum(v.numel() for v in raw['model'].parameters()),
         'noise_measurement_std':.015,'source_data_seed':TRAIN_DATA_SEED,
         'train_data_stats':raw['train_data_stats']}
    try:
        certificate=recover_with_one_swap(scores[:12])
    except ValueError as e:
        out['reason']=str(e)
        return out,raw
    geometry=certificate['geometry']
    out.update(geometry_status='certified',repairs=[list(z) for z in certificate['repairs']],
               candidate_stars=geometry['n_candidate_stars'],
               inferred_vertices=geometry['n_vertices_inferred'],
               inferred_group_order=geometry['group_order_derived'],
               search_nodes=geometry['search_nodes'])
    # Frame inference receives only (input permutation, decoded graph in
    # reconstructed gauge). The fixture's actual secret stays host-owned.
    catalogue=uniform_catalog()
    decoded=[]
    margins=[]
    for row in scores:
        m,margin=decode_soft(geometry,row)
        decoded.append(catalogue['mindex'][m]);margins.append(margin)
    out['heldout_prediction_margins']=margins[12:]
    out['calibrations']=[]
    for fs in frame_seeds:
        secret=hidden_fixture(catalogue,fs)
        train_perm,test_perm=calibration_cohort(catalogue,secret)
        rng=random.Random(fs+29+991)
        inputs=[rng.choice(train_perm) for _ in range(12)]
        observations=tuple((pid,decoded[predict(catalogue,secret,pid)])
                           for pid in inputs)
        hypotheses=version_space(catalogue,observations)
        predicted=hypotheses[0] if hypotheses else None
        true_train={i:decoded[i] for i in range(12)}
        # The population positive/negative test uses three held-out state
        # labels separately. Match both graph and permutation marginals.
        test_true=tuple(decoded[predict(catalogue,secret,pid)] for pid in test_perm)
        pred_out=tuple(predict(catalogue,predicted,pid) for pid in test_perm) if predicted else ()
        true_positive=sum(p==g for p,g in zip(pred_out,test_true))
        false_positive=sum(p==decoded[g] for p in pred_out for g in (12,13,14))
        tpr=Fraction(true_positive,len(test_perm)) if predicted else Fraction(0)
        fpr=Fraction(false_positive,len(test_perm)*3) if predicted else Fraction(0)
        balanced=(1+tpr-fpr)/2
        # An evaluation oracle trained with ALL train pairs is only used to
        # compare model selection, not to choose the 12-pair estimator.
        full=version_space(catalogue,tuple((i,decoded[predict(catalogue,secret,i)]) for i in train_perm))
        out['calibrations'].append({'frame_seed':fs,'n_calibration_pairs':12,
              'n_candidates':len(hypotheses),'unique_all_train_candidate':len(full)==1,
              'matches_full_data_oracle':len(full)==1 and predicted==full[0],
              'heldout_balanced_accuracy':str(balanced),'tpr':str(tpr),'fpr':str(fpr)})
    if save_checkpoint:
        serial={'seed':seed,'data_seed':TRAIN_DATA_SEED,'steps':steps,
                'architecture':'40 -> 15 ReLU SAE -> 40',
                'encoder_weight':raw['model'].encoder.weight.detach().tolist(),
                'encoder_bias':raw['model'].encoder.bias.detach().tolist(),
                'decoder_weight':raw['model'].decoder.weight.detach().tolist(),
                'scores_15_by_15':scores,
                'geometry_status':out['geometry_status'],
                'training_metadata':out['train_data_stats']}
        Path(save_checkpoint).write_text(json.dumps(serial,indent=2)+'\n')
    return out,raw


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,nargs='+',default=[101,102,103])
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--no-anchors',action='store_true')
    p.add_argument('--out',type=Path,default=Path('smc06_results.json'))
    p.add_argument('--checkpoint-seed',type=int,default=-1)
    args=p.parse_args()
    reports=[]
    for seed in args.seeds:
        checkpoint=args.out.with_suffix(f'.seed{seed}.json') if seed==args.checkpoint_seed else None
        report,_=evaluate(seed,args.steps,anchors=not args.no_anchors,save_checkpoint=checkpoint)
        reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('calibrations','loss_history')}),flush=True)
    args.out.write_text(json.dumps({'schema':'smc06_sae_dense_v1','reports':reports,
      'limits':'synthetic dense residual generator and actual trained SAE, not pretrained transformer SAE'},indent=2)+'\n')
