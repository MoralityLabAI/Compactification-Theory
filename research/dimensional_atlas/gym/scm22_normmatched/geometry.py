"""SCM-22: source-frozen training-state SAE and norm-matched causal directions."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import LoopedMath,Case,input_features

class TopKSAE(nn.Module):
    def __init__(self,d:int=48,width:int=96,k:int=8):
        super().__init__();self.k=k
        self.encoder=nn.Linear(d,width);self.decoder=nn.Linear(width,d)
    def forward(self,x):
        raw=F.relu(self.encoder(x));v,i=raw.topk(self.k,dim=-1)
        codes=raw.new_zeros(raw.shape).scatter(-1,i,v)
        return self.decoder(codes),codes

def frozen_states(model:LoopedMath,items:list[Case])->torch.Tensor:
    with torch.no_grad():
        data=[]
        for offset in range(0,len(items),128):
            _,states=model(input_features(items[offset:offset+128]),record=True)
            data.append(torch.stack(states,dim=1).detach())
        return torch.cat(data,dim=0)

def fit_sae(model:LoopedMath,states:torch.Tensor,seed:int,steps:int=250)->tuple[TopKSAE,dict]:
    torch.manual_seed(seed+701)
    flat=states.reshape(-1,states.shape[-1]).detach()
    scaled=flat/flat.square().mean(dim=-1,keepdim=True).clamp_min(1e-10).sqrt()
    sae=TopKSAE(d=scaled.shape[-1]);opt=torch.optim.Adam(sae.parameters(),lr=.002)
    gen=torch.Generator().manual_seed(seed+1101)
    for i in range(steps):
        ids=torch.randint(len(scaled),(256,),generator=gen)
        reco,_=sae(scaled[ids]);loss=F.mse_loss(reco,scaled[ids])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    sae.eval()
    with torch.no_grad():
        reco,codes=sae(scaled)
        uses=(codes>0).sum(0)
        mse=float(F.mse_loss(reco,scaled))
    return sae,{'width':96,'k':8,'steps':steps,'reconstruction_mse':mse,'active_frequencies':uses.tolist()}

def frame(a:torch.Tensor,rank:int=4)->torch.Tensor:
    # QR for a matrix with rank >= rank, in float64 then cast once.
    q,r=torch.linalg.qr(a.double(),mode='reduced')
    if q.shape[1]<rank or (torch.abs(torch.diagonal(r)[:rank])<1e-9).any():
        raise ValueError('rank-four frame degenerate')
    return q[:,:rank].float()

def readout_fiber(model:LoopedMath)->torch.Tensor:
    # Linear-head null only; final LN means this is not a nonlinear exact null.
    mat=model.head.weight.detach().double()
    _,_,v=torch.linalg.svd(mat,full_matrices=True)
    return v[11:,:].T.float()  # [48,37]

def make_frames(model:LoopedMath,sae:TopKSAE,sae_info:dict,calibration:list[tuple[Case,Case]],seed:int,visit:int=1)->tuple[dict,dict]:
    with torch.no_grad():
        a=frozen_states(model,[x for x,_ in calibration])[:,visit]
        b=frozen_states(model,[y for _,y in calibration])[:,visit]
        differences=b-a
        centered=differences-differences.mean(0)
        _,sigma,vt=torch.linalg.svd(centered.double(),full_matrices=False)
        diff=vt[:4].T.float()
        both=torch.cat((a,b));both-=both.mean(0)
        _,_,vstate=torch.linalg.svd(both.double(),full_matrices=False)
        raw=vstate[:4].T.float()
        # A fixed candidate SAE-fiber frame, from activation-frequency ranking alone.
        fiber=readout_fiber(model)
        freq=torch.tensor(sae_info['active_frequencies']);order=torch.argsort(freq,descending=True,stable=True)
        directions=sae.decoder.weight.detach()[:,order]
        projected=fiber@(fiber.T@directions)
        # Pivoted independent Gram-Schmidt, selecting first sufficiently independent SAE directions.
        chosen=[];feat_ids=[]
        for idx in range(projected.shape[1]):
            v=projected[:,idx].double().clone()
            for u in chosen:v-=u*torch.dot(u,v)
            v/=v.norm().clamp_min(1e-15)
            if not torch.isfinite(v).all() or torch.linalg.norm(projected[:,idx].double()-(torch.stack(chosen,1)@torch.stack(chosen,1).T@projected[:,idx].double() if chosen else 0))<1e-6:continue
            chosen.append(v);feat_ids.append(int(order[idx]));
            if len(chosen)==4:break
        if len(chosen)!=4:raise ValueError('SAE fiber directions rank < 4')
        sae_frame=torch.stack(chosen,dim=1).float()
        gen=torch.Generator().manual_seed(seed+9111)
        rand=torch.randn((fiber.shape[1],4),generator=gen)
        random_frame=frame(fiber@rand)
        frames={'difference_pca4':diff,'state_pca4':raw,'sae_fiber4':sae_frame,'random_fiber4':random_frame}
        for label,q in frames.items():
            assert torch.allclose(q.T@q,torch.eye(4),atol=2e-5),label
        var=torch.square(sigma[:4]).sum()/torch.square(sigma).sum().clamp_min(1e-20)
        return frames,{'calibration_pairs':len(calibration),'difference_singular_values':sigma[:8].float().tolist(),
                        'difference_top4_variance_fraction':float(var),'selected_sae_feature_ids':feat_ids,
                        'sae_fiber_linear_head_residual_max':float((model.head.weight.detach()@sae_frame).abs().max()),
                        'random_fiber_linear_head_residual_max':float((model.head.weight.detach()@random_frame).abs().max())}

def matched_patch(d:torch.Tensor,q:torch.Tensor,factor:float=1.)->tuple[torch.Tensor,torch.Tensor,int]:
    """Match per-case L2 norm to factor*||d||; return original projected energy fractions."""
    raw=(d@q)@q.T
    natural=raw.norm(dim=1,keepdim=True)
    target=d.norm(dim=1,keepdim=True)*factor
    fallback=(natural<1e-12).flatten()
    patch=raw/natural.clamp_min(1e-12)*target
    if fallback.any():patch[fallback]=target[fallback]*q[:,0][None,:]
    ratio=(natural / d.norm(dim=1,keepdim=True).clamp_min(1e-12)).flatten()
    return patch,ratio,int(fallback.sum())
