"""OIC-16: learn family-conditioned circulant phase transitions from public histories.

No exact transition shift or initial latent phase is fed to the observer.
Supervised family labels occur only in the loss. All coefficients are shared
across phase counts n; the model has 2n latent phase-family hypotheses.
Raw sensor readings are scored directly, with independent per-sample noise.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

SHIFTS=(-2,-1,0,1,2)

class LearnedBelief(nn.Module):
    def __init__(self):
        super().__init__()
        # Free five-offset kernels, softmax yields row stochastic circulant matrices.
        self.kernel_logits = nn.Parameter(torch.zeros(2, len(SHIFTS)))
        # Shared phase emission, with a smooth Fourier basis and learned weights.
        self.emission_coeff = nn.Parameter(torch.tensor([-2.0, 3.0, 0.25]))
        self.noise_logit = nn.Parameter(torch.tensor(-3.3))

    def kernel(self):
        return torch.softmax(self.kernel_logits,dim=-1)

    def phase_emission(self,n:int):
        z=torch.arange(n,device=self.kernel_logits.device,dtype=self.kernel_logits.dtype)
        a=2*math.pi*z/n
        p=self.emission_coeff
        return torch.sigmoid(p[0]+p[1]*torch.cos(a)+p[2]*torch.cos(2*a))

    def forward(self,n:int,u:torch.Tensor,x:torch.Tensor,return_state:bool=False):
        B,T=u.shape
        if x.shape!=(B,T+1):raise ValueError('x should be [batch,T+1]')
        pobs=torch.sigmoid(self.noise_logit)*0.49/0.5  # strictly below 0.49
        p_kernel=self.kernel()
        g=self.phase_emission(n).view(1,1,n,1)
        mass=torch.zeros(B,2,n,2,device=u.device,dtype=p_kernel.dtype)
        mass[:,:,:,0]=1.0/(2*n)
        def condition(m,raw):
            valid=(raw>=0).to(m.dtype).view(B,1,1,1)
            obs=(raw.clamp(min=0)).view(B,1,1,1)
            v=torch.arange(2,device=u.device).view(1,1,1,2)
            like=torch.where(v==obs,1-pobs,pobs)
            like=like*valid+(1-valid)
            m=m*like
            return m/m.sum((1,2,3),keepdim=True).clamp(min=1.e-38)
        mass=condition(mass,x[:,0])
        evidence=[]
        for t in range(T):
            pulse=g*u[:,t].reshape(B,1,1,1).to(g.dtype)
            before=mass*pulse+mass.flip(-1)*(1-pulse)
            # Convolve each hypothesis-family phase vector with its learned
            # circulant row-stochastic kernel, preserving total family mass.
            fam=[]
            for family in range(2):
                evolved=sum(p_kernel[family,k]*torch.roll(before[:,family,:,:],shifts=shift,dims=1)
                            for k,shift in enumerate(SHIFTS))
                fam.append(evolved)
            mass=torch.stack(fam,dim=1)
            mass=condition(mass,x[:,t+1])
            by_family=mass.sum((2,3)).clamp(min=1.e-38)
            evidence.append(torch.log(by_family[:,1])-torch.log(by_family[:,0]))
        logits=torch.stack(evidence,dim=1)
        if return_state: return logits,mass
        return logits

    def kernel_facts(self,n:int)->dict:
        import numpy as np
        p=self.kernel().detach().cpu().numpy()
        g=self.phase_emission(n).detach().cpu().tolist()
        shifts=list(SHIFTS)
        facts=[]
        for family in range(2):
            weights=p[family]
            ring=np.zeros(n)
            for w,k in zip(weights,shifts):ring[k%n]+=float(w)
            eig=np.fft.fft(ring)
            # After minimizing over cyclic orientation reversal, the nearest
            # single-step generator can be forward or backward.
            steps=[0] if family==0 else [1,n-1]
            target_d=min(float(np.linalg.norm(ring-np.eye(1,n,step)[0])) for step in steps)
            facts.append({'shift_weights':{str(k):float(w) for k,w in zip(shifts,weights)},
                          'mass_on_target_generator':float(ring[0] if family==0 else max(ring[1],ring[-1])),
                          'generator_l2_distance':target_d,
                          'eigenvalues':[[float(v.real),float(v.imag)] for v in eig]})
        return {'kernels':facts,'emission':g,'raw_flip_probability':float(torch.sigmoid(self.noise_logit).detach().item()*0.98)}
