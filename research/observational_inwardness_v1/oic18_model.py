"""OIC-18 preregistered phase-emission architecture comparison.

These models inherit the OIC-16 learned five-offset circulant transition
kernels and raw-state likelihood filter. Model labels and hidden initial phase
are supplied only to the data generator during supervised training.

All extra n-dependent sharpening is an explicit inductive bias with n known.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from oic16_model import LearnedBelief

REFERENCE_PHASE_COUNT = 5.0
ARCHITECTURES = ('fourier','temperature','von_mises','harmonic3')

class PhaseEmissionBelief(LearnedBelief):
    def __init__(self, architecture: str):
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        super().__init__()
        self.architecture = architecture
        if architecture=='temperature':
            # Centered Fourier logit: marker value is c0, rather than c0+c1+c2.
            with torch.no_grad():self.emission_coeff.copy_(torch.tensor([1.25,3.0,0.25]))
            self.temperature_raw = nn.Parameter(torch.tensor(math.log(.80/.20)))
        elif architecture=='von_mises':
            # Emission logit a - k(n)(1-cos theta), k(n)=softplus(raw)*(n/5)^2.
            del self.emission_coeff
            self.marker_logit = nn.Parameter(torch.tensor(1.25))
            self.concentration_raw = nn.Parameter(torch.tensor(3.8))
        elif architecture=='harmonic3':
            # Equal-size three-coefficient Fourier control: second harmonic -> third.
            pass

    def phase_emission(self,n:int):
        if not isinstance(n,int) or n < 2: raise ValueError('n must be an integer >= 2')
        theta=2*math.pi*torch.arange(n,device=self.kernel_logits.device,dtype=self.kernel_logits.dtype)/n
        if self.architecture=='fourier':
            return super().phase_emission(n)
        if self.architecture=='harmonic3':
            c=self.emission_coeff
            logits=c[0]+c[1]*torch.cos(theta)+c[2]*torch.cos(3*theta)
        elif self.architecture=='temperature':
            c=self.emission_coeff
            rho=torch.sigmoid(self.temperature_raw)
            tau=1+rho*((n/REFERENCE_PHASE_COUNT)**2-1)
            logits=c[0]+tau*(c[1]*(torch.cos(theta)-1)+c[2]*(torch.cos(2*theta)-1))
        elif self.architecture=='von_mises':
            k=torch.nn.functional.softplus(self.concentration_raw)*(n/REFERENCE_PHASE_COUNT)**2
            logits=self.marker_logit-k*(1-torch.cos(theta))
        return torch.sigmoid(logits)

    def emission_parameters(self)->dict:
        if self.architecture=='von_mises':
            return {'marker_logit':float(self.marker_logit.detach()),
                    'concentration_base':float(torch.nn.functional.softplus(self.concentration_raw).detach())}
        if self.architecture=='temperature':
            return {'coefficients':[float(c) for c in self.emission_coeff.detach()],
                    'temperature_fraction':float(torch.sigmoid(self.temperature_raw).detach())}
        return {'coefficients':[float(c) for c in self.emission_coeff.detach()]}

    def resource_ledger(self,n:int)->dict:
        count=sum(p.numel() for p in self.parameters())
        return {'architecture':self.architecture,'parameter_count':count,
                'fp32_parameter_bits':count*32,
                'state_float_coordinates_per_episode':4*n,
                'state_fp32_bits_per_episode':128*n,
                'convolution_shift_terms_per_step':2*5,
                'phase_emission_evaluations_per_forward':n,
                'phase_slots':2*n,
                'baseline_noise_raw_readout':True}


def emit_from_weights(architecture: str,n:int,weights:dict) -> list[float]:
    """Standalone scalar emission, for independent pure-Python verification."""
    if architecture not in ARCHITECTURES or n<2:raise ValueError('invalid architecture or n')
    sigmoid=lambda v: (1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v)))
    softplus=lambda v: math.log1p(math.exp(-abs(v)))+max(v,0)
    if architecture=='von_mises':
        m=weights['marker_logit'];r=weights['concentration_raw']
        marker=float(m[0] if isinstance(m,list) else m)
        raw=float(r[0] if isinstance(r,list) else r)
        conc=softplus(raw)*(n/REFERENCE_PHASE_COUNT)**2
        return [sigmoid(marker-conc*(1-math.cos(2*math.pi*z/n))) for z in range(n)]
    c=weights['emission_coeff']
    if architecture=='temperature':
        raw=weights['temperature_raw'];raw=float(raw[0] if isinstance(raw,list) else raw)
        tau=1+sigmoid(raw)*((n/REFERENCE_PHASE_COUNT)**2-1)
        return [sigmoid(c[0]+tau*(c[1]*(math.cos(2*math.pi*z/n)-1)
                    +c[2]*(math.cos(4*math.pi*z/n)-1))) for z in range(n)]
    if architecture=='harmonic3':
        return [sigmoid(c[0]+c[1]*math.cos(2*math.pi*z/n)+c[2]*math.cos(6*math.pi*z/n)) for z in range(n)]
    return [sigmoid(c[0]+c[1]*math.cos(2*math.pi*z/n)+c[2]*math.cos(4*math.pi*z/n)) for z in range(n)]
