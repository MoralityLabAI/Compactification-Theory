"""Actual BPDModel/TiedComponentLinear physical wrapper numerical parity.

Requires pinned MoralityLabAI/BPD package checked out by hosted GitHub Actions:
f22a883144ee7d2caa2bd5d9936dd9e083c7f139.
No native fake or surrogate implementation is substituted if import fails.
"""
from __future__ import annotations

import argparse
import json
import torch
from torch import nn

from research.causal_gauge_compactification.cgc02.experiment import (
    D, WIDTH, XD, RecurrentTeacher, fit_factors, matched_source_pairs,
    train_teacher,
)


class PhysicalCore(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.core=nn.Linear(WIDTH,D,bias=False)
        with torch.no_grad():
            self.core.weight.copy_(model.C.T)

    def forward(self,features):
        return self.core(features)


def native_parity(seed=901,train_steps=650,factor_steps=360,n=48):
    if seed not in (901,902):
        raise ValueError("native engineering is restricted to excluded roots")
    from mechinterp.bpd.components import (
        BPDModel, PassMaskSchedule, ComponentIntervention,
    )
    from mechinterp.bpd.config import AblationConfig

    torch.set_num_threads(1)
    model=train_teacher(seed,updates=train_steps)
    factors=fit_factors(model,seed,0,updates=factor_steps)
    physical=factors[0]@factors[1]
    native=BPDModel(
        PhysicalCore(model),{"core":D},max_passes=9,
        ablation=AblationConfig(mode="zero",sphere_rescale=False),
        init_svd=False,
    )
    wrapper=native.wrappers["core"]
    h0,x,donor,_,_=matched_source_pairs(seed*741+77,n)
    with torch.no_grad():
        wrapper.V.copy_(factors[0])
        wrapper.U.copy_(factors[1])
        schedule=PassMaskSchedule(
            torch.ones(n,9,1,D),torch.zeros(n,9,1))
        direct=h0.clone()
        actual=h0.clone()
        with native.masked_components({"core":schedule}):
            for i in range(9):
                event=x if i==0 else torch.zeros_like(x)
                phi_actual=model.features(actual,event)
                write=native.target_model(phi_actual.unsqueeze(1))[:,0]
                actual=actual+write
                direct=direct+model.features(direct,event)@physical
        rollout_error=float((actual-direct).abs().max())

        # The target mode uses the original frozen C plus a full donor
        # physical BPD factor write, independent of arbitrary factor gauge.
        recipient_features=model.features(h0,x)
        donor_features=model.features(h0,donor)
        donor_acts=(donor_features@factors[0]).unsqueeze(1)
        wrapper.mode="target"
        wrapper.reset_pass_state(pass_offset=0)
        wrapper.intervention=ComponentIntervention(
            components=tuple(range(D)),
            passes=frozenset({0}),
            mode="swap_acts",
            donor_acts=donor_acts,
        )
        try:
            swapped=native.target_model(recipient_features.unsqueeze(1))[:,0]
        finally:
            wrapper.intervention=None
        expected=recipient_features@model.C+(
            donor_features-recipient_features)@physical
        swap_error=float((swapped-expected).abs().max())

    result={
        "excluded_seed":seed,
        "bpd_pin":"f22a883144ee7d2caa2bd5d9936dd9e083c7f139",
        "native_nine_shared_invocations_max_error":rollout_error,
        "native_full_component_donor_swap_max_error":swap_error,
        "factor_weight_mse":float((physical-model.C).square().mean()),
        "native_parity_pass":rollout_error<=1e-5 and swap_error<=1e-5,
    }
    if not result["native_parity_pass"]:
        raise AssertionError(result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=901)
    p.add_argument("--teacher-steps",type=int,default=650)
    p.add_argument("--factor-steps",type=int,default=360)
    p.add_argument("--out")
    a=p.parse_args()
    r=native_parity(a.seed,a.teacher_steps,a.factor_steps)
    print(json.dumps(r,indent=2))
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(r,indent=2)+"\n")
