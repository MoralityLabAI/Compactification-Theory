"""SCM-11 exact LayerNorm-shift symmetry, a causal negative control.

For pre-LN RMP B(X + U) = B(X) + alpha**2 * U whenever every token
row of U is a scalar multiple of the all-ones hidden vector. The equality
is for the mathematical real-valued network and arbitrary finite shifts.
Input reinjection must be fixed with respect to the intervention.
"""
from __future__ import annotations
from fractions import Fraction
from typing import Sequence
import torch
from rmp_lipschitz import rational, block_bound


def gauge_axis_null_control(model, axes:torch.Tensor, *,injected_token:int,
                            observed_tokens:Sequence[int],sequence:int):
    """Certify a cyclic axis as invisible to the chosen downstream chart.

    If one intervention column is EXACTLY channel-constant, all observed
    slots differ from the clean state only through the other angular
    coordinate. The loop factors through [-1,1] and is nullhomotopic
    PROVIDED the entire path remains collision-free in the frozen chart.
    Exact float equality of axis elements supplies a source-level algebraic
    premise; it does not certify float32/CUDA numerical invariance.
    """
    if model.training or not hasattr(model,'blocks') or not hasattr(model,'module_indices'):
        raise ValueError('eval pinned RMP schedule required')
    if axes.ndim!=2 or axes.shape!=(model.hidden_size,2) or not torch.isfinite(axes).all():
        raise ValueError('finite [hidden,2] axes required')
    if not 0<=injected_token<sequence:
        raise ValueError('invalid injection token')
    observed=tuple(int(i) for i in observed_tokens)
    if len(observed)<2 or len(set(observed))!=len(observed) or any(not 0<=j<sequence for j in observed):
        raise ValueError('invalid observed tokens')
    if injected_token in observed:
        raise ValueError('gauge-insensitivity claim requires excluding injected slot')
    if getattr(model,'carry_policy','reset')!='reset':
        raise ValueError('unsupported carry policy')
    for index in model.module_indices:
        block=model.blocks[index]
        block_bound(block,sequence=sequence)
        if block.mlp[0].bias is not None or block.mlp[2].bias is not None:
            raise ValueError('unregistered MLP architecture')
    rows=axes.detach().cpu().to(torch.float64).tolist()
    exact=[]
    for col in (0,1):
        xs=[rational(float(row[col])) for row in rows]
        if xs[0]!=0 and all(x==xs[0] for x in xs):
            exact.append((col,str(xs[0])))
    if len(exact)!=1:
        raise ValueError('exactly one nonzero channel-constant axis required')
    return {'schema':'scm11_exact_preLN_gauge_null_v1',
       'source_theorem':'B(X+U)=B(X)+alpha_squared_U_for_channel_constant_per_token_U',
       'invisible_axis':exact[0][0],
       'exact_channel_value':exact[0][1],
       'observed_tokens':list(observed),
       'injected_token':int(injected_token),
       'link_topology':'conditional_trivial_if_collision_free_over_other_axis_interval',
       'collision_free_interval_certified':False,
       'physical_float_execution_certified':False}
