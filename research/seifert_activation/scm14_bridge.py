"""Bridge verified Hessian *upper bounds* into SCM-13 affine disks.

This converter is a CONDITIONAL geometry interface: an exact-real Jacobian
and exact affine anchor have to be certified independently. JSON containing
plausible-looking floats cannot attest the model's derivative at the origin.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from scm13_affine_disk import affine_disk_certificate
from rmp_lipschitz import rational


SUPPORTED = 'scm14/two_coordinate_source_hessian/v1'


def conditional_affine_bridge(hessian_receipt:dict, *, visit:int,
                              base:Sequence[Sequence],
                              jacobians:Sequence[Sequence[Sequence]],
                              exact_affine_fixture:bool=False,
                              margin='0')->dict:
    """Use H_i/2 as the nonlinear error for a normalized unit disk.

    `exact_affine_fixture=True` is intended ONLY for analytic toy inputs;
    the function does not inspect or attest the provided base/Jacobian.
    For neural source claims it reports unresolved exact-real anchor/Jacobian.
    """
    if hessian_receipt.get('schema') != SUPPORTED:
        raise ValueError('Hessian receipt has unexpected schema')
    if hessian_receipt.get('scope') != 'ideal_real_RMP_weights_conditional_CPU_binary64_BLAS_center_verifier':
        raise ValueError('Unrecognized source-level proof scope')
    if not hessian_receipt.get('hessian_upper_certified_under_source_contract',False):
        raise ValueError('Missing source-level Hessian upper bound')
    if hessian_receipt.get('intervention_coordinate') != 'u_in_unit_disk; patch=amplitude*axes@u':
        raise ValueError('Hessian coordinate mismatch')
    if not isinstance(visit,int) or visit<1:
        raise ValueError('Expected a positive integer visit')
    entries=hessian_receipt.get('per_visit',[])
    if visit>len(entries) or entries[visit-1].get('visit')!=visit:
        raise ValueError('Visit not present in source Hessian receipt')
    observed=entries[visit-1]['observed']
    tokens=hessian_receipt.get('observed_token_positions')
    if (not isinstance(tokens,list) or len(tokens)<2 or
        any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in tokens) or
        len(set(tokens))!=len(tokens) or set(observed)!={str(x) for x in tokens}):
        raise ValueError('Missing or mismatched source observed-token order')
    if len(base)!=len(jacobians) or len(base)!=len(tokens):
        raise ValueError('Affine witnesses must match the number of observed tokens')
    herrors=[]
    for token in tokens:
        rec=observed[str(token)]
        raw=rec.get('taylor_remainder_on_unit_disk_upper')
        if raw is None:
            raise ValueError('Missing per-token Taylor remainder')
        h=_strict_bound(raw)
        herrors.append(h)
    r=affine_disk_certificate(base,jacobians,1,herrors,margin=margin)
    r['bridge_schema']='scm14/conditional_affine_bridge/v1'
    r['source_hessian_claim_requires_exact_replay']=True
    r['source_hessian_verified_under_CPU_contract']=True
    r['exact_affine_anchors_and_jacobians_verified']=False
    r['analytic_fixture_label']=bool(exact_affine_fixture)
    r['model_level_topology_certified']=False
    r['description']='requires independently source-certified exact-real base/Jacobian for trained-model inference'
    return r


def _strict_bound(value):
    if isinstance(value,bool) or not isinstance(value,(int,str,Fraction)):
        raise ValueError('Hessian bound must be exact rational string or integer')
    result=rational(value)
    if result<0:
        raise ValueError('Negative source curvature error')
    return result
