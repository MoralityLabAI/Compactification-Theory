"""LSV-15b: exact rational maximal *sufficient* transverse radii.

The radius is optimal for the proved LayerNorm perturbation inequality and
frozen interval lower margins, not a claim of the true adversarial radius.
"""
from __future__ import annotations
import json
from fractions import Fraction as Q
from gauge_quotient import Bound, normal_output_bounds, head_intervals, replay_receipt


def exact_thresholds(case,weights):
    clean,sigma=normal_output_bounds(case['u_intervals'],weights['gamma_hex'],
                                     weights['beta_hex'],weights['epsilon_hex'])
    logits=head_intervals(clean,weights['head_weight_hex'],weights['head_bias_hex'])
    winner,rival=case['winner_digit'],case['rival_digit']
    s=sigma.lo
    g=[Q.from_float(float.fromhex(x)) for x in weights['gamma_hex']]
    w=[[Q.from_float(float.fromhex(x)) for x in row] for row in weights['head_weight_hex']]
    U=[Bound.from_arb(x).max_abs() for x in case['u_intervals']]
    perfeature=[abs(g[j])*(1+U[j]/s) for j in range(48)]

    def strict_radius(a,k):
        margin=(logits[a]-logits[k]).lo
        if margin<=0:
            raise ValueError('frozen clean class ordering is not provable')
        sensitivity=sum(abs(w[a][j]-w[k][j])*perfeature[j] for j in range(48))
        return (s*margin/(margin+sensitivity)) if sensitivity else s

    winner_thresholds={k:strict_radius(winner,k) for k in range(10) if k!=winner}
    second_thresholds={k:strict_radius(rival,k) for k in range(10) if k not in (winner,rival)}
    winner_limiting=min(winner_thresholds,key=winner_thresholds.get)
    second_limiting=min(second_thresholds,key=second_thresholds.get)
    top1=winner_thresholds[winner_limiting]
    top2=min(top1,second_thresholds[second_limiting])
    if not (Q(0)<top2<=top1<=s):
        raise AssertionError('invalid sufficient radius ordering')
    return {'case_index':case['case_index'],
            'winner_threshold_strict':str(top1),
            'winner_limiting_digit':winner_limiting,
            'top_two_threshold_strict':str(top2),
            'runner_limiting_digit':second_limiting,
            'sigma_lower_bound':str(s)}


def proof_summary(receipt,lock,baseline):
    replay_receipt(receipt,lock,baseline,require_complete=True)
    gamma=[Q.from_float(float.fromhex(x)) for x in receipt['weights']['gamma_hex']]
    rank=47 if all(v!=0 for v in gamma) else None
    rows=[exact_thresholds(c,receipt['weights']) for c in receipt['cases']]
    win_min=min(rows,key=lambda x:Q(x['winner_threshold_strict']))
    top2_min=min(rows,key=lambda x:Q(x['top_two_threshold_strict']))
    return {'schema':'LSV-15b/exact-gauge-thresholds-v1',
            'checkpoint_sha256':receipt['checkpoint_sha256'],
            'roster_sha256':receipt['roster_sha256'],
            'selected_cell_count':len(rows),
            'layernorm_gamma_all_nonzero':rank is not None,
            'layernorm_per_token_quotient_rank':rank,
            'per_token_gauge_dimension':1,
            'n_tokens':81,
            'complete_gauge_kernel_dimension':81 if rank==47 else None,
            'complete_layernorm_rank':81*47 if rank==47 else None,
            'min_all_cases_top1_strict_radius':win_min['winner_threshold_strict'],
            'min_all_cases_top1_limiting_case':win_min['case_index'],
            'min_all_cases_top_two_strict_radius':top2_min['top_two_threshold_strict'],
            'min_all_cases_top_two_limiting_case':top2_min['case_index'],
            'meaning':'strict r<T sufficient for the rational bound; arbitrarily large per-token gauge shifts remain invisible',
            'case_thresholds':rows}


def verify_threshold_receipt(summary,original,lock,baseline):
    recomputed=proof_summary(original,lock,baseline)
    if summary!=recomputed:
        raise ValueError('LSV-15b summary or case thresholds altered')
    return {'verified_cases':recomputed['selected_cell_count'],
            'min_top1_strict_radius':recomputed['min_all_cases_top1_strict_radius'],
            'min_top2_strict_radius':recomputed['min_all_cases_top_two_strict_radius']}
