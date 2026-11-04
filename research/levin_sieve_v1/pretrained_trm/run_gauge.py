"""LSV-15: pinned exact-real checkpoint and correlated LayerNorm gauge audit.

This verifies structured interventions at the FINAL pre-LayerNorm bus. The
full-stage-one ambient rank-4 cube remains LSV-14 (radius 1e-14). The source
of the centered inputs is trusted Arb evaluation of the pinned recurrent graph.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from flint import arb, ctx

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_checkpoint import locked_bytes, load_pretrained, prepare_roster
from validated_minor import (State, CertifiedBlock, exact_matrix,
    _jet_linear, _jet_transpose, _jet_add, _jet_gelu, _jet_layernorm,
    serialized_ball_contains_original)
from gauge_quotient import certificate_for_case, replay_receipt


def pre_final_ln2(net, stage1):
    """Enclose final pre-ln2 inputs after three recurrent passes from z1."""
    st=State(exact_matrix(stage1), ())
    # Original RNN uses one shared block for all three remaining passes.
    for _ in range(2):
        st=net(st)
    tokens=_jet_linear(_jet_transpose(st),net.t0,net.t0b)
    tokens=_jet_linear(_jet_gelu(tokens),net.t1,net.t1b)
    first=_jet_layernorm(_jet_add(st,_jet_transpose(tokens)),net.ln1)
    channels=_jet_linear(first,net.c0,net.c0b)
    channels=_jet_linear(_jet_gelu(channels),net.c1,net.c1b)
    return _jet_add(first,channels).value


def exact_weights(model):
    def hx(x):return float(x).hex()
    ln=model.block.ln2
    head=model.head
    return {'gamma_hex':[hx(z) for z in ln.weight.detach().cpu().tolist()],
            'beta_hex':[hx(z) for z in ln.bias.detach().cpu().tolist()],
            'epsilon_hex':hx(ln.eps),
            'head_weight_hex':[[hx(x) for x in row] for row in head.weight.detach().cpu().tolist()],
            'head_bias_hex':[hx(x) for x in head.bias.detach().cpu().tolist()]}


def centered_arbs(arb_row):
    mean=sum(arb_row,arb(0))/len(arb_row)
    centered=[u-mean for u in arb_row]
    text=[v.str(45) for v in centered]
    if not all(serialized_ball_contains_original(v,s) for v,s in zip(centered,text)):
        raise ArithmeticError('rigorous decimal serialization does not enclose centered input')
    return text


def run(*,radii=('1e-8','1e-5','1e-3','1e-2'),precision_bits=160,
        max_cases=64,fetch=False,checkpoint=None,
        lock_path=HERE/'source_lock.json',
        baseline_path=HERE/'results'/'lsv13_arb_validated_64_20261104.json'):
    source=json.loads(Path(lock_path).read_text())
    baseline=json.loads(Path(baseline_path).read_text())
    if baseline['schema']!='LSV-13/arb-validated-goal-minors-v1':
        raise ValueError('need original LSV-13 exact-rank roster')
    if type(max_cases)!=int or not 1<=max_cases<=64:
        raise ValueError('max_cases out of range')
    if type(precision_bits)!=int or precision_bits<96:
        raise ValueError('need >=96-bit outward interval arithmetic')
    weights_bytes,sha=locked_bytes(source,checkpoint,fetch)
    (inputs,targets,_),roster=prepare_roster(source)
    if sha!=baseline['checkpoint_sha256'] or roster!=baseline['roster_sha256']:
        raise ValueError('source-locked checkpoint/roster mismatch')
    if len(baseline['cases'])!=64:
        raise ValueError('incomplete source-locked baseline')
    torch.set_num_threads(1)
    raw=load_pretrained(source,weights_bytes).eval()
    with torch.inference_mode():
        z1=raw.reason_states(inputs)[1].detach().double()
    model=copy.deepcopy(raw).double().eval()
    ctx.prec=precision_bits
    net=CertifiedBlock(model.block)
    weights=exact_weights(model)
    cases=[]
    t0=time.perf_counter()
    for old in baseline['cases'][:max_cases]:
        i=len(cases)
        if old['case_index']!=i:
            raise ValueError('baseline order changed')
        cell=old['selected_goal_cell']
        if inputs[i,cell]!=0:
            raise ValueError('declared goal cell is not missing')
        pre=pre_final_ln2(net,z1[i])
        row=pre.tolist()[cell]
        case={'case_index':i,'selected_goal_cell':cell,
              'winner_digit':old['winner_digit'],
              'rival_digit':old['rival_digit'],
              'u_intervals':centered_arbs(row)}
        case['certificate']=certificate_for_case(case,weights,radii)
        cases.append(case)
        print('CASE',i,'R',[(r,case['certificate']['transverse_radii'][r]['top1_certified'])
                          for r in radii],flush=True)
    counts={r:{'top1':sum(c['certificate']['transverse_radii'][r]['top1_certified'] for c in cases),
               'top2':sum(c['certificate']['transverse_radii'][r]['top2_certified'] for c in cases)}
            for r in radii}
    receipt={'schema':'LSV-15/gauge-quotient-transverse-v1',
             'status':'exact_rational_conditional_on_trusted_arb_source_enclosures',
             'source_revision':source['revision'],
             'checkpoint_sha256':sha,'roster_sha256':roster,
             'precision_bits':precision_bits,'radii':list(radii),
             'intervention_bus':'per-token pre final LayerNorm2, at last (fourth) recurrent update',
             'intervention_region':'unbounded per-token common shifts plus bounded row-centered deviations',
             'weights':weights,'cases':cases,'counts':counts,
             'wall_seconds':round(time.perf_counter()-t0,3),
             'scope':'exact-real LayerNorm operation only; unrelated to arbitrary stage-one hidden-state damage or hardware rounding'}
    checked=replay_receipt(receipt,source,baseline,require_complete=max_cases==64)
    assert checked==counts
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--max-cases',type=int,default=64)
    p.add_argument('--precision',type=int,default=160)
    p.add_argument('--radii',default='1e-8,1e-5,1e-3,1e-2')
    p.add_argument('--out',type=Path,default=Path('runs/lsv15/gauge.json'))
    a=p.parse_args()
    radius=tuple(s.strip() for s in a.radii.split(','))
    rec=run(radii=radius,max_cases=a.max_cases,precision_bits=a.precision,
            checkpoint=a.checkpoint,fetch=a.fetch)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print('LSV15_RESULT',json.dumps({'cases':len(rec['cases']),'radii':rec['radii'],
                                    'counts':rec['counts'],'seconds':rec['wall_seconds']}))

if __name__=='__main__':main()
