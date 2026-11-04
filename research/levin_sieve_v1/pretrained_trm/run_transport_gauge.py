"""LSV-16 frozen-checkpoint experiment: bridge ambient damage to LSV-15 margins."""
from __future__ import annotations
import argparse
import copy
from fractions import Fraction as Q
import gzip
import json
from pathlib import Path
import sys
import time

from flint import ctx
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
from transport_gauge import CertifiedTransport,goal_cell_radius,upper_decimal,as_dyadic
from gauge_quotient import Bound


def run(*,radii=('1e-8',),max_cases=64,precision_bits=160,fetch=False,
        checkpoint=None,lock_path=HERE/'source_lock.json',
        old_gauge=HERE/'results/lsv15_gauge_64_20261104.json.gz',
        thresholds=HERE/'results/lsv15_exact_thresholds_64_20261104.json'):
    if type(max_cases)!=int or not 1<=max_cases<=64:
        raise ValueError('1..64 cases required')
    if type(precision_bits)!=int or precision_bits<128:
        raise ValueError('at least 128-bit ball precision required')
    if not radii or len(set(radii))!=len(radii) or any(Q(x)<=0 for x in radii):
        raise ValueError('positive unique radii required')
    lock=json.loads(Path(lock_path).read_text())
    with gzip.open(old_gauge,'rt') as f:
        gauge=json.load(f)
    limits=json.loads(Path(thresholds).read_text())
    if (gauge['checkpoint_sha256']!=lock['checkpoint_sha256'] or
        limits['checkpoint_sha256']!=lock['checkpoint_sha256'] or
        limits['roster_sha256']!=gauge['roster_sha256'] or
        len(gauge['cases'])!=64 or len(limits['case_thresholds'])!=64):
        raise ValueError('LSV15 frozen source records disagree')
    checkpoint_bytes,sha=locked_bytes(lock,checkpoint,fetch)
    (inputs,targets,_),roster=prepare_roster(lock)
    if sha!=lock['checkpoint_sha256'] or roster!=gauge['roster_sha256']:
        raise ValueError('checkpoint or task roster mismatch')
    torch.set_num_threads(1)
    raw=load_pretrained(lock,checkpoint_bytes).eval()
    with torch.inference_mode():
        z1=raw.reason_states(inputs)[1].detach().double()
    model=copy.deepcopy(raw).double().eval()
    ctx.prec=precision_bits
    net=CertifiedTransport(model)
    all_records=[]
    counts={r:{'certified_winner':0,'certified_top2':0,'could_not_bound':0} for r in radii}
    t0=time.perf_counter()
    for i in range(max_cases):
        ref=gauge['cases'][i]
        old=limits['case_thresholds'][i]
        if ref['case_index']!=i or old['case_index']!=i:
            raise ValueError('case index mismatch')
        cell=ref['selected_goal_cell']
        if inputs[i,cell].item()!=0:
            raise ValueError('selected goal cell must be missing in puzzle')
        top=Q(old['winner_threshold_strict'])
        top2=Q(old['top_two_threshold_strict'])
        record={'case_index':i,'goal_cell':cell,'ambient_radii':{}}
        for r in radii:
            try:
                pre=net.from_stage_one(z1[i],r)
                row=pre.center.tolist()[cell]
                mean=sum(row)/48
                clean_centered=[z-mean for z in row]
                # The FP32 stage-one state can vary slightly across CPU runtimes.
                # Enclose the clean new/reference difference instead of requiring
                # their extremely narrow Arb intervals to overlap.
                anchor_drift=Q(0)
                for a,b in zip(clean_centered,ref['u_intervals']):
                    oldb=Bound.from_arb(b)
                    newlo,newhi=as_dyadic(a.lower()),as_dyadic(a.upper())
                    anchor_drift=max(anchor_drift,
                        abs(newlo-oldb.hi),abs(newhi-oldb.lo))
                raw_radius=goal_cell_radius(pre,cell)
                scale=10**28
                ceil_q=lambda q: Q(-(-q.numerator*scale//q.denominator),scale)
                transport_upper=ceil_q(raw_radius)
                drift_upper=ceil_q(anchor_drift)
                end_radius=transport_upper+drift_upper
                winner=end_radius<top
                both=end_radius<top2
                record['ambient_radii'][r]={
                    'verified_final_transverse_radius_upper':str(end_radius),
                    'transport_component_upper':str(transport_upper),
                    'reference_anchor_drift_upper':str(drift_upper),
                    'winner_certificate':winner,'top_two_certificate':both,
                    'LSV15_winner_threshold':str(top),
                    'LSV15_top2_threshold':str(top2),
                }
                counts[r]['certified_winner']+=int(winner)
                counts[r]['certified_top2']+=int(both)
            except ValueError as exc:
                record['ambient_radii'][r]={'verified_final_transverse_radius_upper':None,
                    'winner_certificate':False,'top_two_certificate':False,
                    'reason':str(exc)}
                counts[r]['could_not_bound']+=1
        all_records.append(record)
        print('LSV16_CASE',i,{r:record['ambient_radii'][r].get('verified_final_transverse_radius_upper')
                                   for r in radii},flush=True)
    return {'schema':'LSV-16/ambient-to-gauge-certified-transport-v2',
            'status':'rigorous_arb_transport_plus_verified_archived_center_drift_and_rational_margin_check',
            'source_revision':lock['revision'],
            'checkpoint_sha256':sha,'roster_sha256':roster,
            'stage1_input_contract':'exact binary32 forward state from this CPU run embedded in reals; archived LSV15 anchor drift bounded separately',
            'intervention_geometry':'independent ambient l-infinity perturbation on all 3888 stage1 coordinates',
            'model_arithmetic':'exact-real erf-GELU and layernorm with exact binary checkpoint weights; Arb outward enclosures',
            'radii':list(radii),'precision_bits':precision_bits,
            'cases':all_records,'counts':counts,'wall_seconds':round(time.perf_counter()-t0,3),
            'limits':'No machine-code FP32 or BF16 rounding guarantee; finite region and model/source locks apply.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--radii',default='1e-8,1e-10')
    p.add_argument('--max-cases',type=int,default=4)
    p.add_argument('--precision',type=int,default=160)
    p.add_argument('--out',type=Path,default=Path('runs/lsv16/ambient.json'))
    args=p.parse_args()
    rec=run(radii=tuple(x.strip() for x in args.radii.split(',')),
            max_cases=args.max_cases,precision_bits=args.precision,
            fetch=args.fetch,checkpoint=args.checkpoint)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print('LSV16_SUMMARY',json.dumps({'cases':len(rec['cases']),'counts':rec['counts'],
                                      'wall_seconds':rec['wall_seconds']},sort_keys=True))

if __name__=='__main__':main()
