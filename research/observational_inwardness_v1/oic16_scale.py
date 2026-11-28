"""OIC-16 post-hoc phase-count scaling, with a fixed learned Fourier emission.

Uses exact finite phase/action data and independently replays the learned
belief from stored weights. The n>=9 sweep is exploratory and was NOT part of
the original held-out acceptance gate (n=7,8).
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from oic16_verify import load_receipt,report_panel,sigmoid

HERE=Path(__file__).resolve().parent
PATH=HERE/'results'/'oic16_scaling.json'
NS=(9,10,12,16,20,24,32)


def neighbor_gap(n:int,weights):
    _,b,c=weights['emission_coeff']
    exact=b*(1-math.cos(2*math.pi/n))+c*(1-math.cos(4*math.pi/n))
    bound=(2*math.pi**2*abs(b)+8*math.pi**2*abs(c))/n**2
    return exact,bound


def run():
    _,d=load_receipt()
    rows=[]
    for record in d['runs']:
        w=record['weights']
        for n in NS:
            entry=report_panel(n,'1'*n,w)
            coeff=w['emission_coeff']
            z0=coeff[0]+coeff[1]+coeff[2]
            gap,bound=neighbor_gap(n,w)
            rows.append({
                'augmentation':record['augment'],'training_split':record['split'],'seed':record['seed'],
                'n':n,'full_active_accuracy':round(entry['accuracy'],8),
                'full_active_brier':round(entry['brier'],8),
                'emission_pulse_at_marker':round(sigmoid(z0),8),
                'emission_pulse_neighbor':round(sigmoid(z0-gap),8),
                'emission_neighbor_logit_gap':round(gap,8),
                'neighbor_logit_gap_upper_bound':round(bound,8),
            })
    aggregate=[]
    for aug in ('clean','raw_fault_aug'):
        for split in ('full','heldout'):
            for n in NS:
                sel=[r for r in rows if r['augmentation']==aug and r['training_split']==split and r['n']==n]
                aggregate.append({
                    'augmentation':aug,'training_split':split,'n':n,
                    'mean_accuracy':round(sum(r['full_active_accuracy'] for r in sel)/len(sel),8),
                    'min_seed_accuracy':round(min(r['full_active_accuracy'] for r in sel),8),
                    'mean_brier':round(sum(r['full_active_brier'] for r in sel)/len(sel),8),
                    'mean_neighbor_emission':round(sum(r['emission_pulse_neighbor'] for r in sel)/len(sel),8),
                })
    return {'status':'posthoc_phase_count_scaling_cpu_reference',
            'pre_registered_ood_phase_counts':[7,8],
            'posthoc_scaling_phase_counts':list(NS),'runs':rows,'aggregates':aggregate,
            'theorem':'fixed_coefficient_Fourier_neighbor_logit_gap_is_O(n^-2)'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=PATH)
    a=parser.parse_args()
    observed=run()
    if a.check:
        expected=json.loads(PATH.read_text())
        if observed!=expected:raise SystemExit('OIC-16 scaling replay mismatch')
        print('OIC-16 independent large-n scaling replay passed')
    else:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(observed,indent=2,sort_keys=True)+'\n')
        print('wrote',a.out,'with',len(observed['runs']),'per-fit scaling panels')


if __name__=='__main__':main()
