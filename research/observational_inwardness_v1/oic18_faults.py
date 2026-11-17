"""OIC-18 exact one-raw-sample substitution/erasure sweep for learned filters.

At each even n, use n active probes followed by 2 passive transitions, and
apply a single fault at every raw sample position, including initial and final.
All n phases of both families are tested; labels are balanced by construction.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, torch
from oic15_pilot import balanced_panel
from oic18_model import PhaseEmissionBelief

EVAL_NS=(4,6,8,12,16,24,32,64)


def fault_panel(model,n:int,kind:str,max_episodes=4096):
    assert kind in ('flip','erase')
    word=(1,)*n+(0,)*2
    fam,_,u,clean=balanced_panel(n,word)
    total_positions=clean.shape[1]
    B=len(fam)
    output=[]
    per_phase_correct=np.zeros(B,dtype='int64')
    per_phase_pred_mean=np.zeros(B,dtype='float64')
    worst_position_accuracy=1.0
    steps=max(1,max_episodes//B)
    with torch.no_grad():
        for first in range(0,total_positions,steps):
            subset=list(range(first,min(total_positions,first+steps)))
            npos=len(subset)
            raw=clean.repeat(npos,1)
            interventions=u.repeat(npos,1)
            for k,pos in enumerate(subset):
                if kind=='flip':raw[k*B:(k+1)*B,pos]^=1
                else:raw[k*B:(k+1)*B,pos]=-1
            score=torch.sigmoid(model(n,interventions,raw)[:,-1]).cpu().numpy().reshape(npos,B)
            lab=fam.numpy().reshape(1,B)
            right=((score>0.5)==lab)
            pos_accuracies=right.mean(1)
            for pos,a in zip(subset,pos_accuracies):
                output.append({'position':pos,'accuracy':round(float(a),8)})
            worst_position_accuracy=min(worst_position_accuracy,float(np.min(pos_accuracies)))
            per_phase_correct+=right.sum(0).astype('int64')
            per_phase_pred_mean+=score.sum(0)
    total=B*total_positions
    return {
        'n':n,'type':kind,'horizon':len(word),'fault_positions':total_positions,
        'all_episodes':total,'correct_episodes':int(per_phase_correct.sum()),
        'accuracy':round(float(per_phase_correct.sum()/total),8),
        'worst_fault_position_accuracy':round(float(worst_position_accuracy),8),
        'worst_family_phase_accuracy':round(float(per_phase_correct.min()/total_positions),8),
        'per_family_phase_correct_counts':per_phase_correct.tolist(),
        'per_family_phase_probability_sum':[round(float(v),8) for v in per_phase_pred_mean],
        'position_accuracies':output,
    }


def run_file(source:Path,out:Path,n_values=EVAL_NS,max_episodes=4096):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data=json.loads(source.read_text())
    started=time.perf_counter()
    results=[]
    for r in data['runs']:
        model=PhaseEmissionBelief(r['architecture'])
        model.load_state_dict({k:torch.tensor(v) for k,v in r['weights'].items()})
        model.eval()
        item={'architecture':r['architecture'],'split':r['split'],
              'augmentation':r['augmentation'],'seed':r['seed'],'panels':{}}
        for n in n_values:
            item['panels'][str(n)]={kind:fault_panel(model,n,kind,max_episodes) for kind in ('flip','erase')}
        results.append(item)
        print(f"FAULT arch={r['architecture']} aug={r['augmentation']} split={r['split']} seed={r['seed']}"
              +str({n:[item['panels'][str(n)][kind]['accuracy'] for kind in ('flip','erase')] for n in n_values}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'status':'executed_full_raw_sensor_single_fault_sweep',
        'source_sweep_file':source.name,'phase_counts':list(n_values),'runs':results,
        'wall_seconds':round(time.perf_counter()-started,4)},indent=1,sort_keys=True)+'\n')
    print('OIC18 FAULT DONE',len(results),'fits',round(time.perf_counter()-started,1),'seconds')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,nargs='+',default=list(EVAL_NS))
    p.add_argument('--max-episodes',type=int,default=4096)
    a=p.parse_args();run_file(a.source,a.out,tuple(a.n),a.max_episodes)
