"""Replay frozen OIC-16 Fourier baseline weights under OIC-18 locked OOD n.

No new optimizer steps. Base weights came from the completed OIC-16 500x128
training and 5 seeded full/heldout, clean/fault-augmented splits.
"""
from __future__ import annotations
import argparse,base64,gzip,json,time
from pathlib import Path
import torch
from oic18_train import panel,action,EVAL_NS
from oic18_model import PhaseEmissionBelief

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results/oic16_full.json.gz.b64'

def run(out:Path):
    torch.set_num_threads(1)
    data=json.loads(gzip.decompress(base64.b64decode(ARCHIVE.read_text().strip())))
    start=time.perf_counter()
    records=[]
    for old in data['runs']:
        model=PhaseEmissionBelief('fourier')
        model.load_state_dict({k:torch.tensor(v) for k,v in old['weights'].items()})
        model.eval()
        panels={}
        for n in EVAL_NS:
            panels[str(n)]={name:panel(model,n,action(n,name),report_phase=name=='full_active')
                            for name in ('full_active','passive','latch','alternate')}
        records.append({'architecture':'fourier','split':old['split'],'augmentation':old['augment'],
                        'seed':old['seed'],'panels':panels,'weights':old['weights'],
                        'resource_ledger':model.resource_ledger(64),
                        'original_training':old['training']})
        print('BASE',old['augment'],old['split'],old['seed'],
              [round(panels[str(n)]['full_active']['accuracy'],3) for n in (7,8,12,16,24,32,64)],flush=True)
    output={'status':'frozen_OIC16_500_step_weight_replay_OIC18_registered_phase_counts',
            'records':records,'wall_seconds':round(time.perf_counter()-start,3)}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=1,sort_keys=True)+'\n')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.out)
