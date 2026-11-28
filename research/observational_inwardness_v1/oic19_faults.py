"""OIC-19 complete single raw-state substitution/erasure checks, checkpointed.

The exact cyclic-codebook decoder is the reference ceiling; these fault tests
probe learned stochastic filters and make no uniform error-correction claim.
"""
from __future__ import annotations
import argparse,base64,gzip,json,time
from pathlib import Path
import torch
from oic18_model import PhaseEmissionBelief
from oic18_faults import fault_panel
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'results/oic19_full.json.gz.b64'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',nargs='+',default=['predictive'])
    ap.add_argument('--split',nargs='+',default=['full','heldout'])
    ap.add_argument('--seed',type=int,nargs='+',default=list(range(5)))
    ap.add_argument('--n',type=int,nargs='+',default=[8,24,64])
    ap.add_argument('--out',type=Path,default=HERE/'results/oic19_faults.json')
    args=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=json.loads(gzip.decompress(base64.b64decode(SOURCE.read_text().strip())))
    wanted={(v,s,z) for v in args.variant for s in args.split for z in args.seed}
    if args.out.exists():
        results=json.loads(args.out.read_text())
        if results['phase_counts']!=args.n:raise ValueError('resume phase counts mismatch')
    else:results={'status':'finite_learned_stochastic_fault_sweep','phase_counts':args.n,'runs':[]}
    seen={(r['variant'],r['split'],r['seed']) for r in results['runs']}
    started=time.perf_counter()
    for record in source['runs']:
        key=record['variant'],record['split'],record['seed']
        if key not in wanted or key in seen:continue
        model=PhaseEmissionBelief('von_mises')
        model.load_state_dict({k:torch.tensor(v) for k,v in record['weights'].items()})
        model.eval();panels={}
        for n in args.n:
            panels[str(n)]={kind:fault_panel(model,n,kind,max_episodes=8192) for kind in ('flip','erase')}
        results['runs'].append({'variant':key[0],'split':key[1],'seed':key[2],'panels':panels})
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(results,indent=1,sort_keys=True)+'\n')
        print('OIC19FAULT',key,{n:[panels[str(n)][k]['accuracy'] for k in ('flip','erase')] for n in args.n},
              'elapsed',round(time.perf_counter()-started,2),flush=True)
    print('OIC19FAULT CHECKPOINT',len(results['runs']),'completed fits',round(time.perf_counter()-started,2),flush=True)

if __name__=='__main__':main()
