"""Frozen-roster LSV-17 experiment; no training or label-dependent selection."""
from pathlib import Path
import argparse,hashlib,json,sys,time,platform
from fractions import Fraction as Q
import torch
from flint import ctx
from arb_engine import TaylorNet,evaluate_case
HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'pretrained_trm'
sys.path.insert(0,str(PRE))
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster,fingerprint


def run(args):
    ctx.prec=args.precision
    torch.set_num_threads(1)
    lock=json.loads((PRE/'source_lock.json').read_text())
    data,sha=locked_bytes(lock,args.checkpoint,args.fetch)
    model=load_pretrained(lock,data).eval()
    (inputs,targets,_),roster=prepare_roster(lock)
    if not 1<=args.cases<=len(inputs):raise ValueError('case count outside frozen roster')
    radii=[Q(x) for x in args.radii.split(',')]
    if any(r<=0 or r>1 for r in radii):raise ValueError('radius outside (0,1]')
    with torch.no_grad():anchors=model.reason_states(inputs)[1].detach()
    t0=time.perf_counter();net=TaylorNet(model)
    print('SPECTRAL',json.dumps(net.spectral),flush=True)
    records=[]
    for i in range(args.cases):
        cell=int(inputs[i].eq(0).long().argmax())
        rec=evaluate_case(net,anchors[i],cell,radii)
        rec['case_index']=i
        rec['anchor_sha256']=fingerprint(anchors[i].tolist())
        rec['clean_prediction_correct']=rec['winner']==int(targets[i,cell])
        records.append(rec)
        print('CASE',i,[(x['mode'],x['radius'],x['certified']) for x in rec['regions']],flush=True)
    summaries=[]
    for mode in ['full','goal_token']:
        for r in radii:
            selected=[next(t for t in x['regions'] if t['mode']==mode and Q(t['radius'])==r) for x in records]
            summaries.append({'mode':mode,'radius':str(r),'cases':len(records),
                'certified':sum(t['certified'] for t in selected),
                'certified_and_correct':sum(t['certified'] and x['clean_prediction_correct'] for t,x in zip(selected,records)),
                'epsilon_floor_used':sum(t['epsilon_floor_used'] for t in selected)})
    out={'schema':'LSV-17/signed-adjoint-taylor-v1','arithmetic':'Arb exact-real reference, hardware rounding excluded',
        'checkpoint_sha256':sha,'roster_sha256':roster,'precision_bits':args.precision,
        'torch_version':torch.__version__,'python_version':platform.python_version(),
        'anchor_contract':'all 64 stage-one tensors computed in one CPU FP32 batch, entries interpreted exactly',
        'n_cases':len(records),'goal':'first missing cell per case; all 9 competing digits checked',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        'spectral_bounds':net.spectral,'elapsed_seconds':round(time.perf_counter()-t0,3),
        'summaries':summaries,'cases':records}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print('SUMMARY',json.dumps(summaries),flush=True)
    print('SECONDS',out['elapsed_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--cases',type=int,default=64)
    p.add_argument('--precision',type=int,default=160)
    p.add_argument('--radii',default='1e-10,1e-8,1e-7,1e-6,1e-5,1e-4')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.precision<80:raise ValueError('precision >=80 required')
    run(a)
