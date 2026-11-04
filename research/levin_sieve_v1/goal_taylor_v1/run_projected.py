"""LSV-17B full roster, projected curvature plus validated attack witnesses."""
from pathlib import Path
from fractions import Fraction as Q
import argparse,hashlib,json,sys,time
import torch
from flint import ctx
from arb_engine import TaylorNet
from projected import evaluate,signed_attack
HERE=Path(__file__).resolve().parent;PRE=HERE.parent/'pretrained_trm'
sys.path.insert(0,str(PRE))
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster,fingerprint


def run(a):
    if a.precision<80 or not 1<=a.cases<=64:raise ValueError('invalid experiment limits')
    ctx.prec=a.precision;torch.set_num_threads(1)
    lock=json.loads((PRE/'source_lock.json').read_text())
    data,sha=locked_bytes(lock,a.checkpoint,a.fetch)
    model=load_pretrained(lock,data).eval()
    (inputs,targets,_),roster=prepare_roster(lock)
    radii=[Q(x) for x in a.radii.split(',')]
    if any(not 0<r<=1 for r in radii):raise ValueError('invalid radius')
    with torch.no_grad():stages=model.reason_states(inputs)[1].detach()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'stage_one':stages},a.out.with_name('anchors.pt'))
    a.out.with_name('checkpoint.pt').write_bytes(data)
    start=time.perf_counter();net=TaylorNet(model);records=[]
    for i in range(a.cases):
        cell=int(inputs[i].eq(0).long().argmax())
        rec,prepared=evaluate(net,stages[i],cell,radii)
        rec.update(case_index=i,anchor_sha256=fingerprint(stages[i].tolist()),
            clean_prediction_correct=rec['winner']==int(targets[i,cell]))
        if a.attacks:rec['attack']=signed_attack(net,stages[i],cell,prepared)
        records.append(rec)
        print('CASE',i,[(t['mode'],t['radius'],t['certified']) for t in rec['regions']],
              'ATTACK',rec.get('attack',{}).get('radius'),flush=True)
    summary=[]
    for mode in ('full','goal_token'):
        for r in radii:
            rows=[next(t for t in c['regions'] if t['mode']==mode and Q(t['radius'])==r) for c in records]
            summary.append({'mode':mode,'radius':str(r),'cases':len(rows),
                'certified':sum(t['certified'] for t in rows),
                'certified_and_correct':sum(t['certified'] and c['clean_prediction_correct'] for t,c in zip(rows,records))})
    result={'schema':'LSV-17B/projected-curvature-v1','checkpoint_sha256':sha,'roster_sha256':roster,
        'n_cases':len(records),'precision_bits':ctx.prec,'torch_version':torch.__version__,
        'anchor_contract':'same-batch exact binary stage-one anchors; reanchoring requires explicit error budget',
        'spectral_bounds':net.spectral,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        'summaries':summary,'verified_attacks':sum(c.get('attack',{}).get('verified',False) for c in records),
        'elapsed_seconds':round(time.perf_counter()-start,3),'cases':records}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)
    print('ATTACKS',result['verified_attacks'],'SECONDS',result['elapsed_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fetch',action='store_true');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--cases',type=int,default=64);p.add_argument('--precision',type=int,default=160)
    p.add_argument('--radii',default='1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4')
    p.add_argument('--attacks',action='store_true');p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
