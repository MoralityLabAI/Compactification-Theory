"""LSV-09: goal-projected derivative diagnostic for frozen pretrained TRM.

This is *observational first-order sensitivity*, explicitly NOT a finite-radius
robustness certificate. The exact rational affine theorem is tested separately.
"""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
import torch

from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
from run_predictive import single_cell_mask

HERE=Path(__file__).resolve().parent


def quantiles(x):
    x=x.detach().double().flatten()
    def v(q):
        return round(float(torch.quantile(x,torch.tensor(q,dtype=torch.float64))),10)
    return {'min':v(0),'p10':v(.1),'median':v(.5),'p90':v(.9),'max':v(1.)}


def evaluate_gradients(model,z1,inputs,answers):
    t=time.perf_counter()
    if z1.dtype!=torch.float64:
        raise ValueError('exact stored FP32 numbers, converted to FP64, required')
    z1=z1.detach().clone().requires_grad_(True)
    mask=single_cell_mask(inputs)
    idx=mask.any(-1).to(torch.int64).argmax(-1)
    arange=torch.arange(len(inputs))
    z=z1
    for _ in range(3):z=model.block(z)
    logits=model.head(z)[arange,idx]
    top2=logits.topk(2,dim=-1)
    top=top2.indices[:,0].detach()
    second=top2.indices[:,1].detach()
    margin=top2.values[:,0]-top2.values[:,1]
    if not bool((margin>0).all()):raise ValueError('observed tie; cannot define unique goal margin')
    grad=torch.autograd.grad(margin.sum(),z1)[0].detach()
    grad_full=grad.abs().sum((1,2))
    grad_patch=(grad.abs()*mask).sum((1,2))
    if not bool((grad_full>0).all()):
        raise ValueError('unexpected zero derivative on fixed roster')
    top_clean=top.detach()
    observed=[]
    for mode in ('all_coordinates','first_missing_cell'):
        use_mask=None if mode=='all_coordinates' else mask
        active_grad=grad_full if use_mask is None else grad_patch
        r1=margin.detach()/active_grad.clamp(min=1e-100)
        for eps in (0.00001,0.02,0.10,0.50):
            direction=-grad.sign()*eps
            if use_mask is not None:direction=direction*use_mask
            with torch.no_grad():
                y=z1.detach()+direction
                for _ in range(3):y=model.block(y)
                values=model.head(y)[arange,idx]
                after=values.argmax(-1)
                delta=(values.gather(-1,top[:,None]).squeeze(-1)-
                       values.gather(-1,second[:,None]).squeeze(-1))
            observed.append({'mode':mode,'epsilon':eps,
                'first_order_predicted_margin_preserved_cases':int((margin.detach()>eps*active_grad).sum()),
                'observed_original_top_class_preserved':int((after==top_clean).sum()),
                'observed_selected_pair_margin_positive':int((delta>0).sum()),
                'n':len(inputs),
                'caution':'first-order tangent estimate is NOT a finite-radius certificate'})
        observed[-1]['first_order_radius_quantiles']=quantiles(r1)
    return {
       'selected_cells':'first missing Sudoku cell per puzzle',
       'prediction_first_missing_correct':int((top==answers[arange,idx]).sum()),
       'n':len(inputs),
       'output_margin_quantiles':quantiles(margin.detach()),
       'gradient_full_L1_quantiles':quantiles(grad_full),
       'gradient_patch_L1_quantiles':quantiles(grad_patch),
       'ratio_full_to_patch_L1_quantiles':quantiles(grad_full/grad_patch.clamp(min=1e-100)),
       'directional_attack':observed,
       'cpu_elapsed_seconds':round(time.perf_counter()-t,3),
    }


def run(lock_path,checkpoint_path=None,allow_fetch=False):
    lock=json.loads(Path(lock_path).read_text())
    if lock['schema']!='LSV-05/pretrained-trm-source-lock-v1':
        raise ValueError('unexpected model lock')
    data,sha=locked_bytes(lock,checkpoint_path,allow_fetch)
    raw=load_pretrained(lock,data)
    (inputs,answers,_),roster=prepare_roster(lock)
    torch.set_num_threads(1)
    with torch.no_grad():
        z1=raw.reason_states(inputs)[1].detach().double()
    model=copy.deepcopy(raw).double().eval()
    measures=evaluate_gradients(model,z1,inputs,answers)
    return {'schema':'LSV-09/goal-projected-autograd-diagnostics-v1',
            'status':'EXPLORATORY_TANGENT_DIAGNOSTICS_NOT_PROOFS',
            'model_sha256':sha,'roster_sha256':roster,'source_revision':lock['revision'],
            'formal_guarantee':'NONE on nonlinear pretrained checkpoint; local derivative does not bound variation across the damage region',
            'measurements':measures}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lock',type=Path,default=HERE/'source_lock.json')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    rec=run(a.lock,a.checkpoint,a.fetch)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(rec,sort_keys=True,indent=2)+'\n')
    m=rec['measurements']
    print('LSV-09 SHA256',rec['model_sha256'],'ROSTER',rec['roster_sha256'])
    print('MARGIN',json.dumps(m['output_margin_quantiles']))
    print('GRAD-L1-ALL',json.dumps(m['gradient_full_L1_quantiles']))
    print('GRAD-L1-PATCH',json.dumps(m['gradient_patch_L1_quantiles']))
    for r in m['directional_attack']:
        print('ROW',r['mode'],r['epsilon'],'tangent',r['first_order_predicted_margin_preserved_cases'],
              'actually_same',r['observed_original_top_class_preserved'])
    print('elapsed',m['cpu_elapsed_seconds'])

if __name__=='__main__':main()
