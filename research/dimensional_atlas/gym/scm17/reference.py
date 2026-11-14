"""Executed small mock-only reference for SCM-17. No trained RMP checkpoint."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from .rmp_fiber import SAEVisitFiber, calibrate_decision_frame, causal_probe, run_with_adapter
from .campaign import score,train_adapter,state_digest

def run():
    # Same deterministic visit-hook interface, with deliberately independent
    # randomly initialized toy weights and a fake dictionary.
    from .tests.test_scm17 import FakeAnalysis,FakeDictionary
    torch.set_num_threads(1)
    seed=20261114
    torch.manual_seed(seed)
    results={}
    for name,reinject in [('tied_mock',False),('ffu_injection_mock',True)]:
        model=FakeAnalysis(seed=seed+int(reinject),inject=reinject)
        g=torch.Generator().manual_seed(seed+10)
        data=torch.randint(0,16,(64,5),generator=g)
        train=data[:40];test=data[40:]
        train_label=(train[:,0]+2*train[:,1]+train[:,3])%16
        test_label=(test[:,0]+2*test[:,1]+test[:,3])%16
        frame=calibrate_decision_frame(model,train[:2],contrasts=1,max_rank=2)
        adapter=SAEVisitFiber(FakeDictionary(),frame,rank=3,visits=4)
        base_digest=state_digest(model.model)
        before=score(model,test,test_label)
        train_receipt=train_adapter(model,adapter,train,train_label,
            steps=24,seed=seed+3,batch_size=8,learning_rate=.003,device='cpu')
        after=score(model,test,test_label,adapter)
        shuffle=score(model,test,test_label,adapter,visit_order=(3,2,1,0))
        zero=score(model,test,test_label,adapter,zero_codes=True)
        probe=causal_probe(model,train[:1],frame,visit_count=4,epsilon=1e-3)
        if base_digest!=state_digest(model.model):raise AssertionError('base changed')
        results[name]={'frame_rank':frame.q.shape[1],
          'retained_gradient_energy':frame.captured_gradient_energy,
          'train':train_receipt,'baseline':before,'joint':after,'shuffled':shuffle,
          'zero_sae':zero,'fiber_probe':probe,'source_frozen':True}
    return {'schema':'scm17/mock-bridge/v1',
            'source_type':'randomly_initialized_mock_visit_hook_not_trained_RMP',
            'trained_RMP_evidence':False,
            'n_test_per_model':24,'results':results}

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    a=p.parse_args(argv)
    payload=json.dumps(run(),indent=2,sort_keys=True,allow_nan=False)+'\n'
    if a.check:
        if not a.out.exists():raise SystemExit('SCM-17 mock reference missing')
        expected=json.loads(a.out.read_text())
        actual=json.loads(payload)
        def compare(left,right,path='root'):
            if isinstance(left,dict):
                if set(left)!=set(right):raise ValueError('reference keys changed: '+path)
                for k in left:
                    if k.endswith('_sha256'):continue  # hashes of predicted float scores are machine-sensitive
                    compare(left[k],right[k],path+'/'+k)
            elif isinstance(left,list):
                if len(left)!=len(right):raise ValueError('reference length changed: '+path)
                for i,(x,y) in enumerate(zip(left,right)):compare(x,y,path+'/'+str(i))
            elif isinstance(left,float):
                if abs(left-right)>max(.01,abs(left)*.003):
                    raise ValueError('reference float changed: '+path)
            elif left!=right:raise ValueError('reference value changed: '+path)
        compare(expected,actual)
        print('SCM-17 mock reference numerically verified')
    else:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open('x',encoding='utf-8') as file:file.write(payload)
        print('wrote',a.out)
if __name__=='__main__':main()
