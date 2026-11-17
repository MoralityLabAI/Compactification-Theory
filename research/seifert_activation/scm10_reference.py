"""SCM-10 full-shape fixed synthetic receipt, conditional CPU IEEE proof.

The exact-real source model has synthetic dyadic weights and a frozen dyadic
frequency table. Numeric centers and radii may differ harmlessly among BLAS
vendors, so --check validates source hashes, proof scopes and pre-registered
thresholds rather than demanding identical float64 values.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from fractions import Fraction

import torch
import torch.nn.functional as F
from scm09_reference import DeterministicRMP
from exact_input import rmp_input_enclosure
from ball_centered import verified_ball_tubes
from fast_centered import ieee_preflight

SCHEMA='scm10_full_rmp_shape_IEEE_source_ball_v1'
CASE_SPECS=(
    dict(id='tied_beta_010',tied=True,reinjection=False,beta=.1,cap=.01,expect_small=True),
    dict(id='tied_beta_025_negative',tied=True,reinjection=False,beta=.25,cap=1.,expect_small=False),
    dict(id='untied_reinjection_beta_025',tied=False,reinjection=True,beta=.25,cap=.005,expect_small=True),
)


def _hash_model(model):
    sha=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        t=value.detach().cpu().contiguous()
        sha.update(name.encode()+b'\0'+str(tuple(t.shape)).encode()+b'\0')
        sha.update(str(t.dtype).encode()+b'\0'+t.numpy().tobytes(order='C'))
    return sha.hexdigest()


@torch.no_grad()
def torch_shadow(model,source,*,reinjection:bool):
    """Independent float64 RMP graph; empirical check, never a certificate."""
    model=copy.deepcopy(model).double().eval()
    state=source.center.double().unsqueeze(0)
    inp=state.clone(); outputs=[]
    for index in model.module_indices:
        block=model.blocks[index]
        n1=block.norm1(state)
        qkv=block.qkv(n1)
        n,seq,d=qkv.shape
        heads=block.num_heads;hd=d//(3*heads)
        parts=qkv.reshape(n,seq,3,heads,hd)
        query=parts[:,:,0].transpose(1,2)
        key=parts[:,:,1].transpose(1,2)
        value=parts[:,:,2].transpose(1,2)
        scores=(query@key.transpose(-2,-1)) / (hd**.5)
        mask=torch.triu(torch.ones(seq,seq,dtype=torch.bool),diagonal=1)
        weights=torch.softmax(scores.masked_fill(mask,float('-inf')),dim=-1)
        vec=(weights@value).transpose(1,2).reshape(n,seq,d//3)
        att=block.attention_out(vec)
        mid=block.alpha*state+block.beta*att
        out=block.alpha*mid+block.beta*block.mlp(block.norm2(mid))
        state=out+inp if reinjection else out
        outputs.append(state[0].clone())
    return outputs


def one_case(spec):
    model=DeterministicRMP(d=128,seq=64,visits=8,tied=spec['tied'],
                           reinjection=spec['reinjection'],beta=spec['beta'])
    frequencies=torch.tensor([2.**(-k) for k in range(64)],dtype=torch.float32)
    source=rmp_input_enclosure(model,model.tokens,positional_frequency_table=frequencies)
    tubes=verified_ball_tubes(model,source,fixed_input_reinjection=spec['reinjection'])
    largest=[float(max(errors)) for errors in tubes.post_center_errors]
    shadow=torch_shadow(model,source,reinjection=spec['reinjection'])
    gap=[float(torch.linalg.vector_norm(ref-ctr,dim=-1).max())
         for ref,ctr in zip(shadow,tubes.post_centers)]
    within=[g <= bound for g,bound in zip(gap,largest)]
    if not all(within):
        raise AssertionError('independent float64 shadow exceeds validated source ball')
    ok=largest[-1]<spec['cap']
    if ok != spec['expect_small']:
        raise AssertionError(f'synthetic acceptance changed unexpectedly: {spec["id"]}')
    return {'case':spec['id'],'model_sha256':_hash_model(model),
            'frequency_sha256':source.frequency_sha256,
            'model_shape':{'sequence':64,'hidden':128,'heads':4,'visits':8},
            'tied':spec['tied'],'reinjection':spec['reinjection'],'beta':spec['beta'],
            'radius_cap':spec['cap'],'passed_radius_cap':ok,
            'per_visit_max_clean_radius':largest,
            'per_visit_empirical_torch64_shadow_distance':gap,
            'all_empirical_torch64_shadow_within_balls':all(within),
            'numpy_arithmetic_contract':'IEEE_binary64_BLAS_dot_standard_forward_error_assumed',
            'ideal_real_source_bound_conditional':True,
            'trained_rmp_checkpoint_present':False,
            'floating_execution_certified':False,
            'observed_link_certified':False}


def receipt():
    torch.set_num_threads(1)
    preflight=ieee_preflight()
    root=Path(__file__).resolve().parent
    return {'schema':SCHEMA,'date':'2026-11-14',
            'scope':'synthetic_fullshape_exact_real_under_declared_IEEE_BLAS_contract',
            'backend_code_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                   for name in ('fast_centered.py','ball_centered.py')},
            'arithmetic_preflight':preflight,
            'cases':[one_case(spec) for spec in CASE_SPECS]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    args=p.parse_args(argv)
    result=receipt()
    if args.check:
        previous=json.loads(args.out.read_text(encoding='utf-8'))
        if previous['schema'] != result['schema'] or previous['backend_code_sha256']!=result['backend_code_sha256']:
            raise SystemExit('SCM-10 reference provenance or backend mismatch')
        for x,y in zip(result['cases'],previous['cases']):
            if (x['case']!=y['case'] or x['model_sha256']!=y['model_sha256'] or
                x['frequency_sha256']!=y['frequency_sha256'] or
                x['passed_radius_cap']!=y['passed_radius_cap']):
                raise SystemExit('SCM-10 reference mismatch')
        print('SCM10 SOURCE/THRESHOLD REFERENCE MATCH')
    else:
        if args.out.exists():
            raise FileExistsError('SCM-10 source reference immutable; choose another path')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        txt=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
        args.out.write_text(txt,encoding='utf-8')
        print('SCM10 SOURCE REFERENCE',hashlib.sha256(txt.encode()).hexdigest())
        print(json.dumps([{k:x[k] for k in ('case','passed_radius_cap','per_visit_max_clean_radius')}
                         for x in result['cases']],indent=2))

if __name__=='__main__':main()
