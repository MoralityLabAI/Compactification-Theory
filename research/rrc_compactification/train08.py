"""RRC-08 CPU fitting. Local exact teacher supervision; no test generation.

Optional PyTorch dependency for TRAINING only. Inference and CI replay use the
portable integer DSL interpreter. Eight scalar weights per trained arm.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from .operator08 import (PREFIX,SEED,roster,roster_hash,sample_features)
    from .rrc05 import log_derivative_coefficients
    from .rrc import euler_exponents
except ImportError:
    from operator08 import (PREFIX,SEED,roster,roster_hash,sample_features)
    from rrc05 import log_derivative_coefficients
    from rrc import euler_exponents

HERE=Path(__file__).resolve().parent
STEPS=1000
LR=0.06


def _fit(X,y,steps=STEPS,lr=LR):
    import torch
    torch.set_num_threads(1)
    X=torch.tensor(X,dtype=torch.float64)
    y=torch.tensor(y,dtype=torch.float64)
    weights=torch.nn.Parameter(torch.zeros(X.shape[1],dtype=torch.float64))
    optimizer=torch.optim.Adam([weights],lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss=((X@weights-y)**2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        resid=(X@weights-y)
        return {'weights':[float(q) for q in weights.tolist()],
                'mse':float((resid**2).mean()),
                'max_abs_residual':float(resid.abs().max()),
                'design_rank':int(torch.linalg.matrix_rank(X)),
                'condition_number':float(torch.linalg.cond(X))}


def ff_examples(train,degree=16):
    """Eight-parameter local-only model sees f[n], f[n-1], f[n-2]."""
    xb=[];yb=[];xa=[];ya=[]
    for c in train:
        f=c.prefix
        exact_a=euler_exponents(f)
        exact_b=log_derivative_coefficients(f)
        for n in range(1,degree+1):
            fb=(n*f[n],f[n],f[n-1],1)
            fa=(f[n],f[n-1], f[n-2] if n>1 else 0, 1)
            normb=max(1,*map(abs,fb));norma=max(1,*map(abs,fa))
            xb.append([float(v/normb) for v in fb]); yb.append(float(exact_b[n]/normb))
            xa.append([float(v/norma) for v in fa]); ya.append(float(exact_a[n]/norma))
    return xb,yb,xa,ya


def generate():
    import torch
    data=roster(include_test=False)
    train=data['train']
    recurrent=sample_features(train,train_degree=16)
    local=ff_examples(train,degree=16)
    rng=random.Random(SEED+1000)
    def fit_pair(dataset,kind,shuffled=False):
        xb,yb,xa,ya=dataset
        if shuffled:
            yb=yb.copy();ya=ya.copy()
            rng.shuffle(yb);rng.shuffle(ya)
        fit_b=_fit(xb,yb);fit_a=_fit(xa,ya)
        weights=fit_b['weights']+fit_a['weights']
        snapped=[int(round(v)) for v in weights]
        max_delta=max(abs(v-z) for v,z in zip(weights,snapped))
        # This is a fixed integer-code gate, specified independently of eval.
        accepted=(max_delta<=.125 and all(abs(v)<=2 for v in snapped))
        return {'schema':'rrc08.weights.v1','kind':kind,'seed':SEED,
                'training_examples':len(train),'degree_supervised':16,
                'training_rows':len(xb),'steps':STEPS,'learning_rate':LR,
                'optimizer':'torch.Adam.fp64.full_batch',
                'training_data_sha256':roster_hash(train),
                'operator08_sha256':sha256((HERE/'operator08.py').read_bytes()).hexdigest(),
                'train08_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
                'torch_version':torch.__version__,
                'b_fit':fit_b,'a_fit':fit_a,
                'decoded_coefficients':snapped if accepted else None,
                'snap_max_error':max_delta,'executable':accepted,
                'inference_teacher_access':False,
                'training_teacher_traces':True,
                'test_family_generation_in_train':False}
    return {'operator':fit_pair(recurrent,'compiled_recurrent_operator'),
            'local':fit_pair(local,'raw_local_feedforward'),
            'shuffled':fit_pair(recurrent,'label_shuffled_operator',True)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=HERE/'results')
    args=parser.parse_args()
    outputs=generate()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for kind,content in outputs.items():
        path=args.out_dir/f'rrc08_{kind}_weights.json'
        path.write_text(json.dumps(content,sort_keys=True,indent=2)+'\n')
        print(kind,'snap:',content['decoded_coefficients'],'mse_b',content['b_fit']['mse'],
              'mse_a',content['a_fit']['mse'],'->',path)


if __name__=='__main__':main()
