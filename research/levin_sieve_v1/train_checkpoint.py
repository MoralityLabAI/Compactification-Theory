"""Train a tiny nonlinear recurrent state-repair policy; output a rational checkpoint.

CPU-only teaching-scale learned model, no pretrained TRM or LLM weights. Does not
read benchmark labels/seeds. Torch is required ONLY to reproduce training.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, random
from pathlib import Path

SEED = 20261104
TARGET_GAINS = (("0", "9/5"), ("1/5", "0"))  # task generator, external contract


def train(updates: int = 500, samples: int = 512, seed: int = SEED) -> dict:
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(seed + 101)
    ref_a, ref_b = 1.8, .2
    clean = torch.tensor([[rng.uniform(-.9, .9), rng.uniform(-.9, .9)]
                           for _ in range(samples)], dtype=torch.float64)
    forcing = torch.stack((clean[:, 0] - ref_a * clean[:, 1],
                           clean[:, 1] - ref_b * clean[:, 0]), 1)
    noise = torch.tensor([[rng.uniform(-1.5,1.5), rng.uniform(-1.5,1.5)]
                         for _ in range(samples)], dtype=torch.float64)
    initial = clean + noise
    # Certified stability is an architecture constraint; in a future experiment,
    # a competing unconstrained fit must be explicitly compared.
    theta = torch.nn.Parameter(torch.tensor([0.0, -0.5],dtype=torch.float64))
    optim = torch.optim.Adam([theta],lr=.045)
    for update in range(updates):
        a = 2.2 * torch.sigmoid(theta[0])
        b = 0.4 * torch.sigmoid(theta[1])
        x = initial
        losses = []
        for k in range(1,13):
            x0 = forcing[:,0]+a*x[:,1].clamp(-1,1)
            x1 = forcing[:,1]+b*x0.clamp(-1,1)
            x = torch.stack((x0,x1),-1)
            if k in (1,2,4,8,12):
                losses.append((x-clean).square().mean())
        loss=sum(w*v for w,v in zip((.05,.05,.1,.2,.6),losses))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    coeffs=(2.2*torch.sigmoid(theta[0])).item(), (0.4*torch.sigmoid(theta[1])).item()
    # Exact rational deployment representation: denominators 10,000. Exported
    # decimals are interpreted as exact fractions, not floating neural ops.
    q=(round(coeffs[0],4),round(coeffs[1],4))
    str_coeffs=(f"{q[0]:.4f}",f"{q[1]:.4f}")
    ck={"schema":"LSV-04/rational-checkpoint-v1",
        "class":"cpu_trained_two_node_nonlinear_recurrent_proxy",
        "input":"forcing b=(I-A_ref)x_star for |x_star|<=9/10",
        "update":"x_i <- forcing_i + weight_i*clip(x_j,-1,1), asynchronous order 0 then 1",
        "weights":{"a01":str_coeffs[0],"a10":str_coeffs[1]},
        "target_gains":[["0","9/5"],["1/5","0"]],
        "training":{"seed":seed,"samples":samples,"updates":updates,
                    "intermediate_cycles":[1,2,4,8,12],"initial_damage_bound":"3/2",
                    "torch_version":torch.__version__,"python":platform.python_version(),
                    "initial_parameterization":"a=2.2*sigmoid(theta[0]), b=0.4*sigmoid(theta[1])",
                    "optimizer":"Adam lr=0.045, full batch; no test labels seen"},
        "reported_training_loss":float(loss.detach()),
        "limitations":"toy CPU-trained proxy; rational update evaluated exactly; not pretrained TRM, neural language model, or floating-point execution certificate"}
    ck["weights_sha256"]=hashlib.sha256(json.dumps(ck["weights"],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return ck


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--updates',type=int,default=500)
    args=p.parse_args()
    ck=train(updates=args.updates)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(ck,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:ck[k] for k in ('weights','reported_training_loss','weights_sha256')}))

if __name__=='__main__':main()
