"""Exhaustive finite semantic gates and exact register-threaded comparator."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)
AND, OR, XOR, EQ, LT = 8, 14, 6, 9, 2

class LoopedGate(nn.Module):
    def __init__(self, width: int = 32, loops: int = 2):
        super().__init__()
        self.loops = loops
        self.role = nn.Embedding(16, width)
        self.bit = nn.Embedding(2, width)
        self.pos = nn.Parameter(torch.randn(4, width) * .02)
        self.block = nn.TransformerEncoderLayer(
            width, 2, 2 * width, dropout=0., activation="gelu",
            batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor, mask_role: bool = False,
                loops: int | None = None):
        r, a, b = x.unbind(-1)
        rr = self.role(r)
        if mask_role:
            rr = torch.zeros_like(rr)
        h = torch.stack([rr, self.bit(a), self.bit(b), torch.zeros_like(rr)], 1)
        h = h + self.pos
        for _ in range(self.loops if loops is None else loops):
            h = self.block(h)
        return self.head(self.norm(h[:, -1]))

def rows():
    x = torch.tensor([(r, a, b) for r in range(16)
                      for a in range(2) for b in range(2)])
    y = torch.tensor([(r >> (2*a+b)) & 1 for r,a,b in x.tolist()])
    return x, y

class GateBank:
    """Hard finite outputs materialized from a checkpoint; no semantic fallback."""
    def __init__(self, table: np.ndarray, model: LoopedGate | None = None,
                 direct: bool = False):
        self.table = np.asarray(table, dtype=np.int64).reshape(16,2,2)
        self.model = model
        self.direct = direct
        self.calls = 0
        self.elements = 0
        self.role_override = None

    @classmethod
    def load(cls, path: str | Path, direct: bool = False, mask_role: bool = False):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = LoopedGate()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            table = model(rows()[0], mask_role=mask_role).argmax(-1).numpy()
        return cls(table, model, direct)

    def gate(self, role: int, a, b):
        a, b = np.broadcast_arrays(np.asarray(a,dtype=np.int64),
                                   np.asarray(b,dtype=np.int64))
        if np.any((a<0)|(a>1)|(b<0)|(b>1)):
            raise ValueError("A Boolean gate only accepts Bit wires")
        self.calls += 1
        self.elements += a.size
        r = role if self.role_override is None else self.role_override
        if self.direct:
            if self.model is None:
                raise ValueError("Direct execution requires the model")
            flat = np.stack([np.full(a.size,r),a.ravel(),b.ravel()],1)
            result = []
            with torch.no_grad():
                for start in range(0,len(flat),4096):
                    result.append(self.model(
                        torch.as_tensor(flat[start:start+4096])
                    ).argmax(-1).numpy())
            return np.concatenate(result).reshape(a.shape)
        return self.table[r,a,b]

    def leq(self, a, b, bits: int = 12, trace: bool = False):
        """MSB-first comparison. Later gates read earlier register writes."""
        a,b = np.broadcast_arrays(np.asarray(a,dtype=np.int64),
                                  np.asarray(b,dtype=np.int64))
        if np.any((a<0)|(b<0)|(a>=2**bits)|(b>=2**bits)):
            raise ValueError("Operand outside declared unsigned carrier")
        eq = np.ones(a.shape,dtype=np.int64)
        lt = np.zeros(a.shape,dtype=np.int64)
        history = []
        for bit in range(bits-1,-1,-1):
            aa=(a>>bit)&1
            bb=(b>>bit)&1
            lt=self.gate(OR,lt,self.gate(AND,eq,self.gate(LT,aa,bb)))
            eq=self.gate(AND,eq,self.gate(EQ,aa,bb))
            if trace:
                history.append({"bit":bit,"equal":eq.tolist(),"less":lt.tolist()})
        result=self.gate(OR,lt,eq).astype(bool)
        return (result,history) if trace else result

def train(out: Path, seed: int = 1, max_steps: int = 1200):
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    model=LoopedGate()
    opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=1e-4)
    x,y=rows()
    start=time.perf_counter()
    stable=0
    for step in range(max_steps):
        model.train()
        logits=model(x)
        loss=nn.functional.cross_entropy(logits,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits=model(x)
            acc=float((logits.argmax(-1)==y).float().mean())
            margin=float((logits.gather(1,y[:,None]) -
                          logits.gather(1,(1-y)[:,None])).min())
        stable=stable+1 if acc==1 and margin>4 else 0
        if stable>=10:
            break
    torch.save({"state_dict":model.state_dict(),"seed":seed},out/f"gate_seed{seed}.pt")
    bank=GateBank.load(out/f"gate_seed{seed}.pt")
    a,b=np.meshgrid(np.arange(64),np.arange(64),indexing="ij")
    pred=bank.leq(a,b,bits=6)
    with torch.no_grad():
        masked=float((model(x,mask_role=True).argmax(-1)==y).float().mean())
        one=float((model(x,loops=1).argmax(-1)==y).float().mean())
        zero=float((model(x,loops=0).argmax(-1)==y).float().mean())
    bad=GateBank(bank.table.copy())
    bad.role_override=0
    result={"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
            "steps":step+1,"seconds":time.perf_counter()-start,"gate_rows":len(x),
            "gate_exact":acc,"minimum_logit_margin":margin,
            "comparator_correct":int((pred==(a<=b)).sum()),"comparator_cases":a.size,
            "role_mask_gate_accuracy":masked,"one_visit_gate_accuracy":one,
            "zero_visit_gate_accuracy":zero,
            "role_zero_comparator_accuracy":float((bad.leq(a,b,bits=6)==(a<=b)).mean()),
            "neural_gate_elements":bank.elements,"logical_gate_calls":bank.calls,
            "training_domain":"all 64 finite truth-table rows; no ARC data",
            "backend":"CPU, hard categorical outputs; compiled finite table"}
    (out/f"gate_seed{seed}.json").write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("results"))
    p.add_argument("--seed",type=int,default=1)
    args=p.parse_args()
    print(json.dumps(train(args.out,args.seed),indent=2))
