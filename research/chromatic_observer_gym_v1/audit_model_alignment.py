"""Optional PyTorch replay proves frozen weights yield stored activations."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import torch
from run_frozen_loop import FrozenTiedLooper,canonical,VOCAB

def audit(weights_path,trace_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    weights=json.loads(Path(weights_path).read_text())
    traces=json.loads(Path(trace_path).read_text())
    max_abs=0.0
    reports=[]
    for item in traces['items']:
        seed=item['model_seed']
        w=weights[str(seed)]
        sha=hashlib.sha256(canonical(w).encode()).hexdigest()
        assert sha==item['model_checkpoint_sha256']
        m=FrozenTiedLooper()
        s=m.state_dict()
        s={name:torch.tensor(array, dtype=s[name].dtype) for name,array in w.items()}
        m.load_state_dict(s)
        m.eval()
        xs=torch.tensor([(a,b) for a in range(8) for b in range(8)])
        with torch.no_grad():
            logits,z=m(xs,trace=True)
        assert logits.argmax(-1).tolist()==item['model_predictions']
        raw=z.tolist()
        for a in range(8):
            for b in range(8):
                stored=item['activations'][f'{a}:{b}']
                for t in range(4):
                    for j in range(24):
                        max_abs=max(max_abs,abs(raw[8*a+b][t][j]-stored[t][j]))
        reports.append({'seed':seed,'sha256':sha,'accuracy64':int((logits.argmax(-1)==xs[:,0]).sum().item())})
    assert max_abs==0.0,max_abs
    return {'weights_to_trace_max_abs_diff':max_abs,'reports':reports}

if __name__=='__main__':
    root=Path(__file__).parent
    print(json.dumps(audit(root/'frozen_loop_weights.json',root/'frozen_loop_activations.json'),indent=2))
