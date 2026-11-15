"""COE-04 pure-stdlib verifier for stored pretrained-in-sandbox checkpoints.

The gzip artifacts are real frozen weights and hidden states produced by the
optional torch runner; verification of exact model inference needs torch.
"""
import gzip, hashlib, json
from pathlib import Path
from temporal_observers import canonical,validate_bundle

ROOT=Path(__file__).resolve().parent


def read_payload(name,root=ROOT):
    p=Path(root)/name
    if p.suffix=='.gz':
        with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
    return json.loads(p.read_text(encoding='utf8'))


def verify_frozen(root=ROOT):
    x=read_payload('frozen_loop_activations.json.gz',root)
    w=read_payload('frozen_loop_weights.json.gz',root)
    validate_bundle(x)
    verified=[]
    for item in x['items']:
        seed=item['model_seed']
        assert str(seed) in w
        digest=hashlib.sha256(canonical(w[str(seed)]).encode('utf8')).hexdigest()
        if digest != item['model_checkpoint_sha256']:
            raise ValueError('weight hash mismatch for seed '+str(seed))
        verified.append({'seed':seed,'checkpoint_sha256':digest})
    return x,verified
