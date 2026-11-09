"""Optional RRC-02 CPU training, PyTorch. Train split only, frozen hyperparameters.

Run from the repository root:
  python research/rrc_compactification/train_rrc02.py --steps 360
This trains a tied-weight recurrent residue classifier, a near-parameter-
matched feedforward control, and a shuffled-label recurrent control. The three
weights files are consumed and audited by rrc02.py without importing torch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

try:
    from .rrc02 import make_cases, training_rows, HERE, SEED, DEPTHS
    from .student02 import FEATURES, RECURRENT_HIDDEN, FF_HIDDEN, ResidueStudent
except ImportError:
    from rrc02 import make_cases, training_rows, HERE, SEED, DEPTHS
    from student02 import FEATURES, RECURRENT_HIDDEN, FF_HIDDEN, ResidueStudent


# Imported only by the optional, explicitly invoked trainer.
def train(*, steps: int = 360, batch_size: int = 512, seed: int = SEED,
          outdir: Path = HERE / 'results') -> dict:
    import torch
    from torch import nn
    import torch.nn.functional as F

    if steps < 1 or batch_size < 1:
        raise ValueError('positive training steps and batch size required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.set_grad_enabled(True)

    features, labels, membership = training_rows(make_cases(seed))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    count = Counter(labels)
    # Fixed class weighting offsets the natural overwhelming mixed-class mass.
    w = torch.tensor([len(labels) / (3 * count[i]) for i in range(3)], dtype=torch.float32)
    roster_sha = hashlib.sha256(json.dumps((features, labels, membership),
                                        separators=(',', ':')).encode()).hexdigest()

    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(FEATURES, RECURRENT_HIDDEN)
            self.loop = nn.Linear(RECURRENT_HIDDEN, RECURRENT_HIDDEN)
            self.ctx = nn.Linear(FEATURES, RECURRENT_HIDDEN)
            self.out = nn.Linear(RECURRENT_HIDDEN, 3)
        def forward(self, x, depth):
            h = self.enc(x).tanh()
            context = self.ctx(x)
            for _ in range(depth):
                h = (self.loop(h) + context).tanh()
            return self.out(h)

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(FEATURES, FF_HIDDEN[0])
            self.fc2 = nn.Linear(FF_HIDDEN[0], FF_HIDDEN[1])
            self.out = nn.Linear(FF_HIDDEN[1], 3)
        def forward(self, x, depth):
            assert depth == 0
            return self.out(self.fc2(self.fc1(x).tanh()).tanh())

    def fit(model, *, tag: str, shuffle_labels: bool, state_seed: int):
        torch.manual_seed(state_seed)
        model = model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.012)
        rng = random.Random(state_seed + 37)
        targets = y
        if shuffle_labels:
            order = list(range(len(y)))
            rng.shuffle(order)
            targets = y[torch.tensor(order, dtype=torch.long)]
        loss_history = []
        for iteration in range(steps):
            indices = torch.tensor([rng.randrange(len(y)) for _ in range(batch_size)], dtype=torch.long)
            depth = (DEPTHS[iteration % len(DEPTHS)] if tag != 'feedforward' else 0)
            logits = model(x[indices],depth)
            loss = F.cross_entropy(logits,targets[indices],weight=w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (iteration+1) % 60 == 0 or iteration+1 == steps:
                loss_history.append([iteration+1, round(float(loss.detach()), 6)])
        state = {}
        for name in (('enc','loop','ctx','out') if tag != 'feedforward'
                     else ('fc1','fc2','out')):
            layer = getattr(model,name)
            state[name+'_w'] = layer.weight.detach().tolist()
            state[name+'_b'] = layer.bias.detach().tolist()
        param_count = sum(p.numel() for p in model.parameters())
        checkpoint = {
            'schema': 'rrc02.tiny-student.v1', 'variant': 'recurrent' if tag != 'feedforward' else 'feedforward',
            'parameter_count': param_count,
            'training': {'split': 'train', 'optimizer': 'Adam', 'lr': .012,
                         'seed': state_seed, 'steps': steps, 'batch_size': batch_size,
                         'class_counts': {str(i): count[i] for i in range(3)},
                         'shuffled_train_labels': shuffle_labels,
                         'depth_schedule': list(DEPTHS) if tag != 'feedforward' else [0],
                         'torch': str(torch.__version__),
                         'training_rows_sha256': roster_sha, 'loss_history': loss_history},
            'weights': state,
        }
        # Guard that the portable pure-Python interpreter matches torch logits.
        portable = ResidueStudent(checkpoint)
        for depth in (DEPTHS if tag != 'feedforward' else (0,)):
            for row in (0, 1, 70, len(x)-1):
                expect = model(x[row:row+1], depth).detach().tolist()[0]
                got = portable.logits(tuple(features[row]), depth)
                if max(abs(a-b) for a,b in zip(expect,got)) > 5e-6:
                    raise AssertionError(f'torch/Python parity failed at {tag}:{depth}:{row}')
        out = outdir / f'rrc02_{tag}.json'
        out.write_text(json.dumps(checkpoint,indent=2,sort_keys=True)+'\n')
        return {'file': str(out), 'params': param_count,
                'last_weighted_batch_cross_entropy': loss_history[-1][-1],
                'train_label_count': len(y), 'tag': tag}

    outdir.mkdir(parents=True,exist_ok=True)
    return {
        'recurrent': fit(Recurrent, tag='recurrent', shuffle_labels=False, state_seed=seed+1),
        'feedforward': fit(FeedForward,tag='feedforward',shuffle_labels=False,state_seed=seed+2),
        'shuffled': fit(Recurrent,tag='shuffled',shuffle_labels=True,state_seed=seed+3),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=360)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    args=p.parse_args()
    print(json.dumps(train(steps=args.steps,batch_size=args.batch_size,outdir=args.out_dir),indent=2))

if __name__=='__main__':
    main()
