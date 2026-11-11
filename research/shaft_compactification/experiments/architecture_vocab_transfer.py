"""SHAFT-COMP-18: preregistered cross-architecture/vocabulary history-probe test.

Two architectures (GRUCell and LSTMCell), two vocabularies (4,6), and one
history-dependent synthetic lag-rotation task. Entirely local in-memory,
no networking, replication, device actions, or persistence mechanisms.

Phase-gated: calibration runs before heldout. For heldout, pass the frozen
calibration threshold receipt and require a valid SHA-256 hash of its content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ARCHES = ("gru", "lstm")
VOCABS = (4, 6)
HIDDEN = 24
BATCH = 128
TRAIN_STEPS = 1600
TRAIN_LENGTH = 12
PROBE_STEPS = 400
PROBE_DEPTHS = (4, 8, 12)
STATE_DEPTHS = (12, 40)
PER_CELL_PAIR_COUNT = 256
DIAGNOSTIC_STEPS = 5
OUTCOME_STEPS = 32
BOOTSTRAPS = PERMUTATIONS = 1000
SEEDS = {
    "gru_v4": {"calibration": (1811,), "heldout": (1812, 1813)},
    "gru_v6": {"calibration": (1821,), "heldout": (1822, 1823)},
    "lstm_v4": {"calibration": (1831,), "heldout": (1832, 1833)},
    "lstm_v6": {"calibration": (1841,), "heldout": (1842, 1843)},
}
REGISTRATION_COMMIT = "32e97781cc6c4c02eb6c8e1e27e7c655eab015e0"
QUANTILE = 0.9


def cell_name(arch, vocab):
    if arch not in ARCHES or vocab not in VOCABS:
        raise ValueError("invalid architecture or query vocabulary")
    return f"{arch}_v{vocab}"


def stable_sha(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode("utf-8"))
        h.update(v.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


class RecurrentMemory(nn.Module):
    def __init__(self, arch, vocab):
        super().__init__()
        self.arch = arch
        self.vocab = vocab
        self.goals = 2**vocab
        self.dim = HIDDEN if arch == "gru" else 2 * HIDDEN
        self.embedding = nn.Embedding(self.goals, self.dim)
        self.cell = nn.GRUCell(vocab, HIDDEN) if arch == "gru" else nn.LSTMCell(vocab, HIDDEN)
        self.readout = nn.Sequential(nn.Linear(HIDDEN + vocab, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))

    def step(self, states, inputs):
        if self.arch == "gru":
            return self.cell(inputs, states)
        h, c = states.split(HIDDEN, dim=1)
        new_h, new_c = self.cell(inputs, (h, c))
        return torch.cat((new_h, new_c), dim=1)

    def visible(self, states):
        return states[:, :HIDDEN]

    def score(self, states, onehot):
        return self.readout(torch.cat((self.visible(states), onehot), dim=1)).squeeze(-1)

    def forward(self, goals, tokens):
        h = self.embedding(goals)
        out = []
        for t in range(tokens.shape[1]):
            x = F.one_hot(tokens[:, t], self.vocab).float()
            h = self.step(h, x)
            out.append(self.score(h, x))
        return torch.stack(out, dim=1)

    def step_jvp(self, states, delta, x):
        """Exact analytic directional derivative of GRUCell or LSTMCell.

        Avoids forming the full state Jacobian, works at both 24D and 48D.
        """
        if self.arch == "gru":
            gi = F.linear(x, self.cell.weight_ih, self.cell.bias_ih)
            gh = F.linear(states, self.cell.weight_hh, self.cell.bias_hh)
            ir, iz, inn = gi.chunk(3, dim=1)
            hr, hz, hn = gh.chunk(3, dim=1)
            r = torch.sigmoid(ir + hr)
            z = torch.sigmoid(iz + hz)
            n = torch.tanh(inn + r * hn)
            dhr, dhz, dhn = F.linear(delta, self.cell.weight_hh).chunk(3, dim=1)
            dr = r * (1-r) * dhr
            dz = z * (1-z) * dhz
            dn = (1-n.square())*(dr*hn + r*dhn)
            return (1-z)*dn + (states-n)*dz + z*delta
        h, c = states.split(HIDDEN, dim=1)
        dh, dc = delta.split(HIDDEN, dim=1)
        pre = F.linear(x, self.cell.weight_ih, self.cell.bias_ih) + F.linear(h, self.cell.weight_hh, self.cell.bias_hh)
        i, f, g, o = pre.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        dai, daf, dag, dao = F.linear(dh, self.cell.weight_hh).chunk(4, dim=1)
        di = i*(1-i)*dai
        df = f*(1-f)*daf
        dg = (1-g.square())*dag
        do = o*(1-o)*dao
        new_c = f*c + i*g
        new_dc = f*dc + df*c + di*g + i*dg
        tanh_c = torch.tanh(new_c)
        new_dh = do*tanh_c + o*(1-tanh_c.square())*new_dc
        return torch.cat((new_dh, new_dc), dim=1)


def targets(goals, tokens, vocab):
    prev = torch.cat((torch.zeros_like(tokens[:, :1]), tokens[:, :-1]), dim=1)
    bits = ((tokens + prev) % vocab).long()
    return ((goals[:, None] >> bits) & 1).float()


def train_model(arch, vocab, seed, steps=TRAIN_STEPS):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10000 + seed)
    model = RecurrentMemory(arch, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=.01)
    model.train()
    for _ in range(steps):
        g = torch.randint(model.goals, (BATCH,), generator=gen)
        tokens = torch.randint(vocab, (BATCH, TRAIN_LENGTH), generator=gen)
        y = targets(g, tokens, vocab)
        logits = model(g, tokens)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optim.step()
    model.eval()
    return model, round(float(loss.detach()), 9)


@torch.no_grad()
def test_task(model, seed, length=12, examples=512):
    gen = torch.Generator().manual_seed(900000 + seed + 43*length)
    g = torch.randint(model.goals, (examples,), generator=gen)
    tokens = torch.randint(model.vocab, (examples,length), generator=gen)
    truth = targets(g,tokens,model.vocab)
    pred = model(g,tokens)>=0
    return round(float((pred==truth.bool()).float().mean()),6)


@torch.no_grad()
def counterfactual(model, states, order=1, chunk=128):
    """Final-step native logits for all one- or two-token query words."""
    vocab = model.vocab
    if order not in (1,2): raise ValueError("order must be 1 or 2")
    first = torch.arange(vocab).repeat_interleave(vocab) if order==2 else torch.arange(vocab)
    second = torch.arange(vocab).repeat(vocab) if order==2 else None
    q_count = first.numel()
    x1 = F.one_hot(first,vocab).float()
    x2 = F.one_hot(second,vocab).float() if order==2 else None
    out=[]
    for start in range(0,len(states),chunk):
        h = states[start:start+chunk]
        repeated = h.repeat_interleave(q_count,dim=0)
        hot1=x1.repeat(len(h),1)
        state1=model.step(repeated,hot1)
        if order==2:
            hot2=x2.repeat(len(h),1)
            state2=model.step(state1,hot2)
            logits=model.score(state2,hot2)
        else:
            logits=model.score(state1,hot1)
        out.append(logits.reshape(len(h),q_count))
    return torch.cat(out,dim=0)


def sample_prefixes(model, depth, goals, seed):
    gen=torch.Generator().manual_seed(seed)
    tokens=torch.randint(model.vocab,(len(goals),depth),generator=gen)
    with torch.no_grad():
        states=model.embedding(goals)
        for t in range(depth):
            states=model.step(states,F.one_hot(tokens[:,t],model.vocab).float())
    return states


def train_probes(model, seed):
    contexts_per_goal=min(32,2304//(model.goals*len(PROBE_DEPTHS)))
    goal_base=torch.arange(model.goals).repeat_interleave(contexts_per_goal)
    states=[]
    for depth in PROBE_DEPTHS:
        rng=torch.Generator().manual_seed(seed*10000 + depth + 991)
        shuffle=goal_base[torch.randperm(len(goal_base),generator=rng)]
        states.append(sample_prefixes(model,depth,shuffle,seed*20000+depth+52))
    x=torch.cat(states)
    with torch.no_grad():
        y1=(counterfactual(model,x,1)>=0).float()
        y2=(counterfactual(model,x,2)>=0).float()
    trained=[]
    for order,y in [(1,y1),(2,y2)]:
        torch.manual_seed(2000000+seed+order*1000)
        probe=nn.Linear(model.dim,model.vocab**order)
        optim=torch.optim.Adam(probe.parameters(),lr=.04)
        for _ in range(PROBE_STEPS):
            loss=F.binary_cross_entropy_with_logits(probe(x),y)
            optim.zero_grad();loss.backward();optim.step()
        trained.append((probe.eval(),round(float(loss.detach()),9)))
    return trained,x.shape[0]


def test_jvp(model, seed):
    gen=torch.Generator().manual_seed(seed*5+111)
    h=torch.randn(model.dim,generator=gen)
    v=torch.randn(model.dim,generator=gen)
    inp=F.one_hot(torch.tensor([2]),model.vocab).float()
    with torch.no_grad():
        analytic=model.step_jvp(h[None],v[None],inp)[0]
    h=h.detach().requires_grad_(True)
    with torch.enable_grad():
        _,auto=torch.autograd.functional.jvp(lambda s:model.step(s[None],inp)[0],h,v)
    err=float((analytic-auto).abs().max())
    if err>=1e-5:
        raise AssertionError(f"{model.arch}/{model.vocab} JVP error {err}")
    return err


def pairs_for_cell(model, probe1, probe2, seed, depth, pairs_count=256):
    if pairs_count!=PER_CELL_PAIR_COUNT:raise ValueError("must use registered 256 pairs/cell")
    n_each = PER_CELL_PAIR_COUNT//model.goals
    goals=torch.arange(model.goals).repeat_interleave(n_each)
    gen=torch.Generator().manual_seed(6000000 + seed*100 + depth)
    goals=goals[torch.randperm(len(goals),generator=gen)]
    a=sample_prefixes(model,depth,goals,7000000+seed*100+depth)
    b=sample_prefixes(model,depth,goals,8000000+seed*100+depth)
    qdiag=torch.Generator().manual_seed(9000000+seed*100+depth)
    qout=torch.Generator().manual_seed(10000000+seed*100+depth)
    diag=torch.randint(model.vocab,(len(goals),DIAGNOSTIC_STEPS),generator=qdiag)
    future=torch.randint(model.vocab,(len(goals),OUTCOME_STEPS),generator=qout)
    with torch.no_grad():
        ta=counterfactual(model,a,1)>=0
        tb=counterfactual(model,b,1)>=0
        p1a=probe1(a)>=0
        p1b=probe1(b)>=0
        p2a=probe2(a)>=0
        p2b=probe2(b)>=0
        eligible=(ta==tb).all(dim=1)&(p1a==p1b).all(dim=1)
        # Other probe's signature is measured but not part of eligibility.
        probe2_diff=(p2a!=p2b).any(dim=1)
        h1=(p1a==(counterfactual(model,a,1)>=0)).float().mean()
        h2=(p2a==(counterfactual(model,a,2)>=0)).float().mean()
        ha,hb=a.clone(),b.clone()
        for t in range(OUTCOME_STEPS):
            token=F.one_hot(future[:,t],model.vocab).float()
            ha=model.step(ha,token)
            hb=model.step(hb,token)
        diverged=(counterfactual(model,ha,1)>=0)!=(counterfactual(model,hb,1)>=0)
        labels=diverged.any(dim=1)
    with torch.no_grad():
        delta=a-b
        norm=delta.norm(dim=1)
        results=[]
        for probe in (probe1,probe2):
            energy=[]
            for states in (a,b):
                h=states.clone()
                tangent=delta.clone()
                e=torch.zeros(len(goals))
                W=probe.weight.detach()
                for t in range(DIAGNOSTIC_STEPS+1):
                    obs=F.linear(tangent,W)
                    e+=obs.square().sum(dim=1)
                    if t<DIAGNOSTIC_STEPS:
                        x=F.one_hot(diag[:,t],model.vocab).float()
                        tangent=model.step_jvp(h,tangent,x)
                        h=model.step(h,x)
                energy.append(e)
            result=torch.sqrt(.5*(energy[0]+energy[1]).clamp_min(0)/probe.out_features)
            results.append(result)
        score1,score2=results
    rows=[]
    for i in range(len(goals)):
        rows.append({
            "arch":model.arch,"vocab":model.vocab,"seed":seed,"depth":depth,
            "cell":cell_name(model.arch,model.vocab),"pair":i,
            "goal_audit_only":int(goals[i]),"eligible":bool(eligible[i]),
            "diverged_32":int(labels[i]),"history_code_distinguishes":bool(probe2_diff[i]),
            "instant":float(score1[i]),"history":float(score2[i]),
            "instant_norm":float(score1[i]/norm[i].clamp_min(1e-9)),
            "history_norm":float(score2[i]/norm[i].clamp_min(1e-9)),
            "distance":float(norm[i]),
        })
    return rows,{"one_probe_bit_accuracy":round(float(h1),6),"two_probe_bit_accuracy":round(float(h2),6),
                 "history_code_separated_eligible":int((probe2_diff&eligible).sum()),
                 "eligible":int(eligible.sum()),"positives":int((eligible&labels).sum())}


def auc(scores, labels):
    scores=np.asarray(scores,float);labels=np.asarray(labels,int)
    p=int(labels.sum());n=len(labels)-p
    if not p or not n:return None
    order=np.argsort(scores,kind="stable")
    values=scores[order]
    ranks=np.empty(len(scores),float)
    i=0
    while i<len(scores):
        j=i+1
        while j<len(scores) and values[j]==values[i]:j+=1
        ranks[order[i:j]]=(i+1+j)/2
        i=j
    return float((ranks[labels==1].sum()-p*(p+1)/2)/(p*n))


def summarize(rows):
    valid=[r for r in rows if r["eligible"]]
    y=[r["diverged_32"] for r in valid]
    fields=("instant","history","instant_norm","history_norm","distance")
    return {"total":len(rows),"eligible":len(valid),"positives":sum(y),
            "auc":{f:None if (value:=auc([r[f] for r in valid],y)) is None else round(value,6) for f in fields}}


def quantile_threshold(rows, feature):
    neg=np.asarray([r[feature] for r in rows if r["eligible"] and r["diverged_32"]==0],float)
    if len(neg)==0:raise ValueError("cannot set threshold without negatives")
    return float(np.quantile(neg,.9,method="higher"))


def operating_point(rows, feature, threshold):
    e=[r for r in rows if r["eligible"]]
    neg=[r for r in e if not r["diverged_32"]]
    pos=[r for r in e if r["diverged_32"]]
    fp=sum(r[feature]>threshold for r in neg)
    tp=sum(r[feature]>threshold for r in pos)
    return {"threshold":threshold,"negatives":len(neg),"positives":len(pos),
            "false_positives":fp,"true_positives":tp,
            "fpr":None if not neg else round(fp/len(neg),6),
            "tpr":None if not pos else round(tp/len(pos),6)}


def run_phase(phase,steps=TRAIN_STEPS,cells=None):
    if phase not in ("calibration","heldout"):
        raise ValueError("phase must be calibration or heldout")
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    report={"experiment":"SHAFT-COMP-18","phase":phase,"registration_commit":REGISTRATION_COMMIT,
            "torch":torch.__version__,"numpy":np.__version__,"train_steps":steps,
            "readout_scale":"divide directional energy by number of probe outputs",
            "models":{},"cells":{},"rows":[]}
    chosen = tuple(SEEDS) if cells is None else tuple(cells)
    if not chosen or any(key not in SEEDS for key in chosen):
        raise ValueError("empty or invalid requested cell selection")
    for key,seed_group in SEEDS.items():
        if key not in chosen: continue
        arch,vocab=key.split("_v");vocab=int(vocab)
        for seed in seed_group[phase]:
            model,loss=train_model(arch,vocab,seed,steps)
            trained,train_examples=train_probes(model,seed)
            (one,loss1),(two,loss2)=trained
            mod_key=str(seed)
            report["models"][mod_key]={"cell":key,"architecture":arch,"vocab":vocab,
                "parameters":sum(p.numel() for p in model.parameters()),
                "state_dim":model.dim,"model_train_loss":loss,"probe_losses":{"one":loss1,"two":loss2},
                "probe_train_states":train_examples,
                "accuracy_d12":test_task(model,seed,12),"accuracy_d32":test_task(model,seed,32),
                "model_sha256":stable_sha(model),"immediate_probe_sha256":stable_sha(one),
                "history_probe_sha256":stable_sha(two),
                "jvp_parity":test_jvp(model,seed)}
            for depth in STATE_DEPTHS:
                sample,audit=pairs_for_cell(model,one,two,seed,depth)
                report["cells"][f"{mod_key}_d{depth}"]={**audit,"scores":summarize(sample)}
                report["rows"].extend(sample)
    report["aggregate"]=summarize(report["rows"])
    report["by_arch_vocab"]={k:summarize([r for r in report["rows"] if r["cell"]==k]) for k in chosen}
    report["by_model"]={str(seed):summarize([r for r in report["rows"] if r["seed"]==seed])
                        for groups in SEEDS.values() for seed in groups[phase]}
    return report


def summarize_calibration(report):
    assert report["phase"]=="calibration"
    thresholds={f:quantile_threshold(report["rows"],f) for f in ("instant","history")}
    return {"experiment":"SHAFT-COMP-18","phase":"calibration_thresholds",
            "registration_commit":REGISTRATION_COMMIT,"selection":"numpy higher quantile 0.90, strict >",
            "global_thresholds":thresholds,
            "source_operating_points":{f:operating_point(report["rows"],f,x) for f,x in thresholds.items()},
            "cell_thresholds":{cell:{f:quantile_threshold([r for r in report["rows"] if r["cell"]==cell],f)
                    for f in thresholds} for cell in SEEDS},
            "source_models":report["models"],"source_aggregate":report["aggregate"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("calibration","heldout"))
    p.add_argument("--train-steps",type=int,default=TRAIN_STEPS)
    p.add_argument("--cell",choices=tuple(SEEDS),help="execute one preregistered cell; merge all cells before evaluation")
    p.add_argument("--out",type=Path)
    p.add_argument("--thresholds-out",type=Path)
    args=p.parse_args()
    result=run_phase(args.phase,args.train_steps,cells=(args.cell,) if args.cell else None)
    if args.out:
        args.out.parent.mkdir(exist_ok=True,parents=True)
        args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    else: print(json.dumps(result,indent=2,sort_keys=True))
    if args.thresholds_out:
        if args.phase!="calibration" or args.cell:
            raise ValueError("thresholds require complete calibration across all cells")
        args.thresholds_out.write_text(json.dumps(summarize_calibration(result),indent=2,sort_keys=True)+"\n")

if __name__=="__main__":main()
