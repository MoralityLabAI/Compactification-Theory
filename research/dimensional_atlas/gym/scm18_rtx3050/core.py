"""Device-aware SCM-16 reuse, immutable receipts and explicit work accounting."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..model import LoopOutput
from ..scm16.fiber import FiberATVPD, FiberLoopedMathTransformer
from ..tasks import (PROGRAMS, MathTask, encode, make_evalmath_case, check_candidate,
                     verify_order)

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def digest(module, prefix=None):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if prefix and not name.startswith(prefix):
            continue
        v = value.detach().cpu().contiguous()
        h.update(json.dumps([name, str(v.dtype), list(v.shape)]).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def device_setup(device, config):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Disable optimized MHA inference: keep train/eval and intervention code paths alike.
    torch.backends.mha.set_fastpath_enabled(False)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no implicit CPU substitution')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(config['max_allocator_mb'] * 2**20 / total)
        free, _ = torch.cuda.mem_get_info()
        if free < config['min_free_vram_mb'] * 2**20:
            raise RuntimeError('less than registered VRAM headroom')
        torch.cuda.reset_peak_memory_stats()
    return torch.device(device)


class Meter:
    def __init__(self, device, config):
        self.device, self.config = device, config
        self.counts = Counter()
        self.min_free = None
        self.started = time.perf_counter()
        self.sample_memory()

    def sample_memory(self):
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info()
            self.min_free = free if self.min_free is None else min(self.min_free, free)
            if free < self.config['min_free_vram_mb'] * 2**20:
                raise RuntimeError('observed VRAM headroom below registration')

    def forward(self, n, visits):
        self.counts['model_calls'] += 1
        self.counts['forward_examples'] += n
        self.counts['recurrent_example_visits'] += n * visits
        self.sample_memory()

    def report(self):
        self.sample_memory()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return {'work': dict(self.counts), 'seconds': time.perf_counter()-self.started,
                'peak_allocated_mb': torch.cuda.max_memory_allocated()/2**20 if self.device.type == 'cuda' else 0,
                'peak_reserved_mb': torch.cuda.max_memory_reserved()/2**20 if self.device.type == 'cuda' else 0,
                'minimum_sampled_free_mb': self.min_free/2**20 if self.min_free is not None else None,
                'memory_sampling': 'every_model_call_and_receipt_not_continuous',
                'device': str(self.device)}


class MathModel(FiberLoopedMathTransformer):
    """SCM-16 block/head with visit hooks and optionally untied cloned blocks."""
    def __init__(self, config, visits, rank=4, untied=False):
        super().__init__(hidden=config['hidden'], heads=config['heads'], visits=visits,
                         sae_width=config['sae_width'], sae_k=config['sae_k'],
                         vpd_rank=rank, readout_dim=config['readout_dim'])
        self.untied = untied
        if untied:
            original = self.core
            self.core = nn.ModuleList([copy.deepcopy(original) for _ in range(visits)])
        self.meter = None

    def readout(self, h):
        if self.meter:
            self.meter.counts['readout_examples'] += h.shape[0]
        return self.head(h[:, 3])

    def forward(self, data, *, policy='tied', hook=None, intermediate=False):
        if data.ndim != 3 or tuple(data.shape[1:]) != (4, 30):
            raise ValueError('expected public [batch,4,30] inputs')
        if self.meter:
            self.meter.forward(len(data), self.visits)
        h = self.embed(data) + self.positions
        if hook is not None:
            h = hook(-1, h)
        states, codes, logits_by_visit = [], [], []
        energy = h.new_zeros(())
        for v in range(self.visits):
            h = (self.core[v] if self.untied else self.core)(h)
            code = self.sae.encode(F.layer_norm(h, (self.hidden,)))
            # Tied and zero-code inference execute identical adapter operations.
            if bool(self.vpd.frame_ready):
                cc = torch.zeros_like(code) if policy in ('tied', 'zero_codes') else code
                sector = policy if policy in ('fiber_only', 'visible_only') else 'joint'
                order = tuple(reversed(range(self.visits))) if policy == 'shuffled_visit' else None
                delta, _, _, _ = self.vpd(h, cc, v, sector=sector, visit_order=order)
                h = h + delta
                energy = energy + delta.square().mean()
            if hook is not None:
                h = hook(v, h)
            states.append(h)
            codes.append(code)
            if intermediate:
                logits_by_visit.append(self.readout(h))
        logits = logits_by_visit[-1] if intermediate else self.readout(h)
        return LoopOutput(logits, tuple(states), tuple(codes), energy, tuple(logits_by_visit))


def pool(oracle, n, start, split, families, seen):
    result = []
    for i in range(n * 30):
        family = families[i % len(families)]
        task = make_evalmath_case(oracle, start+i, case_id=f'{split}:{start+i}', family=family)
        if task.signature in seen:
            continue
        seen.add(task.signature)
        result.append(task)
        if len(result) == n:
            return result
    raise RuntimeError('fresh generator exhausted')


def tensors(tasks, device):
    return (torch.stack([encode(t) for t in tasks]).to(device),
            torch.tensor([PROGRAMS.index(t.source_graph) for t in tasks], device=device))


def base_parameters(model):
    return [p for name, p in model.named_parameters() if not name.startswith(('sae.', 'vpd.'))]


def train_base(model, tasks, config, seed, objective='final', steps=None):
    device = next(model.parameters()).device
    x, y = tensors(tasks, device)
    steps = config['base_steps'] if steps is None else steps
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith('vpd.') and
                         (not name.startswith('sae.') or objective in ('sae_regularized', 'causal_consistency')))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=.01)
    rng = torch.Generator().manual_seed(seed+1)
    coeff = config['loss_coefficients']
    rows = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        ids = torch.randint(len(tasks), (config['batch'],), generator=rng).to(device)
        out = model(x[ids], intermediate=objective in ('intermediate', 'causal_consistency'))
        math_loss = (torch.stack([F.cross_entropy(z, y[ids]) for z in out.gates]).mean()
                     if objective in ('intermediate', 'causal_consistency') else F.cross_entropy(out.logits, y[ids]))
        sae_loss = math_loss.new_zeros(())
        consistency = math_loss.new_zeros(())
        if objective in ('sae_regularized', 'causal_consistency'):
            hs = torch.cat([F.layer_norm(s, (model.hidden,)) for s in out.states])
            reconstruction, _ = model.sae(hs)
            sae_loss = F.mse_loss(reconstruction, hs)
        if objective == 'causal_consistency':
            consistency = torch.stack([F.kl_div(F.log_softmax(b, -1), F.softmax(a.detach(), -1),
                                                 reduction='batchmean') for a,b in zip(out.gates, out.gates[1:])]).mean()
        loss = math_loss + coeff['sae']*sae_loss + coeff['causal']*consistency
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite base loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 2.)
        optimizer.step()
        if model.meter:
            model.meter.counts['base_optimizer_steps'] += 1
        if step % 20 == 0 or step == steps-1:
            rows.append({'step':step+1, 'math':float(math_loss.detach()), 'sae':float(sae_loss.detach()),
                         'causal':float(consistency.detach()), 'transport':0., 'total':float(loss.detach())})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.lock_fiber()
    return {'objective':objective, 'steps':steps, 'losses':rows, 'seconds':time.perf_counter()-started,
            'parameters_updated':sum(p.numel() for p in params)}


def fit_dictionary(model, tasks, config, seed):
    device = next(model.parameters()).device
    x, _ = tensors(tasks, device)
    cache = []
    with torch.no_grad():
        for begin in range(0, len(x), config['batch']):
            out = model(x[begin:begin+config['batch']])
            cache.extend([F.layer_norm(s, (model.hidden,)).reshape(-1, model.hidden).cpu() for s in out.states])
    target = torch.cat(cache)
    del cache, x, out
    for p in model.sae.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.sae.parameters(), lr=config['lr'])
    rng = torch.Generator().manual_seed(seed+2)
    rows = []
    for step in range(config['sae_steps']):
        ids = torch.randint(len(target), (config['batch'],), generator=rng)
        xb = target[ids].to(device)
        recon, code = model.sae(xb)
        loss = F.mse_loss(recon, xb) + .0001*code.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if model.meter:
            model.meter.counts['sae_optimizer_steps'] += 1
        if step % 20 == 0 or step == config['sae_steps']-1:
            rows.append({'step':step+1, 'reconstruction':float(loss.detach())})
    for p in model.sae.parameters():
        p.requires_grad_(False)
    return {'steps':config['sae_steps'], 'cached_cpu_rows':len(target), 'losses':rows,
            'sha256':digest(model.sae)}


def replace_adapter(model, config, rank, seed):
    torch.manual_seed(seed+3)
    model.vpd = FiberATVPD(model.hidden, config['sae_width'], rank, model.visits,
                            config['readout_dim']).to(next(model.parameters()).device)
    model.vpd.lock_readout(model.head[0].weight)


def operator_counterfactual(task, oracle):
    """Exact do(first-operator=other), holding all source series and topology fixed."""
    graph=(('N' if task.source_graph[0]!='N' else 'R'),*task.source_graph[1:])
    case=task.oracle_case
    demos=tuple((f,oracle.execute(f,task.topology,graph)) for f,_ in case.demos)
    future=oracle.execute(case.query,task.topology,graph)
    changed=replace(case,identifier=case.identifier+':operator_do',demos=demos,future=future,source_graph=graph)
    return MathTask(task.case_id+':operator_do',task.family,task.topology,
                    tuple((tuple(f[:13]),tuple(y[:13])) for f,y in demos),
                    task.query,tuple(future),graph,'evalmath_arg01',changed)


def train_adapter(model, tasks, config, seed, sector='joint', objective='final', oracle=None):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.vpd.parameters():
        p.requires_grad_(True)
    frozen_base, frozen_sae = digest(model, ('embed.', 'positions', 'core.', 'head.')), digest(model.sae)
    device = next(model.parameters()).device
    x,y = tensors(tasks, device)
    if objective=='causal_consistency':
        if oracle is None:
            raise ValueError('independent oracle required for semantic causal training')
        counterfactuals=[operator_counterfactual(t,oracle) for t in tasks]
        cf_x,cf_y=tensors(counterfactuals,device)
    rng = torch.Generator().manual_seed(seed+4)
    opt = torch.optim.AdamW(model.vpd.parameters(), lr=config['lr'], weight_decay=.01)
    rows = []
    started = time.perf_counter()
    for step in range(config['adapter_steps']):
        ids = torch.randint(len(tasks), (config['batch'],), generator=rng).to(device)
        out = model(x[ids], policy=sector)
        ce = F.cross_entropy(out.logits, y[ids])
        sae_loss=ce.new_zeros(())
        causal_loss=ce.new_zeros(())
        if objective in ('sae_regularized','causal_consistency'):
            hs=torch.cat([F.layer_norm(s,(model.hidden,)) for s in out.states])
            reconstruction,_=model.sae(hs)
            sae_loss=F.mse_loss(reconstruction,hs)
        if objective=='causal_consistency':
            donor=model(cf_x[ids],policy=sector)
            at=step % max(1,model.visits-1)
            basis=torch.linalg.qr(model.vpd.write_factor.detach().T).Q
            delta=((donor.states[at]-out.states[at])@basis)@basis.T
            if sector=='fiber_only':
                q=model.vpd.readout_basis
                delta=delta-(delta@q)@q.T
            swapped=model(x[ids],policy=sector,hook=lambda v,h:h+delta if v==at else h)
            causal_loss=F.cross_entropy(swapped.logits,cf_y[ids])
        coeff=config['loss_coefficients']
        loss = ce + coeff['sae']*sae_loss + coeff['causal']*causal_loss + coeff['transport']*out.correction_energy
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite adapter loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.vpd.parameters(), 2.)
        opt.step()
        if model.meter:
            model.meter.counts['adapter_optimizer_steps'] += 1
        if step % 20 == 0 or step == config['adapter_steps']-1:
            rows.append({'step':step+1, 'math':float(ce.detach()), 'sae':float(sae_loss.detach()), 'causal':float(causal_loss.detach()),
                         'transport':float(out.correction_energy.detach()), 'total':float(loss.detach())})
    for p in model.parameters():
        p.requires_grad_(False)
    assert digest(model, ('embed.', 'positions', 'core.', 'head.')) == frozen_base
    assert digest(model.sae) == frozen_sae
    return {'steps':config['adapter_steps'], 'sector':sector, 'losses':rows,
            'seconds':time.perf_counter()-started,
            'trainable_parameters':sum(p.numel() for p in model.vpd.parameters()),
            'base_unchanged':True, 'dictionary_unchanged':True}


def future_correct(oracle, task, index, meter=None):
    graph = PROGRAMS[int(index)]
    if not oracle.check_candidate(task.oracle_case, graph)['accepted']:
        return False
    return bool(oracle.grade(task.oracle_case, {'graph':graph})['future_correct'])


@torch.no_grad()
def evaluate(model, tasks, oracle, budget, policy='tied', hook=None):
    if budget < 1:
        raise ValueError('budget must be positive')
    device = next(model.parameters()).device
    rows = []
    started = time.perf_counter()
    for task in tasks:
        logits = model(encode(task)[None].to(device), policy=policy, hook=hook).logits[0]
        order = logits.argsort(descending=True).cpu().tolist()
        outcome = verify_order(task, order, budget=budget, oracle=oracle)
        top = future_correct(oracle, task, order[0], model.meter)
        if model.meter:
            model.meter.counts['paid_candidate_queries'] += outcome['queries']
        rows.append({'id':task.case_id, 'signature':task.signature, 'family':task.family,
                     'correct':int(outcome['future_correct']), 'top1_correct':int(top),
                     'queries':outcome['queries'], 'false_future':int(outcome['false_future_acceptance'])})
    def aggregate(rs):
        n = len(rs)
        return {'n':n, 'correct':sum(r['correct'] for r in rs),
                'accuracy':sum(r['correct'] for r in rs)/n if n else None,
                'top1_correct':sum(r['top1_correct'] for r in rs),
                'verifier_calls':sum(r['queries'] for r in rs),
                'false_future_acceptances':sum(r['false_future'] for r in rs)}
    groups = defaultdict(list)
    for row in rows:
        groups[row['family']].append(row)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return {'overall':aggregate(rows), 'families':{f:aggregate(rs) for f,rs in groups.items()},
            'records':rows, 'seconds':time.perf_counter()-started, 'budget':budget}


def checkpoint(path, model, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'metadata':metadata}, path)
    return {'file':path.name, 'file_sha256':sha(path), 'model_sha256':digest(model),
            'sae_sha256':digest(model.sae), **metadata}


def load_checkpoint(path, config, device):
    value = torch.load(path, map_location='cpu', weights_only=True)
    meta = value['metadata']
    model = MathModel(config, meta['visits'], meta['rank'], meta.get('untied',False))
    model.load_state_dict(value['state'], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval(), meta


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment():
    return {'python':platform.python_version(), 'platform':platform.platform(),
            'torch':torch.__version__, 'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version',
                                       '--format=csv'],capture_output=True,text=True).stdout.strip()}
