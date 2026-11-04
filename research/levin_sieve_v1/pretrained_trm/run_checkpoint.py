"""LSV-05 checkpoint-locked pretrained recurrent damage / accuracy benchmark.

The source model is an external third-party TRM-inspired 44K Sudoku model.
No training, uploaded user checkpoint or hidden labels enter this evaluator.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time, urllib.request
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from upstream_arch import TinyRecursiveModel
from sudoku_generator import generate_puzzle, is_valid_solution
from certify import certified_bounds
from goal_margin import certify_paired_hidden_states


def blob_sha1(data:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def fingerprint(x)->str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def locked_bytes(lock,checkpoint_path=None,allow_fetch=False):
    if checkpoint_path is not None:
        p=Path(checkpoint_path)
        if not p.is_file():raise ValueError('expected checkpoint file missing')
        data=p.read_bytes()
    else:
        if not allow_fetch:raise ValueError('explicit --fetch required to use network')
        uri=('https://raw.githubusercontent.com/'+lock['upstream_repo']+'/'
             +lock['revision']+'/'+lock['checkpoint_path'])
        with urllib.request.urlopen(uri,timeout=40) as resp:
            data=resp.read(lock['checkpoint_size_bytes']+1)
    assert len(data)==lock['checkpoint_size_bytes'],('unexpected checkpoint byte length',len(data))
    actual_blob=blob_sha1(data)
    if actual_blob != lock['checkpoint_git_blob_sha1']:
        raise ValueError('checkpoint Git blob SHA1 mismatch: '+actual_blob)
    sha=hashlib.sha256(data).hexdigest()
    if lock.get('checkpoint_sha256') is not None and sha!=lock['checkpoint_sha256']:
        raise ValueError('checkpoint SHA256 mismatch')
    return data,sha


def load_pretrained(lock, data):
    import io
    # Unpickle via restricted weights_only mode, never full pickle object loading.
    try:
        blob=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    except Exception as exc:
        raise ValueError('safe checkpoint parser rejected bytes') from exc
    if not isinstance(blob,dict) or set(blob)!={'state_dict','n_steps'} or blob['n_steps']!=4:
        raise ValueError('unexpected checkpoint format')
    model=TinyRecursiveModel(n_steps=4)
    model.load_state_dict(blob['state_dict'],strict=True)
    assert sum(p.numel() for p in model.parameters())==lock['n_parameters']
    return model.eval()


def prepare_roster(lock):
    puzzles=[];targets=[];holes=[]
    for i in range(lock['n_test_cases']):
        x,y,n=generate_puzzle(n_holes=lock['holes_requested'], seed=lock['test_seed_base']+i)
        if n!=lock['holes_requested']:
            raise AssertionError(f'under-dug puzzle {i}: {n}')
        assert is_valid_solution(y) and all(c==0 or c==v for c,v in zip(x,y))
        puzzles.append(x);targets.append(y);holes.append([int(c==0) for c in x])
    if len(set(tuple(p) for p in puzzles))!=len(puzzles):
        raise ValueError('duplicate puzzle detected in heldout roster')
    return (torch.tensor(puzzles,dtype=torch.int64),
            torch.tensor(targets,dtype=torch.int64),
            torch.tensor(holes,dtype=torch.bool)),fingerprint({'puzzles':puzzles,'targets':targets})


def summarize(logits, target, holes):
    p=logits.argmax(dim=-1)
    eq=p.eq(target)
    n=target.shape[0]
    valid=sum(is_valid_solution(row.tolist()) for row in p)
    return {
      'n':n,
      'full_grid_accuracy':round(eq.all(dim=1).float().mean().item(),6),
      'all_cell_accuracy':round(eq.float().mean().item(),6),
      'missing_cell_accuracy':round((eq & holes).sum().item()/holes.sum().item(),6),
      'valid_sudoku_rate':round(valid/n,6),
      'full_grid_correct_count':int(eq.all(dim=1).sum().item()),
      'valid_sudoku_count':int(valid),
    }


def evaluate(model, inputs, targets, holes, lock):
    torch.set_num_threads(1)
    with torch.inference_mode():
        states=model.reason_states(inputs)
        assert len(states)==5 and states[-1].shape==(len(inputs),81,48)
        clean=model.head(states[-1])
        stages={f'clean_depth_{i}':summarize(model.head(states[i]),targets,holes) for i in (1,2,3,4)}
        qmodel=TinyRecursiveModel(n_steps=4)
        qmodel.load_state_dict({k:v.to(torch.bfloat16).float() for k,v in model.state_dict().items()},strict=True)
        qmodel.eval()
        q_states=qmodel.reason_states(inputs)
        qlogits=qmodel.head(q_states[-1])
        quant={**summarize(qlogits,targets,holes),
               'max_logit_abs_delta_fp32':round((qlogits-clean).abs().amax().item(),8),
               'final_output_cell_flip_fraction':round((qlogits.argmax(-1)!=clean.argmax(-1)).float().mean().item(),8),
               'storage_dtype':'bfloat16 weights cast back to fp32; FP32 operator arithmetic'}
        # A delayed input is a full previous latent state, NOT coordinate-wise staleness.
        zdelay=model.block(states[1]);zdelay=model.block(states[1]);zdelay=model.block(zdelay)
        delay={'read_stage1_instead_of_stage2_once':summarize(model.head(zdelay),targets,holes)}
        damages=[]
        for seed in lock['damage_seeds']:
            rng=torch.Generator().manual_seed(seed)
            noise=torch.rand(states[1].shape,generator=rng)*2-1
            for amp_string in lock['damage_amplitudes']:
                amp=float(amp_string)
                z=states[1]+noise*amp
                errors=[float((z-states[1]).abs().amax().item())]
                for k in (2,3,4):
                    z=model.block(z)
                    errors.append(float((z-states[k]).abs().amax().item()))
                rec=summarize(model.head(z),targets,holes)
                rec['posthoc_head_margin']=certify_paired_hidden_states(model, states[-1], z)
                rec.update({'seed':seed,'damage_linf_limit':amp_string,
                    'observed_hidden_linf_after_steps':[round(v,8) for v in errors],
                    'final_output_cell_flip_fraction':round((model.head(z).argmax(-1)!=clean.argmax(-1)).float().mean().item(),8)})
                damages.append(rec)
    return {'stages':stages, 'quantized_storage':quant, 'stale_full_state':delay,'damage':damages}


def run(lock_path,checkpoint_path=None,allow_fetch=False):
    lock=json.loads(Path(lock_path).read_text())
    if lock['schema']!='LSV-05/pretrained-trm-source-lock-v1':raise ValueError('bad lock schema')
    data,sha=locked_bytes(lock,checkpoint_path,allow_fetch)
    model=load_pretrained(lock,data)
    (inputs,targets,holes),rosterhash=prepare_roster(lock)
    cert=certified_bounds(model)
    t0=time.perf_counter()
    metrics=evaluate(model,inputs,targets,holes,lock)
    elapsed=round(time.perf_counter()-t0,3)
    return {'schema':'LSV-05/pretrained-trm-eval-v1','classification':'independently_pretrained_third_party_TRM_inspired_CPU',
       'source_commit':lock['revision'],'checkpoint_git_blob_sha1':lock['checkpoint_git_blob_sha1'],
       'checkpoint_sha256':sha,'checkpoint_bytes':len(data),'roster_sha256':rosterhash,
       'torch_version':torch.__version__,'cpu_threads':1,'inference_seconds_excluding_loading_and_generation':elapsed,
       'batch_cases':len(inputs),'holes_per_case':int(holes[0].sum().item()),
       'mathematical_certificate':cert,'measured':metrics,
       'claim_scope':'weight-only real arithmetic bound, measured torch FP32; no global recovery contraction verified'}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--lock',type=Path,default=HERE/'source_lock.json')
    a.add_argument('--checkpoint',type=Path)
    a.add_argument('--fetch',action='store_true')
    a.add_argument('--out',type=Path)
    a.add_argument('--quick',type=int,help='development use only; short roster for smoke tests')
    v=a.parse_args()
    if v.quick:
        raise ValueError('development short-roster flag disabled for published receipts')
    receipt=run(v.lock,v.checkpoint,v.fetch)
    result=json.dumps(receipt,sort_keys=True,indent=2)+'\n'
    if v.out:
        v.out.parent.mkdir(parents=True,exist_ok=True)
        v.out.write_text(result)
    print(result)

if __name__=='__main__':main()
