"""COE-04: standard-library-only, frozen-trace, pre-split probe benchmark.

Heads are selected solely on probe training contexts, and never fitted or
chosen against the 4 held-out distractor contexts. The evaluator is K8 on
frozen model's task labels, fixed independently of the selected probes.

No checkpoint weights or model computations are needed for the scoring replay.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
import math
import random
from pathlib import Path

N = 8
T = 4
D = 24
FIT_CONTEXTS = (0,1,2,3)
TEST_CONTEXTS = (4,5,6,7)
EDGES = tuple(combinations(range(N),2))


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)


def trace_digest(item):
    return hashlib.sha256(canonical(item["activations"]).encode()).hexdigest()


def validate_bundle(data):
    if data["schema"] != "COE-04-native-tied-loop-frozen-v1":
        raise ValueError("unexpected schema")
    assert data["probe_fit_contexts"] == list(FIT_CONTEXTS)
    assert data["probe_heldout_contexts"] == list(TEST_CONTEXTS)
    seeds = set()
    for item in data["items"]:
        if item["model_seed"] in seeds:raise ValueError("duplicate model seed")
        seeds.add(item["model_seed"])
        assert item["architecture"] == "tied_causal_mha_1block_4visits_2tokens_d24_h3"
        assert item["train_steps"] == 160
        assert len(item["model_predictions"]) == N * N
        assert all(item["model_predictions"][8*a+b] == a for a in range(N) for b in range(N))
        if set(item["activations"]) != {f"{a}:{b}" for a in range(N) for b in range(N)}:
            raise ValueError("incomplete inputs")
        for x in item["activations"].values():
            if len(x)!=T or any(len(step)!=D or not all(math.isfinite(v) for v in step) for step in x):
                raise ValueError("invalid frozen activation tensor")
    return True


def dot(w, x):return sum(a*b for a,b in zip(w,x))


def normalize(v):
    size=math.sqrt(sum(x*x for x in v))
    if size<1e-8:return None
    return tuple(x/size for x in v)


def means(item,contexts=FIT_CONTEXTS):
    return tuple(tuple(tuple(sum(item['activations'][f'{a}:{b}'][t][j] for b in contexts)/len(contexts) for j in range(D)) for t in range(T)) for a in range(N))


def candidate_projections(item,seed=2204,limit_random=240):
    """Generate fixed deterministic heads on fit contexts only."""
    avg=means(item)
    directions=[]
    for t in range(T):
        for a,b in EDGES:
            v=normalize(tuple(avg[a][t][j]-avg[b][t][j] for j in range(D)))
            if v: directions.append(v)
    rng=random.Random(seed)
    for _ in range(limit_random):
        vec=normalize([rng.gauss(0,1) for _ in range(D)])
        if vec:directions.append(vec)
    # Each threshold comes from 1/8, 1/4, 1/2, 3/4, 7/8 quantiles of
    # pooled fit-context projections; none depends on the holdout trace.
    unique={}
    for v in directions:
        vals=sorted(dot(v,item['activations'][f'{a}:{c}'][t]) for a in range(N) for c in FIT_CONTEXTS for t in range(T))
        thresholds=[]
        for fraction in (0.125,0.25,0.375,0.5,0.625,0.75,0.875):
            idx=int((len(vals)-1)*fraction)
            thresholds.append((vals[idx]+vals[idx+1])/2)
        # Additional thresholds between sorted per-class means at each timestep
        for t in range(T):
            arr=sorted(dot(v,avg[a][t]) for a in range(N))
            for j in range(N-1):thresholds.append((arr[j]+arr[j+1])/2)
        for threshold in thresholds:
            words = {str(c):tuple(sum((int(dot(v,item['activations'][f'{a}:{c}'][t])>threshold)<<t) for t in range(T)) for a in range(N)) for c in FIT_CONTEXTS}
            signature=tuple(value for c in FIT_CONTEXTS for value in words[str(c)])
            if signature not in unique:
                unique[signature]=(v,threshold)
    return tuple(unique.values())


def encode(item,head,contexts,mode='temporal',quant_bits=1):
    w,threshold=head
    values=[]
    for c in contexts:
        row=[]
        for a in range(N):
            selected=item['activations'][f'{a}:{c}']
            if mode=='last': selected=selected[-1:]
            elif mode=='repeat_last': selected=selected[-1:]*T
            elif mode != 'temporal':raise ValueError(mode)
            z=tuple(dot(w,vec) for vec in selected)
            if quant_bits==1:code=tuple(int(val>threshold) for val in z)
            else:
                # fixed fit-only calibration for this head, shared across visits
                fit=[dot(w,item['activations'][f'{aa}:{cc}'][tt]) for aa in range(N) for cc in FIT_CONTEXTS for tt in range(T)]
                lo,hi=min(fit),max(fit)
                levels=(1<<quant_bits)-1
                code=tuple(max(0,min(levels,int((v-lo)*levels/(hi-lo)))) if hi>lo else 0 for v in z)
            row.append(code)
        values.append(row)
    return values


def covered_pairs(encoded):
    return sum(int(row[a]!=row[b]) for row in encoded for a,b in EDGES)


def fit_candidate(item, mode, metric_bits=1, cache=None):
    candidates = candidate_projections(item,seed=item['model_seed']+601)
    if cache is None:cache={}
    best=None
    for head in candidates:
        train=encode(item,head,FIT_CONTEXTS,mode=mode,quant_bits=metric_bits)
        coverage=covered_pairs(train)
        # stable codewords across train contexts a secondary preference
        stability=sum(len(set(train[c][a] for c in range(len(FIT_CONTEXTS))))==1 for a in range(N))
        key=(coverage,stability)
        if best is None or key>best['key']:
            best={'head':head,'key':key,'candidates_examined':len(candidates)}
    assert best is not None
    return best


def head_report(item,head,mode,quant_bits=1):
    fitted=encode(item,head,FIT_CONTEXTS,mode,quant_bits)
    heldout=encode(item,head,TEST_CONTEXTS,mode,quant_bits)
    return {
        'fit_edge_hits':covered_pairs(fitted),
        'fit_edge_total':len(FIT_CONTEXTS)*len(EDGES),
        'heldout_edge_hits':covered_pairs(heldout),
        'heldout_edge_total':len(TEST_CONTEXTS)*len(EDGES),
        'heldout_label_unique_per_context':[len(set(tuple(w) for w in row)) for row in heldout],
    }


def run_study(doc):
    validate_bundle(doc)
    rows=[]
    for item in doc['items']:
        temporal=fit_candidate(item,'temporal')
        final=fit_candidate(item,'last')
        temporal_rep=head_report(item,temporal['head'],'temporal')
        final_rep=head_report(item,final['head'],'last')
        repeat_rep=head_report(item,final['head'],'repeat_last')
        assert final_rep['heldout_edge_hits']==repeat_rep['heldout_edge_hits']
        rows.append({
            'seed':item['model_seed'],
            'checkpoint_sha256':item['model_checkpoint_sha256'],
            'candidates_temporal':temporal['candidates_examined'],
            'candidates_final':final['candidates_examined'],
            'heads':{
                'temporal_shared_1bit_x4':{**temporal_rep, 'weight':list(temporal['head'][0]),'threshold':temporal['head'][1]},
                'final_1bit':{**final_rep,'weight':list(final['head'][0]),'threshold':final['head'][1]},
                'repeat_final_1bit_x4':repeat_rep,
                'temporal_shared_2bit_x4':head_report(item,temporal['head'],'temporal',2),
                'final_shared_2bit':head_report(item,final['head'],'last',2),
            }
        })
        print(item['model_seed'],'temporal1',temporal_rep['heldout_edge_hits'],'final1',final_rep['heldout_edge_hits'],'temporal2',rows[-1]['heads']['temporal_shared_2bit_x4']['heldout_edge_hits'], 'candidates',temporal['candidates_examined'])
    return {'schema':'COE-04-temporal-quantization-receipt-v1','split':{'fit':list(FIT_CONTEXTS),'holdout':list(TEST_CONTEXTS)},'total_pairs_per_seed':len(TEST_CONTEXTS)*len(EDGES),'rows':rows}


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path(__file__).with_name('frozen_loop_activations.json'))
    ap.add_argument('--out',type=Path,default=Path(__file__).with_name('temporal_receipt.json'))
    args=ap.parse_args()
    import gzip
    source=json.loads(gzip.open(args.input,'rt').read() if args.input.suffix=='.gz' else args.input.read_text())
    out=run_study(source)
    args.out.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')

# COE-04B fair-bit-budget baselines. The functions below are defined after the
# command-line entry for library use; the standalone CLI calls them in the
# separate `run_temporal_baselines.py` runner.

def separation_mask(row_codes):
    mask=0
    index=0
    for row in row_codes:
        for a,b in EDGES:
            if row[a]!=row[b]:mask|=1<<index
            index+=1
    return mask


def candidate_mask(item, head, visit=None):
    w, threshold=head
    rows=[]
    for c in FIT_CONTEXTS:
        rows.append([int(dot(w,item['activations'][f'{a}:{c}'][visit])>threshold) for a in range(N)])
    return separation_mask(rows)


def fit_greedy_equal_bit_baseline(item,k=4,untied=False):
    """Choose k one-bit heads on fixed fit cells; no holdout access."""
    candidates=candidate_projections(item,seed=item['model_seed']+601)
    if untied and k!=T:
        raise ValueError('untied control uses exactly one head per visit')
    if untied:
        choices=[(head,t) for t in range(T) for head in candidates]
    else:
        choices=[(head,T-1) for head in candidates]
    separation=[candidate_mask(item,head,visit) for head,visit in choices]
    covered=0
    selected=[]
    if untied:
        # Visit assignment fixed; choose the best incremental head for t=0..T-1.
        for t in range(T):
            offset=t*len(candidates)
            j=max(range(offset,offset+len(candidates)),key=lambda idx:(separation[idx]&~covered).bit_count())
            selected.append(choices[j]);covered|=separation[j]
    else:
        for _ in range(k):
            j=max(range(len(choices)),key=lambda idx:(separation[idx]&~covered).bit_count())
            selected.append(choices[j]);covered|=separation[j]
    assert len(selected)==k
    result={'heads':selected,'train_coverage':covered.bit_count(),'candidate_count':len(candidates)}
    result.update(score_multi_heads(item,selected))
    return result


def score_multi_heads(item, heads):
    results={}
    for label,contexts in [('fit',FIT_CONTEXTS),('heldout',TEST_CONTEXTS)]:
        obs=[]
        for c in contexts:
            row=[]
            for a in range(N):
                row.append(tuple(int(dot(w,item['activations'][f'{a}:{c}'][visit])>th) for ((w,th),visit) in heads))
            obs.append(row)
        results[label+'_edge_hits']=covered_pairs(obs)
        results[label+'_edge_total']=len(contexts)*len(EDGES)
        results[label+'_unique_per_context']=[len(set(row)) for row in obs]
    return results
