"""SMC-07: puzzle-clustered analysis of frozen pretrained SAE receipts.

All three SAE seeds are averaged within each heldout puzzle *before*
resampling, preserving 16 independent test clusters. Reports paired
differences with descriptive percentile intervals. No trained fit uses
these results. The sign-flip assumptions needed for inferential p-values
are not asserted.
"""
import argparse
import json
import random
from pathlib import Path

EXPECTED_CHECKPOINT='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'
EXPECTED_ROSTER='eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611'


def avg(data):
    return sum(data)/len(data)


def percentile(values,q):
    sorted_values=sorted(values)
    idx=(len(values)-1)*q
    lo=int(idx)
    high=min(lo+1,len(values)-1)
    a=idx-lo
    return sorted_values[lo]*(1-a)+sorted_values[high]*a


def bootstrap(values,draws=20000,seed=20261115):
    """Percentile interval over IID puzzle clusters, with replacement."""
    if not values:
        raise ValueError('no puzzle observations')
    rng=random.Random(seed)
    n=len(values)
    estimates=[sum(values[rng.randrange(n)] for _ in range(n))/n
               for _ in range(draws)]
    return [percentile(estimates,0.025),percentile(estimates,0.975)]


def analyze(receipt,draws=20000):
    if receipt['schema']!='SMC-07/source-locked-real-recurrence-sae-v1':
        raise ValueError('unexpected schema')
    if receipt['checkpoint_sha256']!=EXPECTED_CHECKPOINT or receipt['roster_sha256']!=EXPECTED_ROSTER:
        raise ValueError('source provenance mismatch')
    if receipt['n_puzzles'] != 64 or set(receipt['train_seed_list']) != {71,89,113}:
        raise ValueError('full 64-puzzle, three-seed receipt required')
    ids=tuple(receipt['split_ids']['test'])
    if len(ids)!=16 or len(set(ids))!=16:
        raise ValueError('expected 16 independent test puzzles')
    runs=receipt['results']
    if len(runs)!=3:
        raise ValueError('three SAE runs required')
    by_puzzle={}
    for run in runs:
        seed=run['seed']
        raw=run['causal_goal_replay']['per_puzzle']
        transport=run['feature_transport']['per_puzzle']
        if {p['puzzle_id'] for p in raw} != set(ids):
            raise ValueError('heldout causal per-puzzle coverage mismatch')
        if {p['puzzle_id'] for p in transport} != set(ids):
            raise ValueError('heldout depth transport coverage mismatch')
        for p in raw:
            by_puzzle.setdefault(p['puzzle_id'],{})[seed]={'causal':p}
        for p in transport:
            by_puzzle[p['puzzle_id']][seed]['transport']=p
    if any(len(d)!=3 for d in by_puzzle.values()):
        raise ValueError('seed-coverage error')
    output={
        'schema':'SMC-07/source-locked-clustered-analysis-v1',
        'checkpoint_sha256':EXPECTED_CHECKPOINT,'roster_sha256':EXPECTED_ROSTER,
        'test_puzzle_ids':list(ids),'seeds':sorted(r['seed'] for r in runs),
        'bootstrap_draws':draws,
        'units':'averages across 3 SAE seeds per puzzle, then resample 16 puzzle IDs',
        'per_seed':[],
        'paired_cluster_differences':{},
    }
    for r in runs:
        recon=r['reconstruction_by_depth']['1']
        g=r['causal_goal_replay']['aggregates']
        output['per_seed'].append({
          'seed':r['seed'],
          'sae_mse':recon['sae_mse_normalized'],
          'pca4_mse':recon['pca4_mse_normalized'],
          'random4_mse':recon['random4_mse_normalized'],
          'active_0p2_mean_goal_effect':g['0.2_sae_active']['mean_goal_logit_delta_norm'],
          'inactive_0p2_mean_goal_effect':g['0.2_sae_inactive']['mean_goal_logit_delta_norm'],
          'isotropic_0p2_mean_goal_effect':g['0.2_isotropic']['mean_goal_logit_delta_norm'],
          'remote_0p2_mean_goal_effect':g['0.2_same_direction_remote_token']['mean_goal_logit_delta_norm'],
          'sae_goal_reconstruction':g['sae_reconstruction'],
          'pca4_goal_reconstruction':g['pca4_reconstruction'],
          'gauge_max_logit_linf':max(x['gauge_logit_linf'] for x in r['causal_goal_replay']['gauge_checks_fp64']),
        })
    for radius in ('0.2','0.8'):
        for control in ('sae_inactive','isotropic','same_direction_remote_token'):
            per_puzzle=[]
            for pid in ids:
                diff=avg([
                    by_puzzle[pid][seed]['causal']['interventions'][radius]['sae_active']['goal_logit_delta_norm'] -
                    by_puzzle[pid][seed]['causal']['interventions'][radius][control]['goal_logit_delta_norm']
                    for seed in output['seeds']])
                per_puzzle.append(diff)
            output['paired_cluster_differences'][radius+'_sae_active_minus_'+control]={
                'mean':avg(per_puzzle),
                'percentile_interval_95':bootstrap(per_puzzle,draws,20261115+len(control)+int(float(radius)*100)),
                'puzzles_positive':sum(v>0 for v in per_puzzle),
                'puzzles_negative':sum(v<0 for v in per_puzzle),
                'per_puzzle':dict(zip(map(str,ids),per_puzzle))
            }
    feature_diffs=[avg([
         by_puzzle[pid][seed]['transport']['same_token_overlap']-
         by_puzzle[pid][seed]['transport']['shuffled_puzzle_overlap']
         for seed in output['seeds']])
         for pid in ids]
    output['paired_cluster_differences']['depth1_to2_same_minus_shuffled_top4']={
        'mean':avg(feature_diffs),
        'percentile_interval_95':bootstrap(feature_diffs,draws,20261215),
        'puzzles_positive':sum(v>0 for v in feature_diffs),
        'per_puzzle':dict(zip(map(str,ids),feature_diffs))
    }
    return output


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--receipt',type=Path,required=True)
    arg.add_argument('--out',type=Path,default=Path('runs/smc07/clustered.json'))
    arg.add_argument('--bootstrap',type=int,default=20000)
    a=arg.parse_args()
    analyzed=analyze(json.loads(a.receipt.read_text()),draws=a.bootstrap)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(analyzed,indent=2,sort_keys=True)+'\n')
    print(json.dumps({
        'receipt':str(a.receipt),'out':str(a.out),
        'per_seed':analyzed['per_seed'],
        'paired_cluster_differences':{
            k:{field:v[field] for field in ('mean','percentile_interval_95','puzzles_positive')}
            for k,v in analyzed['paired_cluster_differences'].items()
        }
    },indent=2))


if __name__=='__main__':main()
