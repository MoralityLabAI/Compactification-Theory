"""Read-only audit linking published SCM21 patch directions to BPD quotient theory.

No checkpoint, gold, model inference, training or sealed held-out access. This
file scores public summary metadata only. Interpret direction and dose jointly.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'scm21_conservative_causal/results/summary.json'
BLOB_SHA='a765043e0837268d52c0548e26d477ba1e813da9'
METHODS=('sham','full_cls','head_readable','head_fiber','pca4','semantic_diff4','sae_group4','native_write4','native_vpd_write','random_fiber4')
EXPECT={'sham':0,'full_cls':432,'head_readable':413,'head_fiber':1,'pca4':101,
        'semantic_diff4':147,'sae_group4':32,'native_write4':0,'native_vpd_write':0,'random_fiber4':0}
EXPECTED_SEEDS=(191,1191,2191)
EXPECTED_STRATA=('id','ood')


def git_blob_sha(data:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def summarize(summary:dict)->dict:
    if summary.get('schema')!='scm21.summary.v1':raise ValueError('SCM21 source schema mismatch')
    rows=[r for r in summary['secondary'] if r['model']=='causal' and r['visit']==1]
    counts={m:[] for m in METHODS};seen=set();block_samples={}
    for r in rows:
        tag=(r['seed'],r['stratum'],r['method'])
        if tag in seen:raise ValueError('duplicate SCM21 method/seed/stratum')
        seen.add(tag)
        if r['method'] not in counts or r['seed'] not in EXPECTED_SEEDS or r['stratum'] not in EXPECTED_STRATA:
            raise ValueError('unregistered method/seed/stratum')
        if not (isinstance(r['correct'],int) and isinstance(r['n'],int) and
                r['n']==128 and 0<=r['correct']<=r['n']):
            raise ValueError('invalid causal counts')
        counts[r['method']].append(r)
        block_samples[(r['seed'],r['stratum'])]=r['n']
    if len(seen)!=60 or len(block_samples)!=6 or any(len(v)!=6 for v in counts.values()):
        raise ValueError('incomplete 3x2x10 visit-two factorial')
    values={}
    for m,v in counts.items():
        n=sum(x['n'] for x in v)
        k=sum(x['correct'] for x in v)
        norm=sum(x['mean_patch_norm']*x['n'] for x in v)/n
        values[m]={'n':n,'correct':k,'fidelity':k/n,'weighted_mean_patch_norm':norm,
                   'by_seed_stratum':[{'seed':r['seed'],'stratum':r['stratum'],'n':r['n'],
                                     'correct':r['correct'],'mean_patch_norm':r['mean_patch_norm']}
                                      for r in sorted(v,key=lambda x:(x['seed'],x['stratum']))]}
    reference=values['native_vpd_write']
    diff=values['semantic_diff4']
    return {'schema':'scm21_bpd_causal_quotient_bridge_v1',
            'source':'scm21.summary.v1, merged Compactification-Theory PR #207',
            'visit_zero_based':1,'common_eligible_pairs_per_seed_stratum':128,
            'independent_seed_blocks':3,'repeated_task_strata_per_seed':2,
            'methods':values,
            'dose_ratio_difference_pca_to_native':diff['weighted_mean_patch_norm']/reference['weighted_mean_patch_norm'],
            'minimum_difference_pca_only_successes_from_marginals':max(0,diff['correct']-reference['correct']),
            'all_six_native_causal_blocks_zero':all(x['correct']==0 for x in counts['native_vpd_write']),
            'interpretation':'retrospective public summary; unnormalized direction-vs-dose confounding; no trained BPD causal inference'}


def strict_check(result:dict):
    if {m:r['correct'] for m,r in result['methods'].items()}!=EXPECT:
        raise ValueError('SCM21 published counts changed')
    if any(v['n']!=768 for v in result['methods'].values()):
        raise ValueError('SCM21 paired denominator changed')
    if result['minimum_difference_pca_only_successes_from_marginals']!=147:
        raise ValueError('minimum-only count changed')
    if not result['all_six_native_causal_blocks_zero']:
        raise ValueError('native write main result changed')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,default=DEFAULT)
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args();raw=a.summary.read_bytes();sha=git_blob_sha(raw)
    if a.check and sha!=BLOB_SHA:raise SystemExit('published SCM21 summary Git blob changed')
    result=summarize(json.loads(raw));result['source_git_blob_sha']=sha
    if a.check:strict_check(result)
    output=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(output,encoding='utf-8')
    else:print(output)

if __name__=='__main__':main()
