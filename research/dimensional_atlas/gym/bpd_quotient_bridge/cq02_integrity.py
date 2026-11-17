"""Read-only source-pinned bridge: trained BPD-CQ-02 positive control vs SCM-21.

The studies have different models, tasks, interventions, and effective doses;
this program verifies their receipts and makes no cross-study causal estimate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

BPD_BLOB = 'cf9fdfd8b307d4c3f70fde47185a9d8be623ef51'
SCM_BLOB = 'a765043e0837268d52c0548e26d477ba1e813da9'
SEEDS = (503,509,521,523,541,547,557,563)
SCM_SEEDS=(191,1191,2191)


def git_blob_sha(data:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def audit_native(native:dict)->dict:
    if native.get('schema')!='BPD-CQ-02/trained-factors-cpu/v1':
        raise ValueError('wrong BPD-CQ-02 schema')
    if native.get('source')!='native BPDModel/TiedComponentLinear with trained factors':
        raise ValueError('expected native BPD backend receipt')
    if tuple(native.get('seeds',()))!=SEEDS or len(native.get('rows',[]))!=len(SEEDS):
        raise ValueError('missing or rearranged independent seed blocks')
    methods=['native','projected_native','random_fiber','sham','visible']
    totals={k:0 for k in methods}
    for seed,row in zip(SEEDS,native['rows']):
        if row.get('seed')!=seed or row.get('backend')!='native':raise ValueError('seed/backend mismatch')
        if row.get('native_parity_max_abs') is None or row['native_parity_max_abs']>2e-6:
            raise ValueError('native wrapper parity failed')
        if row.get('component_observable_rank_now')!=1 or row.get('component_observable_rank_after_one')!=2:
            raise ValueError('delayed observability rank witness changed')
        if row.get('four_pass_route')!=128 or row.get('heldout_boards')!=128:
            raise ValueError('heldout phase routing changed')
        if row.get('causal_pairs')!=64 or row['largest_norm_matching_error']>1e-5:
            raise ValueError('native pair count or norm contract changed')
        if row['immediate_median']>.10 or row['delayed_median']<.90:
            raise ValueError('observer/delayed witness changed')
        for k in methods:
            v=row['transfer'][k]
            if type(v) is not int or not 0<=v<=64:raise ValueError('invalid BPD transfer count')
            totals[k]+=v
    if totals!={'native':512,'projected_native':512,'random_fiber':54,'sham':0,'visible':0}:
        raise ValueError('BPD transfer totals changed')
    return {'trained_seeds':len(SEEDS),'pairs':512,'outcomes':totals,
            'current_observable_rank':1,'delayed_observable_rank':2,
            'source_scope':'trained native BPD factors, engineered linear channel-swap transition'}


def audit_scm21(summary:dict)->dict:
    if summary.get('schema')!='scm21.summary.v1':raise ValueError('wrong SCM21 summary')
    rows=[x for x in summary.get('secondary',[]) if x['model']=='causal' and x['visit']==1]
    selected=('native_vpd_write','semantic_diff4','head_readable','full_cls','sham')
    totals={}
    for method in selected:
        hits=[x for x in rows if x['method']==method]
        keys={(x['seed'],x['stratum']) for x in hits}
        expected={(seed,stratum) for seed in SCM_SEEDS for stratum in ('id','ood')}
        if len(hits)!=6 or keys!=expected or any(x['n']!=128 or type(x['correct']) is not int or not 0<=x['correct']<=128 for x in hits):
            raise ValueError('incomplete SCM21 matched snapshot')
        totals[method]={'successes':sum(x['correct'] for x in hits),'n':768,
                        'mean_patch_norm':sum(x['n']*x['mean_patch_norm'] for x in hits)/768}
    expected={'native_vpd_write':0,'semantic_diff4':147,'head_readable':413,'full_cls':432,'sham':0}
    if {k:v['successes'] for k,v in totals.items()}!=expected:
        raise ValueError('SCM21 published outcomes changed')
    return {'seed_blocks':3,'pairs_across_seed_strata':768,'measurements':totals,
            'source_scope':'SCM21 heldout ID/OOD finite-series tasks, recipient-donor interchange on fixed original checkpoints'}


def crosswalk(native:dict,scm:dict)->dict:
    return {'schema':'SCM-BPD-CQ02/read-only-crosswalk-v1',
            'bpd':audit_native(native),'compact':audit_scm21(scm),
            'comparison_scope':'Two nonexchangeable study populations, engineered T vs learned recurrent dynamics, different patch radii and behavioral targets. Direct accuracy/rate comparisons are scientifically invalid.',
            'next_test':'On original trained recurrent checkpoints freeze native BPD factors, task-based hidden readout and intervention doses on calibration data; estimate delayed-response restricted Gramian; confirm matched-norm counterpart effects in independent groups.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bpd-native',required=True,type=Path)
    p.add_argument('--scm21',required=True,type=Path)
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    native=a.bpd_native.read_bytes();scm=a.scm21.read_bytes()
    if a.check and (git_blob_sha(native)!=BPD_BLOB or git_blob_sha(scm)!=SCM_BLOB):
        raise SystemExit('published source blob changed')
    result=crosswalk(json.loads(native),json.loads(scm))
    result['source_blobs']={'bpd_native':git_blob_sha(native),'scm21':git_blob_sha(scm)}
    rendered=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.out:a.out.write_text(rendered)
    else:print(rendered,end='')
if __name__=='__main__':main()
