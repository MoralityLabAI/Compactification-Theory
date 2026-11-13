"""OIC-21 lightweight independent reference/receipt audit, standard library only.

Checks the committed source/file SHA-256 pins, frozen training metadata, exact
Bayesian oracle numerators and aggregate population scores. It does NOT prove
that optimizer execution produced the archived fitted weights. The optional
CPU PyTorch replay reproduces learned sensor policy decisions from weights.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json
from pathlib import Path
from oic21_exact import ExactActiveOracle
from oic21_policy import fixed_multisensor_accuracy,evaluate_policy,greedy_policy
from oic20_markers import identifies_every_shift

HERE=Path(__file__).resolve().parent
RES=HERE/'results'


def check(cond,message):
    if not cond:raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def read_compressed(path):
    raw=base64.b64decode(Path(path).read_bytes().strip(),validate=True)
    return json.loads(gzip.decompress(raw))


def report(silent=False):
    manifest=read_json(RES/'oic21_manifest.json')
    for name,digest in manifest['sha256'].items():
        path=HERE/name
        check(path.is_file(),f'absent pinned file: {name}')
        check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,
              f'file hash mismatch: {name}')
    archived=read_compressed(RES/'oic21_fits.json.gz.b64')
    fits=archived['fits'];bench=read_json(RES/'oic21_bench.json');faults=read_json(RES/'oic21_faults.json')
    check(len(fits)==10,'expected ten training fits')
    check(archived['training_tag']=='OIC-21-ORACLE-DISTILL-01','training protocol tag')
    check({(x['metadata']['seed'],x['metadata']['split']) for x in fits}==
          {(i,split) for i in range(5) for split in ('full','heldout')},'training seed/split coverage')
    for fit in fits:
        metadata=fit['metadata'];check(metadata['parameter_count']==145,'model architecture drift')
        check(metadata['training_steps']==500 and metadata['batch_size']==96,'training budget drift')
        check(metadata['learning_rate']==0.01,'training lr drift')
        check(len(fit['evaluation'])==9,'phase count coverage per fit')
        for n in (4,5,6):
            check(str(n) in metadata['teacher'],'teacher phase count missing')
        ns=[x['n'] for x in fit['evaluation']]
        check(ns==[4,5,6,7,8,12,16,24,32],'wrong evaluation phase counts')
        for record in fit['evaluation']:
            n=record['n']
            check(record['horizon']==n+2 and record['output_bits']==n+2,'incorrect query budget')
            check(record['total_link_bits']==(n+2)*(1+(n-1).bit_length()),'unbilled address bits')
            check(record['leaves']<=n*n and 1/n<=record['shift_accuracy']<=1,'invalid label metric')
    check(len(bench['results'])==9,'bench phase count')
    check(bench['n_train']==[4,5,6] and bench['seeds']==list(range(5)),'bench training contract')
    for row in bench['results']:
        n=row['n'];Q=n+2
        check(row['query_budget']==Q,'benchmark query count')
        check(identifies_every_shift(n,tuple(row['best_multi_sensor_markers'])),
              'OIC-20 sensor set fails distribution identification')
        multi=fixed_multisensor_accuracy(n,tuple(row['best_multi_sensor_markers']),n)
        check(multi['accuracy']==row['simultaneous_sensor_baseline']['accuracy'],
              'single run OIC-20 distribution/control mismatch')
        greedy=evaluate_policy(n,Q,greedy_policy)
        check(abs(greedy['shift_accuracy']-row['greedy']['shift_accuracy'])<1e-12,
              'greedy oracle replay mismatch')
        for split in ('full','heldout'):
            expected=sorted((x for x in fits if x['metadata']['split']==split),key=lambda x:x['metadata']['seed'])
            stored=[next(e for e in f['evaluation'] if e['n']==n) for f in expected]
            acc=[r['shift_accuracy'] for r in stored]
            shown=row['learned'][split]
            check(all(abs(a-b['shift_accuracy'])<1e-12 for a,b in zip(acc,shown['runs'])),
                  'fit and summary individual score mismatch')
            check(abs(shown['mean_shift_accuracy']-sum(acc)/5)<1e-12,
                  'fit and summary mean mismatch')
        if n in (4,5,6):
            o=ExactActiveOracle(n,Q)
            check(row['optimal_adaptive']['expected_success_numerator']==o.best_adaptive(o.all,0),
                  'exact adaptive oracle mismatch')
    expected={4:(15,13),5:(19,17),6:(22,18)}
    for n,(ad,fx) in expected.items():
        o=ExactActiveOracle(n,5)
        check(o.best_adaptive(o.all,0)==ad,'five-query adaptive numerator')
        check(o.best_fixed()[1]==fx,'five-query nonadaptive numerator')
    check(len(faults['results'])==30,'fault panel coverage')
    check({(r['n'],r['split'],r['seed']) for r in faults['results']}==
          {(n,sp,s) for n in (4,6,8) for sp in ('full','heldout') for s in range(5)},
          'fault seed/split coverage')
    for r in faults['results']:
        n=r['n'];records={x['kind']:x for x in r['panels']}
        check(set(records)=={'clean','flip','erase'},'missing fault type')
        check(records['clean']['episodes']==n*n,'incorrect clean cases')
        for k in ('flip','erase'):
            check(records[k]['episodes']==n*n*(n+2),'missing fault/phase cases')
        for x in records.values():
            check(x['correct']<=x['episodes'] and
                  abs(x['accuracy']-x['correct']/x['episodes'])<1e-12,'fault aggregate wrong')
    for a in faults['aggregates']:
        rs=[r for r in faults['results'] if r['n']==a['n'] and r['split']==a['split']]
        for kindrow in a['kinds']:
            kind=kindrow['kind'];sub=[next(x for x in r['panels'] if x['kind']==kind) for r in rs]
            check(kindrow['episodes']==sum(z['episodes'] for z in sub),'aggregate fault denominator')
            check(kindrow['correct']==sum(z['correct'] for z in sub),'aggregate fault numerator')
    result={'status':'OIC-21 frozen evidence integrity and exact reference PASS',
            'trained_fits':len(fits),'evaluated_phase_counts':[x['n'] for x in bench['results']],
            'fault_panels':len(faults['results']),
            'fault_episodes':sum(a['episodes'] for r in faults['results'] for a in r['panels']),
            'five_query_adaptive_success':[v[0] for v in expected.values()]}
    if not silent:print(json.dumps(result,sort_keys=True,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');parser.parse_args()
    report()

if __name__=='__main__':main()
