"""RRC-08 source-bound CPU replay, exact certificates, and fair resource ledger."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

try:
    from .operator08 import (PREFIX,HORIZON,Operator08,roster,roster_hash,
        serial_circuit,parallel_circuit,parallel_snapshots,score,score_trace)
    from .rrc05 import (exact_program,product_coefficients)
    from .rrc import euler_exponents
    from .rrc06 import roster as previous_roster06
    from .rrc07 import new_roster as previous_roster07
except ImportError:
    from operator08 import (PREFIX,HORIZON,Operator08,roster,roster_hash,
        serial_circuit,parallel_circuit,parallel_snapshots,score,score_trace)
    from rrc05 import (exact_program,product_coefficients)
    from rrc import euler_exponents
    from rrc06 import roster as previous_roster06
    from rrc07 import new_roster as previous_roster07

HERE=Path(__file__).resolve().parent
KINDS=('operator','local','shuffled')


def source_hashes():
    paths=['operator08.py','train08.py','bench08.py','rrc05.py','rrc.py','program04.py',
           'rrc06.py','rrc07.py']
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in paths}


def weights():
    src=source_hashes()
    result={}
    for kind in KINDS:
        path=HERE/'results'/f'rrc08_{kind}_weights.json'
        obj=json.loads(path.read_text())
        if obj['operator08_sha256']!=src['operator08.py'] or obj['train08_sha256']!=src['train08.py']:
            raise AssertionError(f'{kind}: source drift or checkpoint mismatch')
        if obj['kind'] not in ('compiled_recurrent_operator','raw_local_feedforward','label_shuffled_operator'):
            raise ValueError('unrecognized model type')
        result[kind]=(obj,sha256(path.read_bytes()).hexdigest())
    return result


def _rounded(v:float) -> int:
    if not math.isfinite(v):raise ValueError('invalid model prediction')
    return int(round(v))


def feedforward(case, checkpoint):
    """Eight FP64 weights, current and two previous raw f values only."""
    f=case.prefix
    wb=checkpoint['b_fit']['weights'];wa=checkpoint['a_fit']['weights']
    b=[0];a=[0]
    for n in range(1,len(f)):
        fb=(n*f[n],f[n],f[n-1],1)
        fa=(f[n],f[n-1],f[n-2] if n>1 else 0,1)
        b.append(_rounded(sum(x*y for x,y in zip(wb,fb))))
        a.append(_rounded(sum(x*y for x,y in zip(wa,fa))))
    return b,a


def reduce_scores(rows):
    if not rows:return {}
    total=len(rows)
    out={'cases':total,'exact_euler_coordinates':sum(r['exact_coordinates'] for r in rows),
         'total_euler_coordinates':total*PREFIX,
         'verified_trace':sum(r['verified_trace'] for r in rows),
         'accepted_program':sum(r['accepted_program'] for r in rows),
         'correct_future':sum(r['correct_future'] for r in rows),
         'false_accept':sum(r['false_accept'] for r in rows),
         'nonintegral':sum(1-r['integral'] for r in rows),
         'first_error_mean':sum((r['first_error'] if r['first_error'] is not None else PREFIX+1)
                                for r in rows)/total}
    return out


def run():
    saved=weights();models={kind:content for kind,(content,_) in saved.items()}
    choices=models['operator']['decoded_coefficients']
    if choices is None:raise AssertionError('trained operator failed compilation gate')
    circuit=Operator08(tuple(choices[:4]),tuple(choices[4:]))
    data=roster()
    train_signatures={tuple(c.prefix) for c in data['train']}
    alias={k:sum(tuple(c.prefix) in train_signatures for c in cases)
           for k,cases in data.items() if k!='train'}
    # Historical train-prefix aliases reveal the scope of longitudinal blindness.
    history={'RRC06':{tuple(c.prefix) for c in previous_roster06(include_test=False)['train']},
             'RRC07':{tuple(c.prefix) for c in previous_roster07(include_test=False)['train']}}
    previous_aliases={name:{k:sum(c.prefix in known for c in cases)
                            for k,cases in data.items() if k!='train'}
                      for name,known in history.items()}
    parsed={}
    all_cases=[c for group,cases in data.items() if group!='train' for c in cases]
    splits=[k for k in data if k!='train']
    for kind in ('compiled_recurrent_operator','raw_local_feedforward','symbolic_reference'):
        groups={}
        for label in splits:
            local=[]
            for c in data[label]:
                if kind=='compiled_recurrent_operator':
                    result=score(c,circuit)
                elif kind=='raw_local_feedforward':
                    result=score_trace(c,feedforward(c,models['local']))
                else:
                    a=euler_exponents(c.prefix)
                    # Distinct independent symbolic baseline, not learned program execution.
                    prog=exact_program(c.prefix)
                    right=int(prog is not None and tuple(product_coefficients(HORIZON,prog))==c.full)
                    result={'integral':1,'exact_coordinates':PREFIX,'first_error':None,
                            'verified_trace':1,'accepted_program':int(prog is not None),
                            'correct_future':right,'false_accept':int(prog is not None)-right}
                local.append(result)
            groups[label]=reduce_scores(local)
        all_splits=[v for label in splits for v in (groups[label],)]
        aggregate={key:sum(part[key] for part in all_splits) for key in (
            'cases','exact_euler_coordinates','total_euler_coordinates','verified_trace',
            'accepted_program','correct_future','false_accept','nonintegral')}
        aggregate['first_error_mean_weighted']=sum(groups[k]['first_error_mean']*len(data[k])
                                              for k in splits)/sum(len(data[k]) for k in splits)
        parsed[kind]={'aggregate':aggregate,'splits':groups}
    if models['shuffled']['executable']:
        sh=models['shuffled']['decoded_coefficients']
        ctr=Operator08(tuple(sh[:4]),tuple(sh[4:]))
        negative=[score(c,ctr) for c in all_cases]
        shuffled={'executable':True,'scores':reduce_scores(negative)}
    else:shuffled={'executable':False,'rejection':'signed integer operator snap gate failed'}
    # Fixed parallel depth, one cached simultaneous trajectory per case.
    # Identical arithmetic to separate calls, verified by an independent test.
    depths=(0,1,2,4,8,16,32,65)
    parallel={str(d):{'euler_coordinates_correct':0,
                      'total':len(data['unseen_period'])*PREFIX,
                      'complete_traces':0,'max_first_error_degree':0}
              for d in depths}
    for c in data['unseen_period']:
        snapshots=parallel_snapshots(c.prefix,circuit,depths)
        target=euler_exponents(c.prefix)
        for depth,(_,a) in snapshots.items():
            r=parallel[str(depth)]
            r['euler_coordinates_correct']+=sum(a[n]==target[n] for n in range(1,PREFIX+1))
            r['complete_traces']+=int(all(a[n]==target[n] for n in range(1,PREFIX+1)))
            r['max_first_error_degree']=max(r['max_first_error_degree'],
                    next((n for n in range(1,PREFIX+1) if a[n]!=target[n]),PREFIX+1))
    pairs=data['collision']
    assert all(a.prefix==b.prefix and a.full!=b.full for a,b in zip(pairs[::2],pairs[1::2]))
    groups_seen={}
    for a,b in zip(pairs[::2],pairs[1::2]):
        groups_seen[a.pair]={'same_visible_input':a.prefix==b.prefix,
             'same_candidate':serial_circuit(a.prefix,circuit)==serial_circuit(b.prefix,circuit),
             'diverges_future':a.full!=b.full}
    hashes=source_hashes()
    return {
        'schema':'rrc08.receipt.v1',
        'status':'bounded_teacher_forced_operator_induction_cpu',
        'source_sha256':hashes,
        'checkpoint_sha256':{kind:raw_sha for kind,(_,raw_sha) in saved.items()},
        'parameters':{'prefix_degree':PREFIX,'heldout_degree':HORIZON,
                      'supervised_training_degree':16,'seed':2026110908,
                      'operator_parameter_count':8,'feedforward_parameter_count':8,
                      'compiled_operator_syntax_bits_excluding_decoder':circuit.syntax_bits,
                      'fixed_public_basis':[
                          'n*f[n]','convolution_of_prior_b_and_f','previous_b','constant',
                          'b[n]/n','proper_divisor_sum_of_prior_a_over_n','previous_a','constant']},
        'learned_program':circuit.as_dict(),
        'train_roster_sha256':roster_hash(data['train']),
        'roster_sha256':{k:roster_hash(v) for k,v in data.items()},
        'training_prefix_aliases':alias,
        'previous_training_prefix_aliases':previous_aliases,
        'scores':parsed,
        'shuffled_label_control':shuffled,
        'parallel_depth':parallel,
        'collision_pairs':groups_seen,
        'claims_boundary':'Exactness requires the supplied convolution/divisor feature basis, local teacher traces and integer DSL interpreter; no claim of end-to-end neural symbolic discovery or novel Rogers-Ramanujan proof.',
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'results'/'rrc08_pilot.json')
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    result=run()
    rendered=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=rendered:
            raise SystemExit('RRC-08 receipt missing or stale')
        print('RRC-08 frozen exact CPU replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(rendered)
        print(json.dumps({'file':str(args.out),
            'scores':{k:v['aggregate'] for k,v in result['scores'].items()},
            'parallel':result['parallel_depth'],'alias':result['training_prefix_aliases']},indent=2))

if __name__=='__main__':main()
