"""RRC-07 frozen NumPy CPU replay (scalar parity) and exact feedback accounting."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from statistics import mean

try:
    from .rrc07 import (SIZES,PREFIX,HORIZON,SEED,new_roster,roster_hash,
                        raw_features,first_fault,fault_feedback,score_proposal)
    from .rrc05 import exact_program,product_coefficients
    from .rrc06 import roster as previous_roster
    from .fast07 import fast_base,fast_forward
    from .raw_student06 import infer_raw as infer_base
    from .student07 import (portable_forward,propose_int,parameter_count)
except ImportError:
    from rrc07 import (SIZES,PREFIX,HORIZON,SEED,new_roster,roster_hash,
                       raw_features,first_fault,fault_feedback,score_proposal)
    from rrc05 import exact_program,product_coefficients
    from rrc06 import roster as previous_roster
    from fast07 import fast_base,fast_forward
    from raw_student06 import infer_raw as infer_base
    from student07 import (portable_forward,propose_int,parameter_count)

HERE=Path(__file__).resolve().parent
ARMS=('frozen_rrc06','recurrent_2','feedforward','shuffled_recurrent_2')
SPLITS=('dev','unseen_moduli','unseen_cap','local_gap','nonperiodic','collision')
BUDGETS=(0,1,2,4,8,16,32,64)
HASHED_SOURCES=('rrc.py','program04.py','rrc05.py','rrc06.py','raw_student06.py',
                'rrc07.py','student07.py','fast07.py','train07.py','bench07.py',
                'RRC07_PROTOCOL.md','RRC07_THEORY.md')

def digest(p:Path)->str:return sha256(p.read_bytes()).hexdigest()


def load():
    base_path=HERE/'results'/'rrc06_recurrent.json'
    base=json.loads(base_path.read_text())
    if base['schema']!='rrc06.raw_coefficient_student.v1' or base['kind']!='recurrent':
        raise AssertionError('incorrect frozen source')
    models={}
    for kind in ('recurrent','feedforward','shuffled_recurrent'):
        fn=HERE/'results'/f'rrc07_{kind}.json'
        ckpt=json.loads(fn.read_text())
        if ckpt['schema']!='rrc07.raw_iterative_refinement.v1' or ckpt['kind']!=kind:
            raise AssertionError('checkpoint identity mismatch')
        if ckpt['parameter_count']!=parameter_count(kind) or ckpt['exact_residual_at_inference']:
            raise AssertionError('invalid checkpoint model contract')
        if ckpt['frozen_base_sha256']!=digest(base_path) or ckpt['test_generated_in_training']:
            raise AssertionError('base fingerprint or test-blind claim mismatch')
        for path, h in ckpt['source_sha256'].items():
            if digest(HERE/path)!=h:raise AssertionError('training source drift '+path)
        if ckpt['training']['steps']!=240 or ckpt['training']['batch_size']!=64:
            raise AssertionError('training budget mismatch')
        models[kind]=ckpt
    return base,models


def summed(records):
    count=len(records)
    return {'cases':count,
        'a_correct':sum(x['a_correct'] for x in records),
        'a_total':count*PREFIX,
        'mean_first_error_or_65':round(mean((x['first_error'] or 65) for x in records),4),
        'fully_verified_exponent_traces':sum(x['first_residual'] is None for x in records),
        'accepted_programs':sum(x['accepted_program'] for x in records),
        'correct_futures':sum(x['correct_future'] for x in records),
        'false_accepts':sum(x['false_accept'] for x in records)}


def calculate()->dict:
    base,models=load()
    splits=new_roster()
    if {k:len(v) for k,v in splits.items()}!=SIZES:
        raise AssertionError('changed roster size')
    if any(roster_hash(splits[k])!=models['recurrent'][k+'_roster_sha256'] for k in ('train','dev')):
        raise AssertionError('source roster changed')
    seen={x.prefix_hash() for x in splits['train']}
    aliases={k:sum(c.prefix_hash() in seen for c in splits[k]) for k in SPLITS}
    old_train_hashes={x.prefix_hash() for x in previous_roster(include_test=False)['train']}
    old_training_aliases={k:sum(c.prefix_hash() in old_train_hashes for c in splits[k]) for k in SPLITS}
    metrics={name:{} for name in ARMS+('exact_symbolic',)}
    depths={}
    paired_first=[]
    feedback={name:{str(b):[] for b in BUDGETS} for name in
              ('frozen_rrc06','recurrent_2','feedforward')}
    trace=[]
    for split in SPLITS:
        rows={arm:[] for arm in ARMS}
        symbolic=[]
        for c in splits[split]:
            f,base_logits=fast_base(c.prefix,base)
            proposals={'frozen_rrc06':propose_int([row[:3] for row in base_logits])}
            for arm,kind in (('recurrent_2','recurrent'),('feedforward','feedforward'),
                             ('shuffled_recurrent_2','shuffled_recurrent')):
                z=fast_forward(f,base_logits,models[kind])
                proposals[arm]=propose_int(z)
            oracle=exact_program(c.prefix)
            oracle_future=int(oracle is not None and tuple(product_coefficients(HORIZON,oracle))==c.full)
            symbolic.append({'accepted':int(oracle is not None),'correct':oracle_future,
                             'false_accept':int(oracle is not None)-oracle_future})
            row={'id':c.id,'split':split,'prefix_hash':c.prefix_hash(),
                 'symbolic':symbolic[-1]}
            for arm in ARMS:
                rec=score_proposal(c,proposals[arm]);rows[arm].append(rec)
                row[arm]={key:value for key,value in rec.items() if key!='program'}
            if split=='unseen_moduli':
                paired_first.append({arm:rows[arm][-1]['first_error'] or 65 for arm in ARMS})
                for arm in feedback:
                    # Replaying every budget repeats the same deterministic exact
                    # correction history. Compute the complete trajectory once;
                    # its first B edits are identical to a B-budget execution.
                    # Count a fresh terminal verifier query for every budget.
                    fb=fault_feedback(c.prefix,proposals[arm],PREFIX)
                    final=score_proposal(c,fb['proposal'])
                    if final['first_error'] is not None:
                        raise AssertionError('full exact residual feedback failed')
                    edits=fb['edits'];total=len(edits)
                    for budget in BUDGETS:
                        steps=min(budget,total)
                        first_error=edits[steps]['degree'] if steps<total else None
                        feedback[arm][str(budget)].append({
                            'queries':steps+1,'edits':steps,
                            'first_error':first_error,
                            'accepted':final['accepted_program'] if steps==total else 0,
                            'correct_future':final['correct_future'] if steps==total else 0})
                rec_depth={}
                for depth in (0,1,2,4,8):
                    z=fast_forward(f,base_logits,models['recurrent'],passes=depth)
                    rec_depth[str(depth)]=score_proposal(c,propose_int(z))
                depths[c.id]=rec_depth
            trace.append(row)
        metrics['exact_symbolic'][split]={
            'cases':len(symbolic),'accepted_programs':sum(x['accepted'] for x in symbolic),
            'correct_futures':sum(x['correct'] for x in symbolic),
            'false_accepts':sum(x['false_accept'] for x in symbolic)}
        for arm in ARMS: metrics[arm][split]=summed(rows[arm])
    def paired_first_result(left,right):
        delta=[row[left]-row[right] for row in paired_first]
        win=sum(v>0 for v in delta);loss=sum(v<0 for v in delta)
        tie=sum(v==0 for v in delta)
        n=win+loss
        sign_p=min(1.,2*sum(comb(n,k) for k in range(min(win,loss)+1))/(2**n)) if n else 1.
        return {'mean_degree_gain':round(mean(delta),4),'wins':win,'losses':loss,
                'ties':tie,'two_sided_sign_p_descriptive':round(sign_p,6)}
    first_fault_paired={
        'recurrent_vs_frozen':paired_first_result('recurrent_2','frozen_rrc06'),
        'recurrent_vs_feedforward':paired_first_result('recurrent_2','feedforward'),
        'feedforward_vs_frozen':paired_first_result('feedforward','frozen_rrc06')}
    # Exact feedback query budgets and first-error advancement, per arm.
    feedback_scores={}
    for arm,by_budget in feedback.items():
        feedback_scores[arm]={}
        for b,records in by_budget.items():
            feedback_scores[arm][b]={
                'cases':len(records),
                'mean_first_error_or_65':round(mean((r['first_error'] or 65) for r in records),4),
                'accepted_programs':sum(r['accepted'] for r in records),
                'correct_futures':sum(r['correct_future'] for r in records),
                'mean_residual_vector_queries':round(mean(r['queries'] for r in records),4),
                'mean_exact_coordinate_edits':round(mean(r['edits'] for r in records),4)}
    depth_scores={}
    for d in ('0','1','2','4','8'):
        depth_scores[d]=summed([by_depth[d] for by_depth in depths.values()])
    pairs=splits['collision']
    collision_audits=[]
    for i in range(0,len(pairs),2):
        a,b=pairs[i:i+2]
        if a.prefix!=b.prefix or a.full==b.full or a.pair!=b.pair:
            raise AssertionError('collision pair is not indistinguishable')
        first=next(n for n in range(PREFIX+1,HORIZON+1) if a.full[n]!=b.full[n])
        if first!=(67,79,91,107,123)[i//2]:raise AssertionError('late mutation drift')
        # Neural base and all recurrence stages depend only on prefix by construction.
        for case in (a,b):
            if case.prefix!=a.prefix:raise AssertionError('future leakage')
        collision_audits.append({'pair':a.pair,'first_future_difference':first})
    trace_sha=sha256(json.dumps(trace,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'schema':'rrc07.iterative_first_fault.cpu.v1',
            'status':'trained_raw_only_with_separately_charged_exact_feedback',
            'seed':SEED,'prefix_degree':PREFIX,'future_degree':HORIZON,
            'source_sha256':{p:digest(HERE/p) for p in HASHED_SOURCES},
            'checkpoint_sha256':{'rrc06_recurrent':digest(HERE/'results'/'rrc06_recurrent.json'),
                      **{k:digest(HERE/'results'/f'rrc07_{k}.json') for k in models}},
            'roster_sha256':{k:roster_hash(v) for k,v in splits.items()},
            'sizes':{k:len(v) for k,v in splits.items()},
            'aliases_with_rrc07_train':aliases,
            'aliases_with_rrc06_train':old_training_aliases,
            'weights':{k:{'parameters':v['parameter_count'],
                         'fp32_weight_bits':v['fp32_weight_bits'],
                         'hidden_floats':v['fp32_live_hidden'],
                         'training_passes':v['training_passes']} for k,v in models.items()},
            'metrics':metrics,'paired_first_fault':first_fault_paired,
            'recurrent_unseen_moduli_depth':depth_scores,
            'exact_feedback_unseen_moduli':feedback_scores,
            'collision_audits':collision_audits,
            'case_trace_sha256':trace_sha,'case_trace_count':len(trace),
            'selected_cases':[row for row in trace if row['id'] in
                       ('gap2-min1','gap2-min2','late-67-base','late-67-mutant')],
            'claims_boundary':'Known families, fresh held-out instances; no finite theorem proof; exact feedback supplies full residuals, at O(T^2) per query.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path,default=HERE/'results'/'rrc07_pilot.json')
    args=p.parse_args()
    result=calculate()
    value=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=value:
            raise SystemExit('RRC-07 source/checkpoint-bound replay differs from frozen receipt')
        print('RRC-07 source/checkpoint-bound NumPy CPU replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(value)
        print(json.dumps({'receipt':str(args.out),
                    'unseen_model':{k:v['unseen_moduli'] for k,v in result['metrics'].items()},
                    'depth':result['recurrent_unseen_moduli_depth'],
                    'feedback64':{k:v['64'] for k,v in result['exact_feedback_unseen_moduli'].items()}},indent=2,sort_keys=True))

if __name__=='__main__':main()
