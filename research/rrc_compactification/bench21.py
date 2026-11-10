"""RRC-21: selective exact class ranking and fully charged symbolic deferral.

Two distinct experiments:
  budget12: fixed 12 full two-demo verifications, even after a fallback;
  selective_full: each deferred case gets the complete certified class search,
                  which uses a larger case-dependent budget up to 33.
The OOD gate is evaluated *before* neural inference and can skip all model MACs.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

try:
    from . import select21 as d, risk21 as r, train21 as tr, bench19, learn18 as l
    from . import gate20 as mixture, model18 as model, topology16 as top
except ImportError:
    import select21 as d
    import risk21 as r
    import train21 as tr
    import bench19
    import learn18 as l
    import gate20 as mixture
    import model18 as model
    import topology16 as top

HERE=Path(__file__).resolve().parent
SOURCES=('select21.py','risk21.py','train21.py','bench21.py','family19.py',
         'family20.py','gate20.py','model18.py','learn18.py','certify18.py',
         'topology16.py','topology15.py','graph11.py','graph12.py','graph13.py',
         'graph14.py','quotient17.py','bench19.py','train19.py')
ARMS=('neural12','mass12','hybrid6_6','selective12','selective_full',
      'ood_only_full','random_full','full_certified')
MODEL_MAC=l.infer_cost('recurrent')*2+l.infer_cost('feedforward')


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCES}


def checkpoints():
    models,_,past=bench19.checkpoints()
    subset={role:models[role] for role in tr.MODEL_ROLES}
    source=tr.source_hashes()
    path=HERE/'results'/'rrc21_calibration.json'
    calibr=json.loads(path.read_text())
    if calibr['source_sha256']!=source or calibr['base_model_sha256']!={k:past[k] for k in tr.MODEL_ROLES}:
        raise AssertionError('RRC-21 calibration or base model source drift')
    if calibr['schema']!='rrc21.calibration.v1' or calibr['neural_updates']!=0:
        raise AssertionError('invalid calibration provenance')
    if calibr['reference_sha256']!=d.digest(d.dataset(include_test=False)['reference']):
        raise AssertionError('reference roster changed')
    if calibr['calibration_sha256']!=d.digest(d.dataset(include_test=False)['calibration']):
        raise AssertionError('calibration roster changed')
    return subset,calibr,{**{k:past[k] for k in tr.MODEL_ROLES},'rrc21_calibration':sha256(path.read_bytes()).hexdigest()}


def neural_logits(case,models):
    return {k:model.logits(case,models[k]) for k in tr.MODEL_ROLES}


def join_order(first,second,budget=12):
    seen=set();output=[]
    for graph in tuple(first)+tuple(second):
        if graph not in seen:
            seen.add(graph);output.append(graph)
        if len(output)==budget:break
    return tuple(output)


def visible_hash_uniform(case):
    h=sha256(json.dumps(case.visible,separators=(',',':')).encode()).digest()
    return int.from_bytes(h[:8],'big')/2**64


def router(case,calibration,models,classes,*,logits=None):
    """Decision with short-circuit: OOD rejection uses zero neural inference."""
    distance,p=r.calibrated_distance(calibration,case)
    if p<=calibration['ood_alpha']:
        return {'fallback':True,'reason':'ood','pvalue':p,'distance':distance,
                'entropy':None,'macs':0,'logits':None}
    if logits is None: logits=neural_logits(case,models)
    entropy=r.entropy(case,logits,classes)
    defer,reason=r.decision(calibration,case,p,entropy)
    return {'fallback':defer,'reason':reason,'pvalue':p,'distance':distance,
            'entropy':entropy,'macs':MODEL_MAC,'logits':logits}


def choose(case,arm,classes,logits,decision,calibration):
    mass=l.mass_order(case.topology,classes)
    if arm=='full_certified':return mass,len(mass),'full'
    if arm=='mass12':return mass,d.BUDGET,'mass'
    if arm=='ood_only_full':
        if decision['pvalue']<=calibration['ood_alpha']:return mass,len(mass),'ood'
        if logits is None:raise AssertionError('missing neural inference')
        return r.order(case,logits,classes),d.BUDGET,'neural'
    if arm=='random_full':
        if visible_hash_uniform(case)<calibration['calibration_defer_rate']:
            return mass,len(mass),'random'
        return r.order(case,logits,classes),d.BUDGET,'neural'
    if arm in ('selective12','selective_full') and decision['fallback']:
        return mass,(d.BUDGET if arm=='selective12' else len(mass)),decision['reason']
    if logits is None:raise AssertionError('missing neural scores')
    neural=r.order(case,logits,classes)
    if arm in ('selective12','selective_full','neural12'):
        return neural,d.BUDGET,'neural'
    if arm=='hybrid6_6':
        # Six learned proposals, six *new* certified prior representatives.
        return join_order(neural[:6],mass,d.BUDGET),d.BUDGET,'hybrid'
    raise ValueError('unsupported arm')


def auc(known,novel):
    if not known or not novel:raise ValueError('two nonempty cohorts required')
    return sum(1. if b>a else 0.5 if b==a else 0. for a in known for b in novel)/(len(known)*len(novel))


def aggregate(rows):
    n=len(rows)
    return {'cases':n,'accepted':sum(x['accepted'] for x in rows),
            'future_correct':sum(x['future_correct'] for x in rows),
            'false_future_acceptance':sum(x['false_accept'] for x in rows),
            'abstentions':n-sum(x['accepted'] for x in rows),
            'deferred_cases':sum(x['deferred'] for x in rows),
            'ood_deferred':sum(x['reason']=='ood' for x in rows),
            'uncertainty_deferred':sum(x['reason']=='entropy' for x in rows),
            'neural_selected_cases':sum(x['reason']=='neural' for x in rows),
            'neural_selected_verified':sum(x['accepted'] for x in rows if x['reason']=='neural'),
            'neural_selected_future_correct':sum(x['future_correct'] for x in rows if x['reason']=='neural'),
            'mean_queries':round(sum(x['queries'] for x in rows)/n,6),
            'mean_integer_loop_proxy':round(sum(x['integer_loop_proxy'] for x in rows)/n,6),
            'mean_fp_macs':round(sum(x['model_macs'] for x in rows)/n,6),
            'mean_ood_distance_proxy':round(sum(x['ood_distance_proxy'] for x in rows)/n,6),
            'reason_counts':{k:sum(x['reason']==k for x in rows) for k in
                             ('neural','ood','entropy','mass','hybrid','random','full')}}


def run():
    models,calibration,digests=checkpoints()
    groups=d.dataset(include_test=True)
    cls=l.class_map()
    reference={c.visible for c in groups['reference']}
    cal={c.visible for c in groups['calibration']}
    for key,cases in groups.items():
        if key in ('reference','calibration'):continue
        if any(c.visible in reference or c.visible in cal for c in cases):
            raise AssertionError('reference/calibration input leaked to final test')
    results={};traces={};ood_vals={};decisions={};case_details={};pairs={}
    for key,cases in groups.items():
        if key in ('reference','calibration'):continue
        rows={arm:[] for arm in ARMS}
        signatures={arm:[] for arm in ARMS}
        ood_vals[key]=[];decisions[key]=[];case_details[key]=[]
        for case in cases:
            # Shared computation for benchmark; charge each policy as a separate
            # deployment. The selective router independently supports zero-MAC
            # OOD short-circuit and is unit-tested with patched neural inference.
            logits=neural_logits(case,models)
            response=router(case,calibration,models,cls,logits=logits)
            pval=response['pvalue'];uncertainty=response['entropy']
            if response['reason']=='ood':
                # No neural forward pass is needed in the deployed selector;
                # the benchmark computed logits for other arms only.
                assert response['macs']==0
            ood_vals[key].append(response['distance'])
            decisions[key].append((case.identifier,response['fallback'],response['reason'],pval))
            case_details[key].append({'id':case.identifier,'pvalue':pval,'uncertainty':uncertainty,
                                      'route':response['reason'],'fallback':response['fallback']})
            for arm in ARMS:
                order,budget,route=choose(case,arm,cls,logits,response,calibration)
                if len(order)!=len(set(order)) or budget>len(cls[case.topology]['representatives']):
                    raise AssertionError('duplicate/noncertified class proposals')
                pred=l.search(case,order,budget=budget)
                trace=pred['trace']
                if arm not in ('selective_full','ood_only_full','random_full','full_certified') and len(trace)>d.BUDGET:
                    raise AssertionError('fixed budget exceeded')
                if pred['integer_loop_proxy']!=sum(2*top.graph_cost(gr,case.topology) for gr in trace):
                    raise AssertionError('arithmetic verification ledger mismatch')
                grade=top.certificate(case,pred['graph'])
                if pred['graph'] is not None and not grade['accepted']:
                    raise AssertionError('unchecked graph acceptance')
                if arm in ('mass12','full_certified'):macs=0
                elif arm=='selective_full' or arm=='selective12':macs=response['macs']
                elif arm=='ood_only_full':macs=0 if route=='ood' else MODEL_MAC
                elif arm=='random_full':macs=0 if route=='random' else MODEL_MAC
                else:macs=MODEL_MAC
                base={'queries':len(trace),'integer_loop_proxy':pred['integer_loop_proxy'],
                      'model_macs':macs,'ood_distance_proxy':r.cheap_work_proxy(calibration) if arm in
                      ('selective12','selective_full','ood_only_full') else 0,
                      'deferred':int(arm in ('selective12','selective_full') and route!='neural'
                                      or arm=='ood_only_full' and route=='ood'
                                      or arm=='random_full' and route=='random'),
                      'reason':route}
                rows[arm].append({**grade,**base})
                signatures[arm].append((case.identifier,[list(q) for q in trace],
                                        None if pred['graph'] is None else list(pred['graph'])))
        results[key]={arm:aggregate(row) for arm,row in rows.items()}
        traces[key]={arm:sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
                     for arm,v in signatures.items()}
        if key=='collision':
            pairs={arm:{'identical_trace':all(signatures[arm][j][1:]==signatures[arm][j+1][1:]
                         for j in range(0,len(cases),2)),
                         'false_future_acceptance':results[key][arm]['false_future_acceptance']}
                   for arm in ARMS}
        if key not in ('negative','collision') and results[key]['full_certified']['accepted']!=len(cases):
            raise AssertionError('complete class reference missed in-class case')
    if results['negative']['full_certified']['accepted']!=0:
        raise AssertionError('full reference missed out-of-class rejection')
    if not all(x['identical_trace'] for x in pairs.values()):
        raise AssertionError('future leakage in paired collision comparison')
    if pairs['full_certified']['false_future_acceptance']!=d.COLLISION_PAIRS:
        raise AssertionError('future prefix uncertainty mismatch')
    groups_all={'known':d.KNOWN,'novel':d.NOVEL,'positive':d.KNOWN+d.NOVEL}
    summary={name:{arm:{'cases':sum(results[k][arm]['cases'] for k in vals),
                        'accepted':sum(results[k][arm]['accepted'] for k in vals),
                        'future_correct':sum(results[k][arm]['future_correct'] for k in vals),
                        'deferred':sum(results[k][arm]['deferred_cases'] for k in vals),
                        'exact_queries':sum(results[k][arm]['mean_queries']*len(groups[k]) for k in vals)}
                   for arm in ARMS} for name,vals in groups_all.items()}
    known=[x for key in d.KNOWN for x in ood_vals[key]]
    novel=[x for key in d.NOVEL for x in ood_vals[key]]
    out={'schema':'rrc21.receipt.v1','status':'cpu_selective_known_only_calibration_exploratory',
         'source_sha256':source_hashes(),'checkpoint_sha256':digests,
         'roster_sha256':{key:d.digest(cases) for key,cases in groups.items()},
         'calibration':{'known_only':True,'alpha':calibration['ood_alpha'],
                        'entropy_threshold':calibration['entropy_threshold'],
                        'calibration_defer_rate':calibration['calibration_defer_rate'],
                        'calibration_defer_count':calibration['calibration_defer_count'],
                        'reference_cases':calibration['reference_count'],
                        'calibration_cases':calibration['calibration_count'],
                        'calibration_sha256':calibration['calibration_sha256']},
         'parameters':{'seed':d.SEED,'prefix_degree':d.PREFIX,'forecast_degree':d.HORIZON,
                       'strict_budget':d.BUDGET,'full_symbolic_max_budget':max(len(cls[t]['representatives']) for t in cls),
                       'frozen_experts':list(tr.MODEL_ROLES),
                       'experts_fp32_parameters':(6240,6240,6235),
                       'inference_model_mac_proxy':MODEL_MAC,
                       'per_case_ood_nominal_scalar_proxy':r.cheap_work_proxy(calibration),
                       'public_symbolic_normalizations':len(cls)*64,
                       'neural_calibration_updates':0},
         'ood_detection':{'known_n':len(known),'novel_n':len(novel),
                          'auroc_novel_vs_known':round(auc(known,novel),6),
                          'rates':{k:{'cases':len(v),'ood_flagged':sum(row[2]=='ood' for row in decisions[k]),
                                       'deferred_total':sum(row[1] for row in decisions[k])}
                                   for k,v in decisions.items()}},
         'scores':results,'summaries':summary,'trace_sha256':traces,'collision_pairs':pairs,
         'input_decisions_sha256':{k:sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
                                    for k,v in decisions.items()},
         'limits':'Known-source p-value is superuniform only under exchangeability; entropy cutoff is descriptive. Exact finite degree-12 certificates do not entail correct future degree-32 predictions; deferred full class search has a larger query budget. Full interpreters, prior model training, symbolic compilation, calibration distance and separate neural MACs all incur costs.'}
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=HERE/'results'/'rrc21_pilot.json')
    args=parser.parse_args()
    payload=json.dumps(run(),indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=payload:
            raise SystemExit('RRC-21 complete replay or source/checkpoint hash mismatch')
        print('RRC-21 frozen selective exact CPU receipt verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
        r=json.loads(payload)
        for family in d.KNOWN+d.NOVEL:
            print(family,{'ood':r['ood_detection']['rates'][family],
                           'scores':{k:r['scores'][family][k]['future_correct'] for k in ARMS}})
        print('totals',json.dumps({k:{a:(v['future_correct'],v['deferred']) for a,v in vals.items()}
                                    for k,vals in r['summaries'].items()},indent=2))
        print('AUROC',r['ood_detection']['auroc_novel_vs_known'])

if __name__=='__main__':main()
