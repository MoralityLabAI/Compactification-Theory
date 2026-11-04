"""Exact-rational replay of LSV-17 exported scalar proof inequalities.

Arb/source recomputation remains responsible for clean derivative enclosures
and uniform local operator bounds. This independent layer detects corrupted
scalar arithmetic, missing rival classes and falsely reported successes.
"""
from fractions import Fraction as Q
from pathlib import Path
import argparse,json


def require(c,message):
    if not c:raise ValueError(message)


def check_case(case, projected=False):
    winner=case['winner']
    require(type(winner) is int and 0<=winner<10,'bad winner')
    require(case['steps_after_anchor']==3,'wrong horizon')
    expected=set(range(10))-{winner}
    pairs={p['rival']:p for p in case['pairs']}
    require(len(case['pairs'])==9 and set(pairs)==expected,'missing/duplicate clean rivals')
    seen=set();verified=[]
    for region in case['regions']:
        mode=region['mode']; r=Q(region['radius'])
        require(mode in ('full','goal_token') and 0<r<=1,'bad intervention')
        require((mode,r) not in seen,'duplicate region');seen.add((mode,r))
        dim=3888 if mode=='full' else 48
        layers=region['layers']
        require(len(layers)==12,'wrong number of macro layers')
        previous=None
        for i,l in enumerate(layers):
            R,a,b,defect=(Q(l[k]) for k in ('input_radius_upper','jacobian_upper','hessian_upper','local_defect_upper'))
            require(min(R,a,b,defect)>=0,'negative bound')
            if i==0:require(R*R>=dim*r*r,'initial radius too small')
            else:require(R>=previous,'propagated radius too small')
            require(defect>=b*R*R/2,'understated local Taylor defect')
            previous=a*R
        observed={p['rival']:p for p in region['pairs']}
        require(len(region['pairs'])==9 and set(observed)==expected,'missing/duplicate final rivals')
        good=[]
        for rival in sorted(expected):
            p=pairs[rival];v=observed[rival]
            m=Q(p['margin_lower'])
            g=Q(p['gradient_l1_full_upper' if mode=='full' else 'gradient_l1_patch_upper'])
            qs=[Q(x) for x in p['adjoint_l2_upper']]
            require(m>0 and g>=0 and len(qs)==12 and min(qs)>=0,'invalid clean derivatives')
            rem=Q(v['remainder_upper']);low=Q(v['margin_lower'])
            if projected:
                hs=[Q(x) for x in v['goal_hessian_upper']]
                require(len(hs)==12 and min(hs)>=0,'invalid projected curvature')
                needed=sum((h*Q(l['input_radius_upper'])**2/2 for h,l in zip(hs,layers)),Q(0))
                generic=sum((q*Q(l['local_defect_upper']) for q,l in zip(qs,layers)),Q(0))
                require(Q(v['generic_remainder_upper'])>=generic,'invalid generic comparison')
            else:
                needed=sum((q*Q(l['local_defect_upper']) for q,l in zip(qs,layers)),Q(0))
            require(rem>=needed,'understated final remainder')
            require(low<=m-r*g-rem,'overstated final margin')
            good.append(low>0)
        cert=all(good)
        require(type(region['certified']) is bool and region['certified']==cert,'false success flag')
        verified.append((mode,str(r),cert))
    return verified


def replay(data,complete=False):
    require(data['schema'] in ('LSV-17/signed-adjoint-taylor-v1','LSV-17B/projected-curvature-v1'),'wrong schema')
    projected=data['schema']=='LSV-17B/projected-curvature-v1'
    require(data['checkpoint_sha256']=='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1','wrong checkpoint')
    require(data['roster_sha256']=='eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611','wrong roster')
    cs=data['cases']
    require(len(cs)==data['n_cases'] and len(cs)>0,'wrong case count')
    require([c['case_index'] for c in cs]==list(range(len(cs))),'bad case indices')
    if complete:require(len(cs)==64,'complete roster required')
    summary={}
    template=None
    for c in cs:
        regions=check_case(c,projected=projected)
        keys={(m,r) for m,r,_ in regions}
        if template is None:template=keys
        require(keys==template,'case-dependent region selection')
        for mode,r,cert in regions:
            k=(mode,r)
            summary.setdefault(k,[0,0])
            summary[k][0]+=1;summary[k][1]+=int(cert)
    for s in data['summaries']:
        key=(s['mode'],str(Q(s['radius'])))
        require(key in summary,'unexpected summary')
        require(summary[key]==[s['cases'],s['certified']],'summary mismatch')
    require(len(data['summaries'])==len(summary),'missing summaries')
    require(len({(s['mode'],str(Q(s['radius']))) for s in data['summaries']})==len(summary),'duplicate summaries')
    return [{'mode':m,'radius':r,'cases':n,'certified':c} for (m,r),(n,c) in sorted(summary.items())]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--complete',action='store_true')
    a=p.parse_args();print(json.dumps(replay(json.loads(a.receipt.read_text()),a.complete),indent=2))
