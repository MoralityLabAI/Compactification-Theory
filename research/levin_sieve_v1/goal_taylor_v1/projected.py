"""LSV-17B: project curvature onto each clean goal adjoint before bounding it.

For elementwise GELU, the scalar Hessian bound uses the infinity norm of
its downstream adjoint. For LayerNorm it uses the centered gamma-weighted
adjoint separately for each token, removing a second irrelevant direction.
"""
from fractions import Fraction as Q
from flint import arb,arb_mat
from arb_engine import (TaylorNet,Mix,LN,exact_matrix,up,endpoint,aq,norm2,serialize_upper)


def curvature_data(net,op,G):
    if isinstance(op,Mix):
        x=G.transpose() if op.transpose else G
        p=x*op.W1.transpose()
        mx=max(abs(v).upper() for row in p.tolist() for v in row)
        a=net.op['t0' if op.transpose else 'c0']
        return {'kind':'mix','coefficient':up(net.g2*mx*a*a)}
    if isinstance(op,LN):
        norms=[];d=len(op.gamma)
        for row in G.tolist():
            a=[g*x for g,x in zip(op.gamma,row)]
            mean=sum(a,arb(0))/d
            norms.append(up(sum((abs(x-mean).upper()**2 for x in a),arb(0)).sqrt()))
        return {'kind':'ln','norms':norms}
    raise TypeError('unknown operator')


def goal_curvature(op,data,R):
    if data['kind']=='mix':return data['coefficient']
    d=len(op.gamma);vals=[]
    for n,s in zip(data['norms'],op.std):
        floor=max((s.lower()-R/arb(d).sqrt()).lower(),op.eps.sqrt().lower())
        vals.append(up(2*n/((3*arb(d)).sqrt()*floor*floor)))
    return max(vals)


def prepare(net,stage,cell,steps=3):
    final,tape=net.forward(stage,steps)
    logits=net.logits(final,cell)
    winner=max(range(10),key=lambda j:float(logits[j].mid()))
    if not all(logits[winner].lower()>logits[k].upper() for k in range(10) if k!=winner):
        raise ArithmeticError('uncertified clean winner')
    W=exact_matrix(net.raw.head.weight);pairs={}
    for rival in range(10):
        if rival==winner:continue
        row=[W[winner,j]-W[rival,j] for j in range(48)]
        G=arb_mat([[row[j] if i==cell else arb(0) for j in range(48)] for i in range(81)])
        qs=[];curves=[]
        for op in reversed(tape):
            qs.append(norm2(G));curves.append(curvature_data(net,op,G));G=op.vjp(G)
        rows=G.tolist()
        pairs[rival]={'q':list(reversed(qs)),'curvature':list(reversed(curves)),
            'gradient':G,'margin':(logits[winner]-logits[rival]).lower(),
            'full':up(sum((abs(x).upper() for row in rows for x in row),arb(0))),
            'goal_token':up(sum((abs(x).upper() for x in rows[cell]),arb(0)))}
    return tape,logits,winner,pairs


def evaluate(net,stage,cell,radii,steps=3):
    tape,logits,winner,pairs=prepare(net,stage,cell,steps)
    record={'cell':cell,'winner':winner,'steps_after_anchor':steps,'pairs':[], 'regions':[]}
    for rival,d in pairs.items():
        record['pairs'].append({'rival':rival,'margin_lower':serialize_upper(d['margin']),
           'gradient_l1_full_upper':serialize_upper(d['full']),
           'gradient_l1_patch_upper':serialize_upper(d['goal_token']),
           'adjoint_l2_upper':[serialize_upper(q) for q in d['q']]})
    for mode,dim in [('full',3888),('goal_token',48)]:
        for r in radii:
            trans=net.transport(tape,r,dim)
            out=[]
            for rival,d in pairs.items():
                coeff=[min(up(q*t['b']),goal_curvature(op,c,t['R']))
                       for q,t,op,c in zip(d['q'],trans,tape,d['curvature'])]
                rem=up(sum((h*t['R']**2/2 for h,t in zip(coeff,trans)),arb(0)))
                generic=up(sum((q*t['defect'] for q,t in zip(d['q'],trans)),arb(0)))
                low=(d['margin']-aq(r)*d[mode]-rem).lower()
                out.append({'rival':rival,'goal_hessian_upper':[serialize_upper(h) for h in coeff],
                            'remainder_upper':serialize_upper(rem),'generic_remainder_upper':serialize_upper(generic),
                            'margin_lower':str(endpoint(low))})
            record['regions'].append({'mode':mode,'radius':str(Q(r)),
               'certified':all(Q(x['margin_lower'])>0 for x in out),
               'epsilon_floor_used':any(t['fallback'] for t in trans),
               'layers':[{'name':t['name'],'input_radius_upper':serialize_upper(t['R']),
                 'jacobian_upper':serialize_upper(t['a']),'hessian_upper':serialize_upper(t['b']),
                 'local_defect_upper':serialize_upper(t['defect'])} for t in trans],
               'pairs':out})
    return record,(tape,logits,winner,pairs)


def forward_from_arb(net,X,steps=3):
    for k in range(steps):
        X,_=net.mix(X,'t',True,f'{k}.token')
        X,_=net.ln(X,net.norms[0],f'{k}.ln1')
        X,_=net.mix(X,'c',False,f'{k}.channel')
        X,_=net.ln(X,net.norms[1],f'{k}.ln2')
    return X


def signed_attack(net,stage,cell,prepared,radii=('1/1000','1/100','1/10','1/2')):
    """Find an existential attack using a verified clean gradient direction.

    The direction is frozen before testing radii. Every signed coordinate is
    exactly +r, -r or 0 in real arithmetic. Attack success is verified by Arb.
    """
    _,_,winner,pairs=prepared
    rival=min(pairs,key=lambda k:float(pairs[k]['margin'])/max(float(pairs[k]['full']),1e-100))
    g=pairs[rival]['gradient'].tolist()
    signs=[[-1 if x.lower()>0 else 1 if x.upper()<0 else 0 for x in row] for row in g]
    original=exact_matrix(stage)
    for r in radii:
        delta=arb_mat([[aq(r)*s for s in row] for row in signs])
        final=forward_from_arb(net,original+delta)
        logits=net.logits(final,cell)
        witnesses=[j for j in range(10) if j!=winner and logits[j].lower()>logits[winner].upper()]
        if witnesses:
            j=witnesses[0]
            return {'verified':True,'radius':str(Q(r)),'original_winner':winner,
               'beating_class':j,'positive_margin_lower':str(endpoint((logits[j]-logits[winner]).lower())),
               'targeted_rival':rival,'direction_encoding':''.join('+' if x>0 else '-' if x<0 else '0' for row in signs for x in row)}
    return {'verified':False,'radii_tested':[str(Q(r)) for r in radii],
            'caution':'failed search supplies no robustness lower bound'}
