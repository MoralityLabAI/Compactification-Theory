"""LSV-17 signed-adjoint Taylor transport in validated Arb arithmetic.

Scope: exact-real erf-GELU / population-LayerNorm network with exact stored
binary weights and exact binary stage-one anchors. All margins use all rivals.
The complete Hessian is never formed. Floating SVD only proposes a bound;
Arb LDL positivity is what certifies the spectral bound.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
import sys

from flint import arb, arb_mat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretrained_trm'))
from validated_minor import exact_matrix


def endpoint(x):
    m,e=x.man_exp(); m,e=int(m),int(e)
    return Q(m*(2**e),1) if e>=0 else Q(m,2**(-e))


def aq(x):
    q=Q(x)
    return arb(q.numerator)/q.denominator


def up(x):
    if not x.is_finite():
        raise ArithmeticError('nonfinite enclosure')
    return x.upper()


def norm2(M):
    return up(sum((abs(x).upper()**2 for row in M.tolist() for x in row),arb(0)).sqrt())


def hadamard(A,B):
    return arb_mat([[a*b for a,b in zip(ar,br)] for ar,br in zip(A.tolist(),B.tolist())])


def verified_spectral(weight):
    """Return exact dyadic t with ||W||_2^2 < t, checked by Arb LDL.

    The positive pivots prove tI-W^T W positive definite; use WW^T if smaller.
    Failure of a proposed bound causes conservative doubling and rechecking.
    """
    import torch,math
    W=exact_matrix(weight)
    G=W.transpose()*W if W.ncols()<=W.nrows() else W*W.transpose()
    n=G.nrows()
    proposed=float(torch.linalg.matrix_norm(weight.detach().double(),ord=2))**2
    t=Q(max(1,math.ceil((proposed*1.001+1e-12)*(2**24))),2**24)
    for _ in range(20):
        C=[[aq(t)*int(i==j)-G[i,j] for j in range(n)] for i in range(n)]
        L=[[arb(0) for _ in range(n)] for _ in range(n)]
        piv=[]; valid=True
        for i in range(n):
            p=C[i][i]-sum((L[i][k]**2*piv[k] for k in range(i)),arb(0))
            if not p.lower()>0:
                valid=False;break
            piv.append(p);L[i][i]=arb(1)
            for j in range(i+1,n):
                L[j][i]=(C[j][i]-sum((L[j][k]*L[i][k]*piv[k] for k in range(i)),arb(0)))/p
        if valid:
            return up(aq(t).sqrt()),{'squared_upper':str(t),'gram_order':n,
                'ldl_pivot_lower_min':str(min(endpoint(p.lower()) for p in piv))}
        t*=2
    raise ArithmeticError('could not verify spectral bound')


def gelu_value_slope(X):
    rt2=arb(2).sqrt(); rt2pi=(2*arb.pi()).sqrt()
    val=[];der=[]
    for row in X.tolist():
        vr=[];dr=[]
        for x in row:
            phi=(1+(x/rt2).erf())/2
            vr.append(x*phi)
            dr.append(phi+x*(-x*x/2).exp()/rt2pi)
        val.append(vr);der.append(dr)
    return arb_mat(val),arb_mat(der)


@dataclass
class Mix:
    W0:arb_mat
    W1:arb_mat
    slope:arb_mat
    transpose:bool
    a:arb
    b:arb
    name:str

    def vjp(self,G):
        x=G.transpose() if self.transpose else G
        out=x+hadamard(x*self.W1.transpose(),self.slope)*self.W0.transpose()
        return out.transpose() if self.transpose else out

    def bounds(self,R):
        return self.a,self.b,False


@dataclass
class LN:
    centered:list
    std:list
    gamma:list
    eps:arb
    name:str

    def vjp(self,G):
        d=len(self.gamma);out=[]
        for grow,u,s in zip(G.tolist(),self.centered,self.std):
            a=[v*g for v,g in zip(grow,self.gamma)]
            avg=sum(a,arb(0))/d
            dot=sum((v*z for v,z in zip(a,u)),arb(0))
            out.append([(a[j]-avg)/s-u[j]*dot/(d*s**3) for j in range(d)])
        return arb_mat(out)

    def bounds(self,R):
        d=len(self.gamma)
        rootd=arb(d).sqrt()
        clean=min(s.lower() for s in self.std)
        floor=(clean-R/rootd).lower()
        absolute=self.eps.sqrt().lower()
        fallback=floor<absolute
        floor=max(floor,absolute)
        g=max(abs(v).upper() for v in self.gamma)
        return up(g/floor),up(2*g/((3*arb(d)).sqrt()*floor**2)),fallback


class TaylorNet:
    def __init__(self,model):
        block=model.block
        self.raw=model
        self.W={};self.bias={};self.op={};self.spectral={}
        for name,linear,nrows in (
            ('t0',block.tok_mix[0],48),('t1',block.tok_mix[2],48),
            ('c0',block.chan_mix[0],81),('c1',block.chan_mix[2],81)):
            self.W[name]=exact_matrix(linear.weight).transpose()
            b=[arb(float(x)) for x in linear.bias.detach().cpu().tolist()]
            self.bias[name]=arb_mat([b[:] for _ in range(nrows)])
            self.op[name],self.spectral[name]=verified_spectral(linear.weight)
        x=arb(2).sqrt()
        self.g1=up((1+(x/arb(2).sqrt()).erf())/2+x*(-x*x/2).exp()/(2*arb.pi()).sqrt())
        self.g2=up((2/arb.pi()).sqrt())
        self.norms=[block.ln1,block.ln2]

    def mix(self,X,prefix,transpose,name):
        V=X.transpose() if transpose else X
        W0,W1=self.W[prefix+'0'],self.W[prefix+'1']
        g,slope=gelu_value_slope(V*W0+self.bias[prefix+'0'])
        Y=V+g*W1+self.bias[prefix+'1']
        a=up(1+self.op[prefix+'1']*self.g1*self.op[prefix+'0'])
        b=up(self.op[prefix+'1']*self.g2*self.op[prefix+'0']**2)
        return (Y.transpose() if transpose else Y),Mix(W0,W1,slope,transpose,a,b,name)

    def ln(self,X,norm,name):
        gamma=[arb(float(x)) for x in norm.weight.detach().cpu().tolist()]
        beta=[arb(float(x)) for x in norm.bias.detach().cpu().tolist()]
        eps=arb(float(norm.eps));d=len(gamma)
        us=[];ss=[];out=[]
        for row in X.tolist():
            mean=sum(row,arb(0))/d
            u=[v-mean for v in row]
            s=(eps+sum((v*v for v in u),arb(0))/d).sqrt()
            if not s.lower()>0:raise ArithmeticError('uncertain clean denominator')
            us.append(u);ss.append(s)
            out.append([g*v/s+b for g,v,b in zip(gamma,u,beta)])
        return arb_mat(out),LN(us,ss,gamma,eps,name)

    def forward(self,stage,steps=3):
        X=exact_matrix(stage);tape=[]
        for k in range(steps):
            X,op=self.mix(X,'t',True,f'{k}.token');tape.append(op)
            X,op=self.ln(X,self.norms[0],f'{k}.ln1');tape.append(op)
            X,op=self.mix(X,'c',False,f'{k}.channel');tape.append(op)
            X,op=self.ln(X,self.norms[1],f'{k}.ln2');tape.append(op)
        return X,tape

    def logits(self,X,cell):
        w=exact_matrix(self.raw.head.weight)
        b=[arb(float(x)) for x in self.raw.head.bias.detach().cpu().tolist()]
        return [sum((w[k,j]*X[cell,j] for j in range(48)),b[k]) for k in range(10)]

    def margin_adjoint(self,tape,cell,winner,rival):
        w=exact_matrix(self.raw.head.weight)
        row=[w[winner,j]-w[rival,j] for j in range(48)]
        G=arb_mat([[row[j] if i==cell else arb(0) for j in range(48)] for i in range(81)])
        qs=[]
        for op in reversed(tape):
            qs.append(norm2(G))
            G=op.vjp(G)
        rows=G.tolist()
        full=up(sum((abs(x).upper() for row in rows for x in row),arb(0)))
        patch=up(sum((abs(x).upper() for x in rows[cell]),arb(0)))
        return {'q':list(reversed(qs)),'g_full':full,'g_patch':patch,'gradient':G}

    def transport(self,tape,radius,dimensions):
        r=aq(radius)
        R=up(arb(dimensions).sqrt()*r)
        rec=[]
        for op in tape:
            a,b,fallback=op.bounds(R)
            defect=up(b*R*R/2)
            rec.append({'name':op.name,'R':R,'a':a,'b':b,
                        'defect':defect,'fallback':fallback})
            R=up(a*R)
        return rec


def serialize_upper(x):
    return str(endpoint(x.upper()))


def evaluate_case(net,stage,cell,radii,steps=3):
    final,tape=net.forward(stage,steps)
    logits=net.logits(final,cell)
    winner=max(range(10),key=lambda j:float(logits[j].mid()))
    if not all(logits[winner].lower()>logits[k].upper() for k in range(10) if k!=winner):
        raise ArithmeticError('clean argmax not verified')
    adjoints={k:net.margin_adjoint(tape,cell,winner,k) for k in range(10) if k!=winner}
    output={'cell':cell,'winner':winner,'goal':'first missing cell, all nine rivals',
            'steps_after_anchor':steps,'pairs':[],'regions':[]}
    for k,data in adjoints.items():
        output['pairs'].append({'rival':k,'margin_lower':str(endpoint((logits[winner]-logits[k]).lower())),
            'gradient_l1_full_upper':serialize_upper(data['g_full']),
            'gradient_l1_patch_upper':serialize_upper(data['g_patch']),
            'adjoint_l2_upper':[serialize_upper(q) for q in data['q']]})
    for mode,dim,gkey in [('full',3888,'g_full'),('goal_token',48,'g_patch')]:
        for r in radii:
            trans=net.transport(tape,r,dim)
            losses=[]
            for k,data in adjoints.items():
                rem=up(sum((q*t['defect'] for q,t in zip(data['q'],trans)),arb(0)))
                margin=logits[winner]-logits[k]-aq(r)*data[gkey]-rem
                losses.append({'rival':k,'remainder_upper':serialize_upper(rem),
                               'margin_lower':str(endpoint(margin.lower()))})
            output['regions'].append({'mode':mode,'radius':str(Q(r)),
                'certified':all(Q(x['margin_lower'])>0 for x in losses),
                'epsilon_floor_used':any(t['fallback'] for t in trans),
                'layers':[{'name':t['name'],'input_radius_upper':serialize_upper(t['R']),
                    'jacobian_upper':serialize_upper(t['a']),'hessian_upper':serialize_upper(t['b']),
                    'local_defect_upper':serialize_upper(t['defect'])} for t in trans],
                'pairs':losses})
    return output
