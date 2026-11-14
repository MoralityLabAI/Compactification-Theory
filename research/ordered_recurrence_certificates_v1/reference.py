"""OVC-01: exact ordered formula verification and recurrent cut certificates.

Implements the rational jet operators printed in OpenAI Math #116.
The universal formula-size guarantee remains conditional on that paper.
Nonzero evaluations independently certify inequality. Zero evaluations
below the declared bound provide no universal identity certificate.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction as F
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def var(i):
    if type(i) is not int or i<1: raise ValueError('positive variable index required')
    return ('v',i)


def const(c): return ('c',F(c))
def add(a,b): return ('+',a,b)
def mul(a,b): return ('*',a,b)
def sub(a,b): return add(a,mul(const(-1),b))


def shape(f):
    """Tree size includes every occurrence, even when Python objects alias."""
    if not isinstance(f,tuple) or not f: raise ValueError('invalid expression')
    tag=f[0]
    if tag=='v' and len(f)==2:
        var(f[1]); return (1,f[1],1)
    if tag=='c' and len(f)==2:
        F(f[1]); return (1,0,0)
    if tag in ('+','*') and len(f)==3:
        a,b=shape(f[1]),shape(f[2])
        return (a[0]+b[0]+1,max(a[1],b[1]),
                max(a[2],b[2]) if tag=='+' else a[2]+b[2])
    raise ValueError('unknown expression node')


def polynomial(f,limit=100000):
    """Independent exact word expansion; explicit support budget."""
    shape(f); tag=f[0]
    if tag=='c': return {():F(f[1])} if f[1] else {}
    if tag=='v': return {(f[1],):F(1)}
    a,b=polynomial(f[1],limit),polynomial(f[2],limit)
    out=defaultdict(F)
    if tag=='+':
        for p in (a,b):
            for w,c in p.items(): out[w]+=c
    else:
        if len(a)*len(b)>limit: raise ValueError('word expansion budget exceeded')
        for u,c in a.items():
            for v,d in b.items(): out[u+v]+=c*d
    if len(out)>limit: raise ValueError('word support budget exceeded')
    return {w:c for w,c in out.items() if c}


def scalar_eval(f,assignment):
    tag=f[0]
    if tag=='c': return F(f[1])
    if tag=='v': return F(assignment[f[1]])
    a,b=scalar_eval(f[1],assignment),scalar_eval(f[2],assignment)
    return a+b if tag=='+' else a*b


def jet_entry(i,r,c):
    """Source #116 T_i[r,c] for zero-based Taylor coordinates."""
    if type(i) is not int or i<1 or min(r,c)<0: raise ValueError('invalid jet entry')
    return F((-1)**(r-c-1),r*i**(r-c)) if c<r else F()


def jet_apply(i,values):
    """O(d) exact action of J_i(a)=integral a(z)/(z+i) dz.

    If b=a/(z+i), then b_j=(a_j-b_(j-1))/i. The retained
    integral coefficient at degree j+1 is b_j/(j+1).
    """
    var(i)
    if not values: raise ValueError('nonempty jet required')
    out=[F()]*len(values); previous=F()
    for j in range(len(values)-1):
        previous=(F(values[j])-previous)/i
        out[j+1]=previous/(j+1)
    return tuple(out)


def dense_jet_apply(i,values):
    return tuple(sum((jet_entry(i,r,c)*v for c,v in enumerate(values)),F())
                 for r in range(len(values)))


def evaluate_jet(f,values):
    tag=f[0]
    if tag=='c': return tuple(F(f[1])*v for v in values)
    if tag=='v': return jet_apply(f[1],values)
    if tag=='+': return tuple(a+b for a,b in zip(evaluate_jet(f[1],values),evaluate_jet(f[2],values)))
    if tag=='*': return evaluate_jet(f[1],evaluate_jet(f[2],values))
    raise ValueError('invalid expression tag')


def word(w):
    if not w: return const(1)
    f=var(w[0])
    for i in w[1:]: f=mul(f,var(i))
    return f


def compare(left,right,dimension=None,max_dimension=10000):
    diff=sub(left,right); size,r,degree=shape(diff); r=max(1,r)
    bound=2*r*size*size
    d=bound if dimension is None else dimension
    if type(d) is not int or not 1<=d<=max_dimension: raise ValueError('jet dimension outside execution budget')
    unit=(F(1),)+(F(),)*(d-1)
    column=evaluate_jet(diff,unit)
    witness=next((j for j,c in enumerate(column) if c),None)
    exact=polynomial(diff)
    if witness is not None and not exact: raise AssertionError('nonzero jet on exact zero polynomial')
    if d>=bound and exact and witness is None: raise AssertionError('bounded source hitting claim failed on finite case')
    return {'difference_tree_size':size,'variables_bound':r,'formal_degree_upper':degree,
            'jet_dimension':d,'source_formula_dimension_bound':bound,
            'full_source_dimension_used':d>=bound,
            'nonzero_first_column':witness is not None,
            'witness_coefficient_index':witness,
            'witness_coefficient':None if witness is None else str(column[witness]),
            'word_oracle_equal':not exact,
            'zero_probe_is_universal_certificate':False,
            'zero_probe_source_conditional':witness is None and d>=bound,
            'max_numerator_bits':max(abs(x.numerator).bit_length() for x in column),
            'max_denominator_bits':max(x.denominator.bit_length() for x in column)}


def leading_word_coefficients(w):
    if not w or any(type(i) is not int or i<1 for i in w): raise ValueError('nonempty positive word required')
    k=len(w); leading=F(1,math.factorial(k)*math.prod(w))
    next_coefficient=-leading*sum((F(k-j,i) for j,i in enumerate(w)),F())/(k+1)
    return leading,next_coefficient


def adjacent_certificate(w,j):
    if type(j) is not int or not 0<=j<len(w)-1: raise ValueError('invalid adjacent location')
    if w[j]==w[j+1]: raise ValueError('adjacent symbols must differ')
    altered=list(w); altered[j],altered[j+1]=altered[j+1],altered[j]
    k=len(w); coeff,_=leading_word_coefficients(w)
    expected=coeff*(F(1,w[j+1])-F(1,w[j]))/(k+1)
    column=evaluate_jet(sub(word(w),word(altered)),(F(1),)+(F(),)*(k+1))
    if any(column[:k+1]) or column[k+1]!=expected or not expected:
        raise AssertionError('adjacent-swap coefficient formula failed')
    return expected


def finite_shift_action(f,vector,degree):
    """Truncated free-word shift representation, without dense allocation.

    T_i e_w=e_(i,w) until length degree. Evaluating on e_empty
    recovers every coefficient through this degree. Its full basis has
    1+r+...+r^degree entries, exponential in the degree for r>1.
    """
    if type(degree) is not int or degree<0: raise ValueError('nonnegative degree required')
    tag=f[0]
    if tag=='c': return {w:F(f[1])*c for w,c in vector.items() if F(f[1])*c}
    if tag=='v': return {(f[1],)+w:c for w,c in vector.items() if len(w)<degree and c}
    if tag=='*': return finite_shift_action(f[1],finite_shift_action(f[2],vector,degree),degree)
    if tag=='+':
        out=defaultdict(F)
        for side in (f[1],f[2]):
            for w,c in finite_shift_action(side,vector,degree).items(): out[w]+=c
        return {w:c for w,c in out.items() if c}
    raise ValueError('invalid formula')


def fixtures():
    a,b,c=var(1),var(2),var(3)
    valid=[('multiply_associativity',mul(mul(a,b),c),mul(a,mul(b,c))),
           ('left_distribution',mul(a,add(b,c)),add(mul(a,b),mul(a,c))),
           ('right_distribution',mul(add(a,b),c),add(mul(a,c),mul(b,c))),
           ('scalar_move',mul(const(2),mul(a,b)),mul(mul(const(2),a),b)),
           ('cancellation',sub(a,a),const(0)),
           ('sum_associativity',add(add(a,b),c),add(a,add(b,c)))]
    invalid=[('swap_ab',mul(a,b),mul(b,a)),
             ('reverse_abc',mul(mul(a,b),c),mul(mul(c,b),a)),
             ('move_sum_across_a',mul(a,add(b,c)),mul(add(b,c),a)),
             ('commutative_square',mul(add(a,b),add(a,b)),
               add(add(mul(a,a),mul(const(2),mul(a,b))),mul(b,b))),
             ('collect_a',mul(mul(a,b),a),mul(mul(a,a),b)),
             ('swap_two_pairs',add(mul(a,b),mul(b,c)),add(mul(b,a),mul(c,b)))]
    return [(name,x,y,True) for name,x,y in valid]+[(name,x,y,False) for name,x,y in invalid]


def imm_words(n,limit=20000):
    """Free ordered IMM path polynomial, start/end vertex zero."""
    if type(n) is not int or not 2<=n<=7 or n**(n-1)>limit: raise ValueError('IMM finite support budget')
    out={}
    for middle in itertools.product(range(n),repeat=n-1):
        nodes=(0,)+middle+(0,)
        symbols=tuple(t*n*n+nodes[t]*n+nodes[t+1]+1 for t in range(n))
        out[symbols]=F(1)
    if len(out)!=n**(n-1): raise AssertionError('IMM path words collided')
    return out


def cut_certificate(n,k):
    if type(k) is not int or not 1<=k<n: raise ValueError('internal cut required')
    P=imm_words(n)
    prefixes=[];suffixes=[]
    for j in range(n):
        prefix_nodes=(0,)*k+(j,)
        suffix_nodes=(j,)+(0,)*(n-k)
        prefixes.append(tuple(t*n*n+prefix_nodes[t]*n+prefix_nodes[t+1]+1 for t in range(k)))
        suffixes.append(tuple((k+t)*n*n+suffix_nodes[t]*n+suffix_nodes[t+1]+1 for t in range(n-k)))
    matrix=[[P.get(u+v,F()) for v in suffixes] for u in prefixes]
    if matrix!=[[F(i==j) for j in range(n)] for i in range(n)]: raise AssertionError('cut coefficient minor not identity')
    return {'n':n,'cut':k,'identity_minor_order':n,'minimum_linear_cut_width':n,
            'path_monomials':len(P),'bilinear_summands_upper':n}


def imm_jet_column(n,dimension):
    """Backward operator recurrence computes the first output column.

    Multiplication order is preserved. Reversing evaluation order follows
    matrix action on columns; it does not reverse the source word.
    """
    if type(n) is not int or not 2<=n<=6 or type(dimension) is not int or not 1<=dimension<=2000:
        raise ValueError('finite IMM jet execution budget')
    zero=(F(),)*dimension
    unit=(F(1),)+(F(),)*(dimension-1)
    columns=[unit]+[zero]*(n-1)
    for t in reversed(range(n)):
        next_columns=[]
        for i in range(n):
            total=[F()]*dimension
            for j in range(n):
                image=jet_apply(t*n*n+i*n+j+1,columns[j])
                total=[a+b for a,b in zip(total,image)]
            next_columns.append(tuple(total))
        columns=next_columns
    return columns[0]


def polynomial_jet_column(P,dimension):
    total=[F()]*dimension
    unit=(F(1),)+(F(),)*(dimension-1)
    for w,coefficient in P.items():
        image=evaluate_jet(word(w),unit)
        total=[a+coefficient*b for a,b in zip(total,image)]
    return tuple(total)


def abp_probe_budget(n,candidate_tree_size):
    if type(n) is not int or n<2 or type(candidate_tree_size) is not int or candidate_tree_size<1:
        raise ValueError('n>=2 and positive formula size required')
    r=n**3; target_states=n*(n-1)+2
    difference_states=target_states+2*candidate_tree_size
    dimension=(r-1)*math.comb(difference_states,2)+difference_states
    return {'variables':r,'target_ABP_states':target_states,
            'candidate_tree_size':candidate_tree_size,'difference_ABP_states_upper':difference_states,
            'source_path_program_dimension_upper':dimension,
            'universal_guarantee_conditional_on_116':True,
            'candidate_DAG_size_is_not_formula_tree_size':True}


def dense_row_product(matrices):
    if not matrices: raise ValueError('nonempty matrix stream required')
    n=len(matrices[0]); row=[F(int(j==0)) for j in range(n)]
    for matrix in matrices:
        if len(matrix)!=n or any(len(r)!=n for r in matrix): raise ValueError('matrix shape mismatch')
        row=[sum((row[i]*F(matrix[i][j]) for i in range(n)),F()) for j in range(n)]
    return row[0]


def numeric_path_product(matrices):
    n=len(matrices[0]); L=len(matrices)
    return sum((math.prod(F(matrices[t][nodes[t]][nodes[t+1]]) for t in range(L))
                for middle in itertools.product(range(n),repeat=L-1)
                for nodes in [(0,)+middle+(0,)]),F())


def report():
    rows=[]
    for name,left,right,equal in fixtures():
        r=max(shape(left)[1],shape(right)[1]); scalar_trials=0
        for vals in itertools.product(range(-2,3),repeat=r):
            assignment=dict(enumerate(vals,1)); scalar_trials+=1
            if scalar_eval(left,assignment)!=scalar_eval(right,assignment):
                raise AssertionError('fixture unexpectedly distinguishable by scalar assignment')
        row=compare(left,right,dimension=12)
        if row['word_oracle_equal']!=equal or row['nonzero_first_column']==equal:
            raise AssertionError('declared rewrite fixture failed')
        row.update(name=name,correct_identity=equal,scalar_trials=scalar_trials,
                   scalar_trials_all_equal=True)
        rows.append(row)
    full=[compare(var(1),var(1)),compare(mul(var(1),var(2)),mul(var(2),var(1))),
          compare(mul(mul(var(1),var(2)),var(3)),mul(var(1),mul(var(2),var(3))))]
    swaps=0
    for length in range(2,7):
        for w in itertools.product(range(1,4),repeat=length):
            for j in range(length-1):
                if w[j]!=w[j+1]: adjacent_certificate(w,j); swaps+=1
    cuts=[cut_certificate(n,k) for n in range(2,7) for k in range(1,n)]
    resource=[{'n':n,'stream_state_scalars':n,'double_buffer_scalars':2*n,
        'dense_multiplications_upper':n**3,'dense_additions_upper':n*n*(n-1),
        'expanded_path_monomials':n**(n-1),'binary_expansion_tree_gates':2*n**n-1,
        'source_135_finite_n_lower_bound_not_asserted':True}
        for n in (2,3,4,5,6,8,16)]
    return {'schema':'ovc01.order_certificate.v1','status':'EXACT_ORDERED_RECURRENCE_CONTROL',
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'paper_families':[116,135],'rewrites':rows,'full_source_dimension_cases':full,
       'adjacent_swap_certificates':swaps,'cut_rank_certificates':cuts,'resources':resource,
       'source_116_general_hitting_proof_reproduced':False,
       'source_135_general_lower_bound_reproved':False,'trained_neural_checkpoint_evaluated':False,
       'exact_rational_arithmetic':True,'circuit_model':'ordered division-free formula trees'}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path)
    args=ap.parse_args(argv);r=report()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':r['status'],'rewrites':[
        {k:x[k] for k in ('name','correct_identity','scalar_trials','jet_dimension','witness_coefficient_index','witness_coefficient')}
        for x in r['rewrites']], 'adjacent_swaps':r['adjacent_swap_certificates'],
        'full_dimension_cases':r['full_source_dimension_cases'],'resources':r['resources']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
