"""RRC-09: bounded, executable formal-series program composition.

The `C` (convolution-square) and `D` (Euler/divisor inverse) primitives have
no labeled examples in the training roster. Their public implementations are
available to the exact and execution-guided synthesizers during evaluation.
The direct neural model only receives paired coefficient prefixes.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from typing import Sequence

try:
    from .rrc import euler_exponents
    from .rrc05 import Program05, product_coefficients
except ImportError:
    from rrc import euler_exponents
    from rrc05 import Program05, product_coefficients

PREFIX = 18
PREVIEW = 6
HORIZON = 48
SEED = 2026110909
OPS = ('N','S','R','C','D')
TRAIN_OPS = ('N','S','R')
PAD = '.'
TRAIN_COUNT = 900
DEV_COUNT = 90
TEST_COUNTS = {'seen_graph':48,'heldout_C':48,'heldout_D':48,'heldout_CD':32,'out_of_grammar':24,'late_collision':16}


def valid_series(values: Sequence[int]) -> None:
    if not values or values[0] != 1 or any(type(v) is not int for v in values):
        raise ValueError('formal series must have integer coefficients and constant 1')


def operate(values: Sequence[int], token: str) -> tuple[int,...]:
    """Public interpreter; exact integer coefficients through supplied degree."""
    valid_series(values)
    nmax = len(values)-1
    if token=='N':
        result = [v*(-1 if n%2 else 1) for n,v in enumerate(values)]
    elif token=='S':
        result = [values[n//2] if n%2==0 else 0 for n in range(nmax+1)]
    elif token=='R':
        result = [1]+[0]*nmax
        for n in range(1,nmax+1):
            result[n] = -sum(values[k]*result[n-k] for k in range(1,n+1))
    elif token=='C':
        result = [0]*(nmax+1)
        for i,vi in enumerate(values):
            if vi:
                for j in range(nmax+1-i):
                    result[i+j] += vi*values[j]
    elif token=='D':
        result = [1]+euler_exponents(values)[1:]
    else:
        raise ValueError(f'unknown primitive: {token}')
    return tuple(result)


def execute(values: Sequence[int], program: Sequence[str]) -> tuple[int,...]:
    state=tuple(values)
    for token in program:
        state=operate(state,token)
    return state


def all_programs(ops:Sequence[str]=OPS) -> tuple[tuple[str,...],...]:
    """Strictly ordered by code length then alphabet rank; identity included."""
    from itertools import product
    return ((),*((o,) for o in ops),*product(ops,repeat=2))


def first_fault(left: Sequence[int],right:Sequence[int],start:int=1) -> int|None:
    if len(left)!=len(right):raise ValueError('different series lengths')
    return next((i for i in range(start,len(left)) if left[i]!=right[i]),None)


def arithmetic_work(program:Sequence[str],degree:int)->int:
    """Static estimate of integer coefficient-loop iterations in this interpreter.

    Excludes bit complexity, input generation, model inference, sorting, and
    benchmark scoring. Nonlinear primitives have quadratic nominal cost.
    """
    if degree<0:raise ValueError('negative polynomial degree')
    divisor_steps=sum(degree//d-1 for d in range(1,degree+1))
    table={'N':degree+1,'S':degree+1,
           'R':degree*(degree+1)//2,
           'C':(degree+1)*(degree+2)//2,
           'D':degree*(degree-1)//2+divisor_steps+degree}
    if any(token not in table for token in program):
        raise ValueError('unknown primitive in work estimate')
    return sum(table[token] for token in program)


def catalog_search(source:Sequence[int], target:Sequence[int], *,
                   ops:Sequence[str]=OPS, degree:int=PREFIX) -> dict:
    """Bounded exact synthesis with prefix execution and 2-stage pruning.

    Counts one program candidate per attempt, one public primitive interpreter
    call per operation executed, and full-prefix verifier attempts. Interpreter
    uses exact arithmetic. No access to coefficient values beyond `degree`.
    """
    if degree<PREVIEW or degree>=len(source) or len(source)!=len(target):
        raise ValueError('invalid observed prefix')
    source=tuple(source[:degree+1]);target=tuple(target[:degree+1])
    attempted=0; primitive_calls=0; full_checks=0; work=0
    for prog in all_programs(ops):
        attempted+=1
        preview=execute(source[:PREVIEW+1],prog)
        primitive_calls+=len(prog);work+=arithmetic_work(prog,PREVIEW)
        if preview != target[:PREVIEW+1]:continue
        full_checks+=1
        actual=execute(source,prog)
        primitive_calls+=len(prog);work+=arithmetic_work(prog,degree)
        if actual==target:
            return {'program':list(prog),'candidates_examined':attempted,
                    'primitive_calls':primitive_calls,'full_verifications':full_checks,
                    'estimated_arithmetic_work':work}
    return {'program':None,'candidates_examined':attempted,
            'primitive_calls':primitive_calls,'full_verifications':full_checks,
                    'estimated_arithmetic_work':work}


@dataclass(frozen=True)
class Case09:
    identifier: str
    split: str
    family: str
    source:tuple[int,...]
    target:tuple[int,...]
    hidden_program:tuple[str,...]|None
    pair:str|None=None

    @property
    def input_signature(self)->tuple[tuple[int,...],tuple[int,...]]:
        return self.source[:PREFIX+1],self.target[:PREFIX+1]


def _random_source(rng:random.Random) -> tuple[int,...]:
    """Independent product-series generator; exact source never carries labels."""
    while True:
        modulus=rng.choice((3,4,5,6,7,8,9,10,11,12,13,17))
        cap=rng.choice((0,1,2))
        density=rng.choice((.25,.4,.6,.75))
        residues=tuple(r for r in range(modulus) if rng.random()<density)
        if not residues:continue
        f=tuple(product_coefficients(HORIZON,Program05(cap,modulus,residues)))
        if len(set(f[1:8]))>1 and sum(v!=0 for v in f[1:8])>=3:
            return f


def _programs_seen() -> tuple[tuple[str,...],...]:
    return tuple(p for p in all_programs(TRAIN_OPS) if p)


def _target_programs(family:str) -> tuple[tuple[str,...],...]:
    if family=='seen_graph':return _programs_seen()
    if family=='heldout_C':return tuple(p for p in all_programs() if 'C' in p and 'D' not in p)
    if family=='heldout_D':return tuple(p for p in all_programs() if 'D' in p and 'C' not in p)
    if family=='heldout_CD':return (('C','D'),('D','C'))
    raise ValueError(f'bad family {family}')


def generate(*,include_test:bool=True, train_count:int=TRAIN_COUNT,dev_count:int=DEV_COUNT)->dict[str,list[Case09]]:
    if train_count<1 or dev_count<1:raise ValueError('positive train/dev sizes')
    split_counts={'train':train_count,'dev':dev_count}
    if include_test:split_counts.update(TEST_COUNTS)
    result={name:[] for name in split_counts}
    observed=set()

    def insert(split,f,g,program,family,pair=None,allow_pair=False):
        f=tuple(f);g=tuple(g)
        if len(f)!=HORIZON+1 or len(g)!=HORIZON+1:raise AssertionError('horizon')
        signature=(f[:PREFIX+1],g[:PREFIX+1])
        if signature in observed and not allow_pair:return False
        observed.add(signature)
        result[split].append(Case09(f'{split}-{len(result[split]):04d}',split,family,f,g,program,pair))
        return True

    def draw(split,count,seed,family,programs):
        rng=random.Random(seed);attempts=0
        while len(result[split])<count:
            attempts+=1
            if attempts>200*count:raise RuntimeError('uniqueness exhausted: '+split)
            program=rng.choice(programs)
            src=_random_source(rng)
            output=execute(src,program)
            insert(split,src,output,program,family)

    draw('train',train_count,SEED+11,'known_primitives',_programs_seen())
    draw('dev',dev_count,SEED+22,'known_primitives',_programs_seen())
    if not include_test:return result
    for i,family in enumerate(('seen_graph','heldout_C','heldout_D','heldout_CD')):
        draw(family,TEST_COUNTS[family],SEED+100+i,family,_target_programs(family))

    # Negative control: a deterministic perturbation at an OBSERVED degree,
    # accepted only when the full grammar cannot explain the resulting prefix.
    rng=random.Random(SEED+777)
    attempts=0
    while len(result['out_of_grammar'])<TEST_COUNTS['out_of_grammar']:
        attempts+=1
        if attempts>40000:raise RuntimeError('negative case exhaustion')
        f=_random_source(rng);p=rng.choice(_target_programs('heldout_CD'))
        g=list(execute(f,p));degree=rng.randrange(2,PREFIX-1)
        g[degree]+=rng.choice((-2,-1,1,2))
        if catalog_search(f,g)['program'] is not None:continue
        insert('out_of_grammar',f,g,None,'out_of_grammar')

    # Eight exact-prefix collisions, two cases per pair. Mutant's hidden
    # continuation differs at degree K>PREFIX, making future proof impossible.
    rng=random.Random(SEED+888)
    for pair_index in range(TEST_COUNTS['late_collision']//2):
        while True:
            f=_random_source(rng);p=rng.choice(_target_programs('heldout_CD'))
            base=execute(f,p)
            k=PREFIX+1+(pair_index*3)%(HORIZON-PREFIX)
            g=list(base);g[k]+=1 if pair_index%2==0 else -1
            label=f'collision-{pair_index:02d}'
            if insert('late_collision',f,base,p,'late_collision',pair=label):
                ok=insert('late_collision',f,g,None,'late_collision',pair=label,allow_pair=True)
                assert ok
                break
    return result


def roster_digest(cases: Sequence[Case09], *, include_full:bool=True) -> str:
    payload=[{'id':c.identifier,'split':c.split,'family':c.family,
              'source':list(c.source if include_full else c.source[:PREFIX+1]),
              'target':list(c.target if include_full else c.target[:PREFIX+1]),
              'hidden_program':list(c.hidden_program) if c.hidden_program is not None else None,
              'pair':c.pair} for c in cases]
    return sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def feature_pairs(case:Case09)->list[list[float]]:
    """Raw coefficient features only, model source/target prefixes."""
    from math import log1p, copysign
    def value(x:int)->float:
        return copysign(min(30.0,log1p(abs(x))),x)/10.0
    return [[value(a),value(b)] for a,b in zip(case.source[:PREFIX+1],case.target[:PREFIX+1])]


def decode_tokens(indices:Sequence[int]) ->tuple[str,...]:
    vocab=(PAD,)+OPS
    if len(indices)!=2 or any(not 0<=int(i)<len(vocab) for i in indices):
        raise ValueError('two tokens in vocabulary required')
    if indices[0]==0:return ()
    if indices[1]==0:return (vocab[indices[0]],)
    return (vocab[indices[0]],vocab[indices[1]])


def canonical_tokens(case:Case09, *, ops:Sequence[str]=TRAIN_OPS)->tuple[str,...]:
    """Training-only canonical label; ties are resolved by public DSL order."""
    found=catalog_search(case.source,case.target,ops=ops)
    if found['program'] is None:raise AssertionError('training case not representable')
    return tuple(found['program'])


def candidate_features(source:Sequence[int],target:Sequence[int],prog:Sequence[str]):
    """Public low-degree execution sketch, available only to guided arms."""
    from math import log1p
    y=execute(source[:PREVIEW+1],prog)
    diffs=[abs(y[n]-target[n]) for n in range(1,PREVIEW+1)]
    first=next((n for n,v in enumerate(diffs,1) if v),PREVIEW+1)
    return (sum(v==0 for v in diffs)/PREVIEW,
            first/(PREVIEW+1),
            log1p(sum(diffs))/20,
            log1p(max(diffs))/20,
            len(prog)/2)


def guided_search(source:Sequence[int],target:Sequence[int],model_weights:Sequence[float]|None,*,
                  budget:int=4)->dict:
    """Score all public candidate executions at degree 6, then verify up to
    `budget` candidates exactly through degree 18. Preview costs are charged.
    """
    if budget<1:raise ValueError('positive verification budget')
    candidates=[]
    for rank,prog in enumerate(all_programs()):
        features=candidate_features(source,target,prog)
        if model_weights is None:
            # Transparent, untrained exact-sketch ranking comparator.
            priority=(-features[1],-features[0],features[2],features[3],len(prog),rank)
        else:
            score=sum(w*x for w,x in zip(model_weights[:-1],features))+model_weights[-1]
            priority=(-score,rank)
        candidates.append((priority,rank,prog))
    candidates.sort()
    calls=sum(len(p) for _,_,p in candidates)
    work=sum(arithmetic_work(p,PREVIEW) for _,_,p in candidates)
    for i,(_,_,prog) in enumerate(candidates[:budget],1):
        calls+=len(prog);work+=arithmetic_work(prog,PREFIX)
        if execute(source[:PREFIX+1],prog)==tuple(target[:PREFIX+1]):
            return {'program':list(prog),'preview_candidates':len(candidates),
                    'primitive_calls':calls,'full_verifications':i,
                    'estimated_arithmetic_work':work}
    return {'program':None,'preview_candidates':len(candidates),
            'primitive_calls':calls,'full_verifications':min(budget,len(candidates)),
            'estimated_arithmetic_work':work}
