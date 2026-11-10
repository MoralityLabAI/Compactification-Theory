"""RRC-23: exact horizon/grammar scaling with a frozen degree-12 selector.

Candidate search is deliberately separate from the learned decision. A paid
candidate executes both public examples; progressive checks re-execute shorter
truncations and account for every one of these interpreter calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random

try:
    from . import graph11 as g, topology16 as t
except ImportError:
    import graph11 as g
    import topology16 as t

SEED = 2026111023
HORIZONS = (12, 24, 48, 96, 192)
MAX_DEGREE = max(HORIZONS) + 24
FAMILIES = ('signed', 'product', 'sparse', 'rational', 'squared', 'twisted_product', 'late_onset')
TOPOLOGIES = t.TOPOLOGIES
PER_CELL = 3
EXTENDED_PER_TOPOLOGY = 10
EXTENDED_DEGREES = (12, 48, 96)
STRESS_DEGREE = 384
STRESS_CASES = 8
PRE_OPS = ('I', 'N', 'R', 'S')


def source(rng: random.Random, family: str, degree: int = MAX_DEGREE) -> tuple[int, ...]:
    """Fresh, exact integral unit series generated to a shared maximum degree."""
    if degree < 12:raise ValueError('degree must include original model observation')
    if family == 'signed':
        return (1,) + tuple(rng.randint(-2, 2) for _ in range(degree))
    if family == 'sparse':
        return (1,) + tuple(rng.choice((-2,-1,1,2)) if rng.random()<.18 else 0
                            for _ in range(degree))
    if family in ('product','twisted_product'):
        modulus=rng.choice((5,7,9,11,13))
        density=rng.choice((.25,.5,.75))
        mask={k for k in range(modulus) if rng.random()<density}
        if not mask:mask={rng.randrange(modulus)}
        values=[1]+[0]*degree
        for part in range(1,degree+1):
            if part%modulus in mask:
                for n in range(part,degree+1): values[n]+=values[n-part]
        if family=='twisted_product':
            return tuple(v*(-1 if n%2 else 1) for n,v in enumerate(values))
        return tuple(values)
    if family in ('rational','squared'):
        polynomial=[1]+[0]*degree
        for _ in range(5):
            index=rng.randint(1,14)
            polynomial[index]+=rng.choice((-2,-1,1,2))
        if family=='rational':return g.inv(tuple(polynomial))
        return g.mul(tuple(polynomial),tuple(polynomial))
    if family=='late_onset':
        first=rng.randint(30,80)
        polynomial=[1]+[0]*degree
        for index in range(first,degree+1):
            if rng.random()<.12:polynomial[index]=rng.choice((-2,-1,1,2))
        polynomial[first]=rng.choice((-2,-1,1,2))
        return tuple(polynomial)
    raise ValueError('unknown source family')


def create_case(rng,topology,family,graph,identifier,*,pre='I',degree=MAX_DEGREE):
    def run(x):
        return t.execute(g.unary(x,pre),graph,topology)
    demos=tuple((a:=source(rng,family,degree),run(a)) for _ in range(2))
    query=source(rng,family,degree)
    return t.Case(identifier,family,topology,demos,query,run(query),graph)


def dataset(*,per_cell=PER_CELL,seed=SEED):
    """Source families and topology appear in a fully crossed, fixed roster."""
    if per_cell<1:raise ValueError('positive cell count required')
    cases=[];seen=set()
    for i,family in enumerate(FAMILIES):
        for j,topology in enumerate(TOPOLOGIES):
            rng=random.Random(seed+101*i+1009*j)
            attempts=0
            while sum(c.family==family and c.topology==topology for c in cases)<per_cell:
                attempts+=1
                if attempts>200*per_cell:raise RuntimeError('uniqueness exhausted')
                graph=rng.choice(g.GRAPHS)
                case=create_case(rng,topology,family,graph,f'{family}-{topology}-{len(cases):03d}')
                signature=(topology,case.demos)
                if signature in seen:continue
                seen.add(signature);cases.append(case)
    return tuple(cases)


def extended_dataset(*,per_topology=EXTENDED_PER_TOPOLOGY,seed=SEED+19000):
    """Four-token programs, for which frozen old model has no pre-op training."""
    if per_topology<1:raise ValueError('positive extended count required')
    out=[];seen=set()
    for i,topology in enumerate(TOPOLOGIES):
        rng=random.Random(seed+111*i)
        while sum(c.topology==topology for c,pre in out)<per_topology:
            family=rng.choice(FAMILIES[:-1]); pre=rng.choice(PRE_OPS)
            graph=rng.choice(g.GRAPHS)
            case=create_case(rng,topology,family,graph,f'extended-{topology}-{len(out):03d}',pre=pre)
            signature=(topology,case.demos)
            if signature in seen:continue
            seen.add(signature);out.append((case,pre))
    return tuple(out)


def stress_dataset(*,seed=SEED+40000):
    """Post-primary long-horizon timing set: two new inputs per topology."""
    out=[]
    families=('signed','product','rational','late_onset')
    for i,topology in enumerate(TOPOLOGIES):
        rng=random.Random(seed+211*i)
        for j in range(2):
            family=families[(i+j)%len(families)]
            graph=rng.choice(g.GRAPHS)
            out.append(create_case(rng,topology,family,graph,
                   f'stress-{topology}-{j:02d}',degree=STRESS_DEGREE+24))
    return tuple(out)


def observed(case,degree):
    if degree<12 or len(case.query)<degree+25:raise ValueError('invalid experiment horizon')
    if any(len(x)<=degree for x,y in case.demos):raise ValueError('short case')
    demos=tuple((x[:degree+1],y[:degree+1]) for x,y in case.demos)
    return t.Case(case.identifier,case.family,case.topology,demos,
                  case.query[:degree+25],case.future[:degree+25],case.hidden_graph,case.pair)


def execute(f,program,topology):
    if len(program)==3:return t.execute(f,tuple(program),topology)
    if len(program)==4 and program[0] in PRE_OPS:
        return t.execute(g.unary(f,program[0]),tuple(program[1:]),topology)
    raise ValueError('invalid graph program')


def program_cost(program,topology,degree):
    if len(program)==3:return t.graph_cost(tuple(program),topology,degree)
    if len(program)==4 and program[0] in PRE_OPS:
        overhead=(degree*(degree+1)//2 if program[0]=='R' else degree+1)
        return overhead+t.graph_cost(tuple(program[1:]),topology,degree)
    raise ValueError('invalid cost program')


def stages_for(degree,progressive):
    if degree<12:raise ValueError('degree too small')
    return tuple(sorted({min(x,degree) for x in (4,12,degree)})) if progressive else (degree,)


def verify(case,order,*,degree,budget=None,progressive=False):
    """Pay for two demonstration executions at EACH visited test resolution."""
    if budget is None:budget=len(order)
    if budget<1:raise ValueError('budget must be positive')
    stage_plan=stages_for(degree,progressive)
    trace=[];work=0;demo_executions=0;stage_counts={str(s):0 for s in stage_plan}
    for candidate in order:
        if len(trace)>=budget:break
        candidate=tuple(candidate)
        if candidate in trace:raise AssertionError('duplicate candidate graph')
        trace.append(candidate)
        valid=True
        for stage in stage_plan:
            checks=[execute(x[:stage+1],candidate,case.topology)==y[:stage+1]
                    for x,y in case.demos]
            demo_executions+=2
            stage_counts[str(stage)]+=1
            work+=2*program_cost(candidate,case.topology,stage)
            if not all(checks):valid=False;break
        if valid:
            return {'program':candidate,'trace':tuple(trace),'queries':len(trace),
                    'demo_executions':demo_executions,'nominal_work':work,
                    'stage_counts':stage_counts}
    return {'program':None,'trace':tuple(trace),'queries':len(trace),
            'demo_executions':demo_executions,'nominal_work':work,
            'stage_counts':stage_counts}


def grade(case,program):
    if program is None:return {'accepted':0,'future_correct':0,'false_accept':0}
    # Check against the observed degree, separate from withheld query output.
    degree=len(case.demos[0][0])-1
    accepted=int(all(execute(x,program,case.topology)==y for x,y in case.demos))
    correct=int(bool(accepted) and execute(case.query,program,case.topology)==case.future)
    return {'accepted':accepted,'future_correct':correct,'false_accept':accepted-correct}


def extended_order(base_order,*,kind='mass'):
    """Additional pre-operation uses an explicit, supplied four-way template.

    Pure learned ranking is recycled across each prefix primitive; this is an
    architectural extension, not a trained four-token decoder.
    """
    if kind=='mass':return tuple((pre,)+tuple(graph) for pre in PRE_OPS for graph in base_order)
    if kind=='round_robin':
        return tuple((pre,)+tuple(graph) for graph in base_order for pre in PRE_OPS)
    raise ValueError('invalid expansion order')


def digest(cases):
    return sha256(json.dumps([(c.identifier,c.family,c.topology,c.demos,c.query,c.future,c.hidden_graph)
                              for c in cases],separators=(',',':')).encode()).hexdigest()
