"""COE-04 constructive temporal tied-gate binary observers and finite oracles.

Theorems for equilateral K_n, n >= 5, with aggregate binary Hamming
budget <=2: min total bits = n-1 at margin 1, n at margin 2;
margin >=3 is infeasible. Unrolling T visits with k bits per visit gives
k = ceil(total_bits / T). Tied recurrent implementation tracks phase.
"""
from itertools import combinations


def minimum_heads(n, visits, margin):
    if n < 5 or visits < 1 or margin < 1:
        raise ValueError('the exact large-clique theorem needs n>=5, T>=1, r>=1')
    if margin > 2:
        return None
    required=n-1 if margin==1 else n
    return (required+visits-1)//visits


def singleton_codes(n, visits, heads, margin=1):
    width=visits*heads
    minimum=n-1 if margin==1 else n
    if n<5 or margin not in (1,2) or width<minimum:
        raise ValueError('insufficient history bandwidth')
    return tuple(([0]*(width) if margin==1 and index==n-1 else [int(j==index) for j in range(width)]) for index in range(n))


def tied_step(state):
    symbol,phase=state
    return symbol,phase+1


def tied_probe(state,codes,heads):
    symbol,phase=state
    return tuple(codes[symbol][phase*heads:phase*heads+heads])


def tied_trace(symbol,codes,visits,heads):
    state=(symbol,0)
    output=[]
    for _ in range(visits):
        word=tied_probe(state,codes,heads)
        assert len(word)==heads
        output.extend(word)
        state=tied_step(state)
    return tuple(output)


def distance(a,b):return sum(x!=y for x,y in zip(a,b))


def verify_tied(n,visits,margin):
    heads=minimum_heads(n,visits,margin)
    if heads is None:return {'possible':False}
    codes=singleton_codes(n,visits,heads,margin)
    stream=[tied_trace(i,codes,visits,heads) for i in range(n)]
    ds=[distance(stream[u],stream[v]) for u,v in combinations(range(n),2)]
    assert min(ds)>=margin and max(ds)<=2
    assert len(set(stream))==n
    return {'possible':True,'n':n,'visits':visits,'margin':margin,
            'heads':heads,'total_bits':heads*visits,
            'min_pair_distance':min(ds),'max_pair_distance':max(ds)}


def maximum_equidistant_binary_family(width):
    """Independent exact clique oracle: count max 2-distance code size."""
    if width>4:raise ValueError('finite oracle restricted to <=4 bits')
    words=tuple(range(1<<width))
    distances={w:tuple(v for v in words if v>w and (w^v).bit_count()==2) for w in words}
    best=0
    def rec(chosen,remaining):
        nonlocal best
        best=max(best,len(chosen))
        if len(chosen)+len(remaining)<=best:return
        for i,v in enumerate(remaining):
            rec(chosen+(v,),tuple(w for w in remaining[i+1:] if w in distances[v]))
    rec((),words)
    return best


def contracting_late_code(inputs,alpha,delta,visits):
    """Toy diagnostic: common fixed point zero, threshold delta > 0."""
    if not 0<alpha<1 or delta<=0:raise ValueError('invalid contraction')
    tstar=0
    R=max(abs(x) for x in inputs)
    while R*alpha**tstar>=delta:tstar+=1
    codes=[[int(alpha**t*x>=delta) for t in range(visits)] for x in inputs]
    assert all(len({word[t] for word in codes})==1 for t in range(tstar,visits))
    return {'terminal_innovation_visit':tstar,'codes':codes}
