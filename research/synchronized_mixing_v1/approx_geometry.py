"""SMC-06: bounded, fail-closed sparse-feature incidence decoder.

Input: anonymous approximate SAE feature activations. The matching-class
prior is explicit; n and the induced automorphism group are derived from
feature count and a globally certified six-star incidence cover.

No planted dictionary, matching IDs, labels, or frame keys enter fitting.
"""
from itertools import combinations, permutations
from math import isqrt, factorial


class SearchBudgetExhausted(ValueError):
    pass


def infer_model_shape(features, active):
    """Infer even vertex count from F=n(n-1)/2 and perfect-matching size n/2."""
    if features < 3 or active < 2:
        raise ValueError('insufficient features or matching cardinality')
    disc=1+8*features
    root=isqrt(disc)
    if root*root!=disc or (1+root)%2:
        raise ValueError('feature count is not a complete-graph edge count')
    n=(1+root)//2
    if n%2 or n//2 !=active:
        raise ValueError('active count inconsistent with perfect-matching model')
    return {'n_vertices':n,'n_edges':features,'matching_cardinality':active,
            'automorphism_group_order':factorial(n)}


def preprocess_rows(rows):
    rows=tuple(tuple(int(a) for a in row) for row in rows)
    if not rows or len(set(rows))!=len(rows):
        raise ValueError('observations must be distinct')
    features=len(rows[0]); active=sum(rows[0])
    shape=infer_model_shape(features,active)
    if any(len(row)!=features or sum(row)!=active or any(v not in (0,1) for v in row)
           for row in rows):
        raise ValueError('malformed sparse feature rows')
    # OLM-05/06 establish finite complete-case certificates for 12 and 13 rows.
    if features!=15 or active!=3 or len(rows) not in (12,13):
        raise ValueError('complete finite certificate covers 12 or 13 of 15 K6 states')
    counts=[sum(r[j] for r in rows) for j in range(features)]
    if max(counts)>3:
        raise ValueError('each edge occurs in at most three K6 perfect matchings')
    coactive=set()
    for row in rows:
        pair=list(combinations((j for j,x in enumerate(row) if x),2))
        if any(p in coactive for p in pair):
            raise ValueError('two distinct perfect matchings cannot share a pair of edges')
        coactive.update(pair)
    if len(coactive)!=3*len(rows):
        raise ValueError('each state must contribute three distinct coactivation pairs')
    return rows,shape,coactive


def bounded_cover(rows,max_star_candidates=None,max_nodes=1000):
    """Return unique K6 star cover or raise ambiguity; bound worst-case search."""
    rows,shape,coactive=preprocess_rows(rows)
    max_star_candidates=(47 if len(rows)==12 else 13) if max_star_candidates is None else max_star_candidates
    candidates=[]
    for sub in combinations(range(15),5):
        if all(p not in coactive for p in combinations(sub,2)):
            candidates.append(frozenset(sub))
            if len(candidates)>max_star_candidates:
                raise ValueError('too many five-cliques for the bounded K6 certificate')
    covers=[]
    nodes=0
    def rec(start,chosen,used):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:
            raise SearchBudgetExhausted('K6 inverse search budget exceeded')
        if len(covers)>1:return
        if len(chosen)==6:
            if all(c==2 for c in used):covers.append(tuple(chosen))
            return
        for k in range(start,len(candidates)):
            c=candidates[k]
            if any(len(c & old)!=1 for old in chosen):continue
            if any(used[j]>=2 for j in c):continue
            new=list(used)
            for j in c:new[j]+=1
            rec(k+1,chosen+[c],new)
    rec(0,[],[0]*15)
    if len(covers)!=1:
        raise ValueError('ambiguous or inconsistent K6 sparse geometry')
    stars=covers[0]
    feat={j:tuple(i for i,s in enumerate(stars) if j in s) for j in range(15)}
    if any(len(e)!=2 for e in feat.values()) or len(set(feat.values()))!=15:
        raise AssertionError('cover is not K6')
    for r in rows:
        edges=[feat[j] for j,x in enumerate(r) if x]
        if sorted(v for a,b in edges for v in (a,b))!=list(range(6)):
            raise AssertionError('row is not a perfect matching')
    # Build group candidates from the reconstructed incidence graph. This
    # infers the concrete action, while matching-family assumption remains.
    abstract_edges=set(feat.values())
    autos=[]
    for p in permutations(range(6)):
        if {tuple(sorted((p[a],p[b]))) for a,b in abstract_edges}==abstract_edges:
            autos.append(p)
    if len(autos)!=factorial(6):
        raise AssertionError('unexpected recovered graph automorphism group')
    return {'feature_edges':feat,'vertex_stars':tuple(tuple(sorted(s)) for s in stars),
            'n_candidate_stars':len(candidates),'search_nodes':nodes,
            'n_automorphisms':len(autos),'n_vertices_inferred':shape['n_vertices'],
            'group_order_derived':len(autos),'coactive_pairs':len(coactive)}


def top_cardinality(scores,k=3):
    """A fixed output sparsity choice; model metadata supplies k."""
    return tuple(tuple(int(j in set(sorted(range(len(r)),key=lambda j:(-r[j],j))[:k]))
                       for j in range(len(r))) for r in scores)


def recover_with_one_swap(scores,max_repairs=1):
    """Find a UNIQUE geometric correction of up to one 3-hot row swap.

    Score margins are recorded for auditing, never used to break ties among
    distinct consistent solutions. A second feasible repair causes abstention.
    """
    if len(scores)!=12 or any(len(row)!=15 for row in scores):
        raise ValueError('expected 12x15 activation scores')
    rows=top_cardinality(scores)
    try:
        geom=bounded_cover(rows)
        return {'geometry':geom,'masks':rows,'repairs':(),
                'reconstruction':'direct','candidate_repair_count':0}
    except SearchBudgetExhausted:
        raise ValueError('geometry search exhausted budget; abstain')
    except ValueError:
        if max_repairs<1:
            raise
    solutions=[]
    for i,row in enumerate(rows):
        on=[j for j,x in enumerate(row) if x]
        off=[j for j,x in enumerate(row) if not x]
        for a in on:
            for b in off:
                edited=list(rows)
                fixed=list(row)
                fixed[a],fixed[b]=0,1
                edited[i]=tuple(fixed)
                try:
                    geom=bounded_cover(edited)
                except SearchBudgetExhausted:
                    raise ValueError('repair search exhausted budget; abstain')
                except ValueError:
                    continue
                solutions.append({'geometry':geom,'masks':tuple(edited),
                                  'repairs':((i,a,b),),'reconstruction':'one_swap',
                                  'score_loss':float(scores[i][a]-scores[i][b])})
                if len(solutions)>1:
                    raise ValueError('multiple one-swap repairs fit; abstain')
    if len(solutions)!=1:
        raise ValueError('no unique geometry within repair budget')
    solution=solutions[0]
    solution['candidate_repair_count']=1
    return solution


def model_matching_states(geometry):
    """All recovered perfect-matching masks, without an oracle edge map."""
    from transport_lift import all_matchings
    inv={edge:j for j,edge in geometry['feature_edges'].items()}
    masks={}
    for matching in all_matchings(tuple(range(geometry['n_vertices_inferred']))):
        mask=tuple(int(j in [inv[e] for e in matching]) for j in range(15))
        masks[tuple(matching)]=mask
    if len(masks)!=15 or len(set(masks.values()))!=15:
        raise AssertionError('recovered model is not a 15-state matching code')
    return masks


def decode_soft(geometry,scores):
    """Pick the highest scored valid matching, rather than raw top3 support.

    Ties are abstention: no hidden ground-truth matching is used.
    """
    masks=model_matching_states(geometry)
    ranked=sorted([(sum(float(z) for z,b in zip(scores,mask) if b),m)
                   for m,mask in masks.items()],reverse=True)
    if len(ranked)>1 and abs(ranked[0][0]-ranked[1][0])<1e-9:
        raise ValueError('ambiguous matching scores')
    return ranked[0][1],ranked[0][0]-ranked[1][0]
