"""SMC-05: infer K6 edge-coordinate geometry from anonymous sparse matching masks.

The reconstruction routine receives only anonymous binary feature rows,
without vertex labels, matching IDs, channel scramble or group actions.
The six-vertex/15-edge matching model class remains an explicit prior.
"""
from itertools import combinations
import random

FEATURES = 15
VERTICES = 6
STAR_SIZE = 5


def infer_feature_geometry(rows):
    """Return the UNIQUE six-star cover; fail closed on ambiguity.

    Input: 11 or 12 distinct three-hot states of 15 anonymous features.
    This function has no ground-truth edge map or access to fixture secrets.
    """
    rows = tuple(tuple(int(x) for x in row) for row in rows)
    if (len(rows) not in (11,12) or
        any(len(row)!=FEATURES or sum(row)!=3 or any(x not in (0,1) for x in row)
            for row in rows)):
        raise ValueError('expected 11 or 12 binary three-hot rows over 15 channels')
    if len(set(rows))!=len(rows):
        raise ValueError('geometry requires distinct observed states')
    coactive=set()
    for row in rows:
        active=[i for i,x in enumerate(row) if x]
        coactive.update(tuple(sorted(p)) for p in combinations(active,2))
    candidates=[]
    for candidate in combinations(range(FEATURES),STAR_SIZE):
        if all((a,b) not in coactive for a,b in combinations(candidate,2)):
            candidates.append(frozenset(candidate))
    covers=[]
    def search(start,chosen,incidences):
        if len(covers)>=2:
            return
        if len(chosen)==VERTICES:
            if all(count==2 for count in incidences):
                covers.append(tuple(chosen))
            return
        for j in range(start,len(candidates)):
            star=candidates[j]
            if any(len(star & old)!=1 for old in chosen):
                continue
            if any(incidences[i]==2 for i in star):
                continue
            next_counts=list(incidences)
            for i in star:
                next_counts[i]+=1
            search(j+1,chosen+[star],next_counts)
    search(0,[],[0]*FEATURES)
    if len(covers)!=1:
        raise ValueError(f'non-identifiable: {len(covers)} K6 star covers')
    cliques=covers[0]
    if any(len(a & b)!=1 for a,b in combinations(cliques,2)):
        raise AssertionError('star intersections violate certificate')
    feature_edges={}
    for f in range(FEATURES):
        endpoints=[i for i,star in enumerate(cliques) if f in star]
        if len(endpoints)!=2:
            raise ValueError('each feature must belong to exactly two stars')
        feature_edges[f]=tuple(endpoints)
    if len(set(feature_edges.values()))!=FEATURES:
        raise ValueError('duplicate abstract edges')
    for row in rows:
        abstract=[feature_edges[i] for i,active in enumerate(row) if active]
        if sorted([v for a,b in abstract for v in (a,b)])!=list(range(VERTICES)):
            raise ValueError('state does not decode to a perfect matching')
    return {'cliques':tuple(tuple(sorted(s)) for s in cliques),
            'candidate_5_cliques':len(candidates),
            'unique_six_star_cover':True,
            'feature_edges':feature_edges,
            'observed_matching_masks':rows,
            'coactive_pairs':len(coactive),
            'noncoactive_pairs':FEATURES*(FEATURES-1)//2-len(coactive)}


def decode_mask(geometry,row):
    row=tuple(int(x) for x in row)
    if len(row)!=FEATURES or sum(row)!=3:
        raise ValueError('expected 15-bit three-hot exact observation')
    match=tuple(sorted(geometry['feature_edges'][i] for i,x in enumerate(row) if x))
    if sorted([v for a,b in match for v in (a,b)])!=list(range(VERTICES)):
        raise ValueError('observation outside recovered matching manifold')
    return match


def scramble_masks(masks,seed=1701):
    """Fixture-only feature relabel; never used by the geometry estimator."""
    rng=random.Random(seed)
    perm=list(range(FEATURES))
    rng.shuffle(perm)
    out=[]
    for mask in masks:
        row=[0]*FEATURES
        for old,new in enumerate(perm):
            row[new]=int(mask[old])
        out.append(tuple(row))
    return tuple(out),tuple(perm)


def noisy_replicates(masks,repeats=40,sensitivity=0.9,false_positive=0.1,seed=117):
    """State-episode grouping is supplied; feature identities remain anonymous."""
    if not 0 <= false_positive < sensitivity <= 1 or repeats<=0:
        raise ValueError('invalid binary observation channel')
    rng=random.Random(seed)
    return tuple(tuple(tuple(int(rng.random() < (sensitivity if x else false_positive))
                        for x in state)
                  for _ in range(repeats)) for state in masks)


def majority_denoise(replicates,threshold=0.5):
    if not 0 < threshold < 1:
        raise ValueError('invalid threshold')
    rows=[]
    for episode in replicates:
        if not episode or any(len(row)!=FEATURES for row in episode):
            raise ValueError('bad episode')
        n=len(episode)
        rows.append(tuple(int(sum(row[j] for row in episode)/n > threshold)
                          for j in range(FEATURES)))
    return tuple(rows)


def switching_null(masks,seed=2718,steps=4000):
    """Random 2x2 switches preserve every row and column sum exactly."""
    rows=[list(row) for row in masks]
    if not rows or any(len(r)!=FEATURES for r in rows):
        raise ValueError('invalid incidence matrix')
    rng=random.Random(seed)
    for _ in range(steps):
        r,s=rng.sample(range(len(rows)),2)
        a,b=rng.sample(range(FEATURES),2)
        if rows[r][a]==1 and rows[s][a]==0 and rows[r][b]==0 and rows[s][b]==1:
            rows[r][a],rows[s][a]=0,1
            rows[r][b],rows[s][b]=1,0
        elif rows[r][a]==0 and rows[s][a]==1 and rows[r][b]==1 and rows[s][b]==0:
            rows[r][a],rows[s][a]=1,0
            rows[r][b],rows[s][b]=0,1
    return tuple(tuple(r) for r in rows)


def exact_recovery_oracle():
    """Simulator-only fixture; the estimator above cannot access this list."""
    from transport_lift import all_matchings
    edge_list=tuple(combinations(range(6),2))
    matchings=tuple(all_matchings(tuple(range(6))))
    observed=[matching for i,matching in enumerate(matchings) if i not in (1,5,10)]
    masks=[tuple(int(edge in matching) for edge in edge_list) for matching in observed]
    return tuple(masks),tuple(matchings),edge_list


if __name__=='__main__':
    orig,_,_=exact_recovery_oracle()
    latent,_=scramble_masks(orig)
    geom=infer_feature_geometry(latent)
    print({'stars':len(geom['cliques']),
           'candidate_5_cliques':geom['candidate_5_cliques'],
           'recovered_states':len(set(decode_mask(geom,row) for row in latent))})
