"""SMC-04: exact inference of an unknown vertex frame AND reference matching.

The learner sees (permutation, observed matching) examples and knows the
hypothesis family G = sigma * permutation * B. Neither sigma nor B enters
the inference functions. Enumeration is bounded to six vertices (S6).
All finite probabilities and expectation certificates use Fraction.
"""
from collections import Counter
from fractions import Fraction
from itertools import permutations
from math import sqrt
import random
from transport_lift import all_matchings, base_matching, relabel

ZERO=Fraction(0)
ONE=Fraction(1)


def inverse(perm):
    q=[0]*len(perm)
    for i,j in enumerate(perm): q[j]=i
    return tuple(q)


def compose(a,b):
    return tuple(a[b[i]] for i in range(len(a)))


def uniform_catalog(k=3):
    if k!=3:
        raise ValueError("bounded exact catalog supports six vertices")
    n=2*k
    ps=tuple(permutations(range(n)))
    ms=tuple(all_matchings(tuple(range(n))))
    mindex={g:i for i,g in enumerate(ms)}
    pindex={p:i for i,p in enumerate(ps)}
    table=tuple(tuple(mindex[relabel(p,g)] for g in ms) for p in ps)
    return {'n':n,'perms':ps,'matchings':ms,'mindex':mindex,
            'pindex':pindex,'transport':table,
            'id':pindex[tuple(range(n))],
            'ref':mindex[base_matching(k)]}


def hidden_fixture(catalog,seed=1719):
    rng=random.Random(seed)
    return rng.randrange(len(catalog['perms'])),rng.randrange(len(catalog['matchings']))


def predict(catalog,hypothesis,perm_index):
    sigma,base=hypothesis
    T=catalog['transport']
    return T[sigma][T[perm_index][base]]


def calibration_cohort(catalog,hypothesis,heldout=3):
    m=len(catalog['matchings'])
    if not 1<=heldout<m:
        raise ValueError("invalid heldout class count")
    cut=m-heldout
    train=[i for i in range(len(catalog['perms']))
           if predict(catalog,hypothesis,i)<cut]
    test=[i for i in range(len(catalog['perms']))
          if predict(catalog,hypothesis,i)>=cut]
    assert len(train)==cut*len(catalog['perms'])//m
    assert len(test)==heldout*len(catalog['perms'])//m
    return train,test


def observed_calibrations(catalog,hypothesis,indices):
    return tuple((i,predict(catalog,hypothesis,i)) for i in indices)


def version_space(catalog,observations):
    """One positive observation reduces 10,800 pairs to 720 frame candidates.

    Given (S,G) and sigma, the only possible base is S^-1 sigma^-1 G.
    """
    if not observations:
        raise ValueError("at least one positive pair required")
    ps=catalog['perms']
    T=catalog['transport']
    pi=catalog['pindex']
    s0,g0=observations[0]
    s0_inv=pi[inverse(ps[s0])]
    survivors=[]
    for sigma,p in enumerate(ps):
        sigma_inv=pi[inverse(p)]
        b=T[s0_inv][T[sigma_inv][g0]]
        if all(T[sigma][T[sid][b]]==gid for sid,gid in observations[1:]):
            survivors.append((sigma,b))
    return tuple(survivors)


def maximum_consensus(catalog,observations):
    """Noisy likelihood: maximize correct matching predictions (uniform noise).

    Full 720*15 exact candidates, deterministic lexicographic tie-break.
    """
    if not observations:
        raise ValueError("observations required")
    T=catalog['transport']
    best_score=-1
    best=[]
    for sigma in range(len(catalog['perms'])):
        row=T[sigma]
        for b in range(len(catalog['matchings'])):
            score=sum(row[T[p][b]]==g for p,g in observations)
            if score>best_score:
                best_score=score
                best=[(sigma,b)]
            elif score==best_score:
                best.append((sigma,b))
    return {'best_score':best_score,'maximizers':tuple(best),
            'lexicographic_hat':best[0],'sample_count':len(observations)}


def hypothesis_agreement_histogram(catalog,truth,cohort=None):
    """Exact agreement count for every candidate on a specified input support."""
    T=catalog['transport']
    ids=range(len(catalog['perms'])) if cohort is None else tuple(cohort)
    targets=[predict(catalog,truth,p) for p in ids]
    hist=Counter()
    max_false=(0,None)
    true_count=0
    for sigma in range(len(catalog['perms'])):
        row=T[sigma]
        for b in range(len(catalog['matchings'])):
            hits=sum(row[T[p][b]]==g for p,g in zip(ids,targets))
            hist[hits]+=1
            if (sigma,b)==truth:
                assert hits==len(targets)
                true_count+=1
            elif hits>max_false[0]:
                max_false=(hits,(sigma,b))
    assert true_count==1
    if hist[len(targets)]!=1:
        raise AssertionError("different hypotheses are indistinguishable")
    return {'sample_support':len(targets),'num_hypotheses':sum(hist.values()),
            'max_false_agreement':Fraction(max_false[0],len(targets)),
            'max_false_hypothesis':max_false[1],
            'counts':dict(sorted(hist.items()))}


def expected_spurious(hist,samples):
    n=hist['sample_support']
    return sum(((count-(1 if hits==n else 0))*Fraction(hits,n)**samples
                for hits,count in hist['counts'].items()),ZERO)


def noisy_ambiguity_upper(hist,samples,noise,m=15):
    """Chernoff union bound: false consensus score at least true score."""
    if not 0<=noise<=1:
        raise ValueError("noise must be between 0 and 1")
    a=1-float(noise)+float(noise)/m
    b=float(noise)/m
    n=hist['sample_support']
    total=0.0
    for hits,count in hist['counts'].items():
        false_count=count-(1 if hits==n else 0)
        if not false_count:
            continue
        alpha=hits/n
        factor=1-(1-alpha)*(sqrt(a)-sqrt(b))**2
        total+=false_count*factor**samples
    return min(1.0,total)


def compatibility_accuracy(catalog,hat,truth,test_indices,test_graph_ids):
    """Equal-prior accuracy for orbit-locked positives versus independent null."""
    T=catalog['transport']
    sigma,b=hat
    good=sum(predict(catalog,hat,i)==predict(catalog,truth,i)
             for i in test_indices)
    false=sum(T[sigma][T[i][b]] in test_graph_ids for i in test_indices)
    tpr=Fraction(good,len(test_indices))
    fpr=Fraction(false,len(test_indices)*len(test_graph_ids))
    return {'balanced_accuracy':(1+tpr-fpr)/2,'tpr':tpr,'fpr':fpr}


def paired_campaign(seeds=(43,77,91),noise=ZERO,
                    prefixes=(1,2,3,4,5,6,8,10,12,16,24),trials_per_seed=60):
    """Nested iid calibration prefixes, 3 unknown models, fixed control splits."""
    c=uniform_catalog()
    max_n=max(prefixes)
    keys=('mean_candidates','mean_balanced_accuracy',
          'mean_positive_recall','mean_false_positive_rate')
    totals={n:{'correct':0,'singleton':0,**{k:ZERO for k in keys}}
            for n in prefixes}
    for seed in seeds:
        truth=hidden_fixture(c,seed)
        train,test=calibration_cohort(c,truth)
        for trial in range(trials_per_seed):
            rng=random.Random(seed*100003+trial)
            inputs=[rng.choice(train) for _ in range(max_n)]
            observations=[]
            for pid in inputs:
                g=predict(c,truth,pid)
                if noise and rng.random()<float(noise):
                    g=rng.randrange(len(c['matchings']))
                observations.append((pid,g))
            for n in prefixes:
                pair_prefix=observations[:n]
                if noise:
                    hypotheses=maximum_consensus(c,pair_prefix)['maximizers']
                else:
                    hypotheses=version_space(c,pair_prefix)
                assert hypotheses
                hat=hypotheses[0]
                result=compatibility_accuracy(c,hat,truth,test,tuple(range(12,15)))
                out=totals[n]
                out['correct']+=hat==truth
                out['singleton']+=len(hypotheses)==1
                out['mean_candidates']+=len(hypotheses)
                out['mean_balanced_accuracy']+=result['balanced_accuracy']
                out['mean_positive_recall']+=result['tpr']
                out['mean_false_positive_rate']+=result['fpr']
    n_trials=len(seeds)*trials_per_seed
    return {'seeds':list(seeds),'trials_per_seed':trials_per_seed,
            'num_trials':n_trials,'noise':str(noise),
            'iid_inputs_with_replacement':True,
            'n_matching_states':15,'n_hypotheses':10800,
            'negative_test_matching_law':'uniform among three withheld graph states',
            'series':[{'n':n,**{key:(str(Fraction(val,n_trials))
                                 if key.startswith('mean') else val)
                              for key,val in data.items()},
                       'exact_recovery_fraction':str(Fraction(data['correct'],n_trials))}
                      for n,data in sorted(totals.items())]}


def negative_calibration_control(seeds=(43,77,91),examples=12,
                                 trials_per_seed=20):
    """Independent graph labels, identical truth-conditioned input cohort."""
    c=uniform_catalog()
    accuracies=[]
    for seed in seeds:
        truth=hidden_fixture(c,seed)
        train,test=calibration_cohort(c,truth)
        for trial in range(trials_per_seed):
            rng=random.Random(seed*200003+trial)
            observations=[(rng.choice(train),rng.randrange(12))
                          for _ in range(examples)]
            estimate=maximum_consensus(c,observations)['lexicographic_hat']
            result=compatibility_accuracy(c,estimate,truth,test,range(12,15))
            accuracies.append(result['balanced_accuracy'])
    return {'examples':examples,'trials':len(accuracies),
            'mean_heldout_balanced_accuracy':sum(accuracies,ZERO)/len(accuracies),
            'minimum_heldout_balanced_accuracy':min(accuracies),
            'maximum_heldout_balanced_accuracy':max(accuracies)}
