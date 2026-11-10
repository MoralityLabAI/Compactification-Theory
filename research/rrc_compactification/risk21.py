"""RRC-21: source-shift conformal p-value and separate entropy selector.

The calibrator sees only independent known-family reference/calibration cases.
Each test prediction uses public inputs; exact program checks happen later.
"""
from __future__ import annotations
from hashlib import sha256
import json
import math

try:
    from . import select21 as d, gate20 as mix, graph11 as g, learn18 as l
except ImportError:
    import select21 as d
    import gate20 as mix
    import graph11 as g
    import learn18 as l

OOD_ALPHA=0.10
ENTROPY_QUANTILE=0.80
FEATURES=10


def features(case):
    x=mix.source_features(case)[:FEATURES]
    if len(x)!=FEATURES or not all(math.isfinite(z) for z in x):
        raise AssertionError('source features invalid')
    return tuple(x)


def fit_reference(cases):
    """Independent training-style source profiles, with diagonal normalization."""
    samples=[features(c) for c in cases]
    if len(samples)<20:raise ValueError('insufficient reference sources')
    avg=tuple(sum(z[j] for z in samples)/len(samples) for j in range(FEATURES))
    scales=tuple(max(.05, math.sqrt(sum((z[j]-avg[j])**2 for z in samples)/len(samples)))
                 for j in range(FEATURES))
    norm=tuple(tuple((v-avg[j])/scales[j] for j,v in enumerate(z)) for z in samples)
    return {'mean':avg,'scales':scales,'reference':norm}


def distance(fv,reference):
    vec=tuple((fv[j]-reference['mean'][j])/reference['scales'][j] for j in range(FEATURES))
    return min(sum((x-y)**2 for x,y in zip(vec,ref)) for ref in reference['reference'])


def conformal_pvalue(nonconformity,calibration_scores):
    if not calibration_scores:raise ValueError('calibration scores required')
    return (1+sum(x>=nonconformity for x in calibration_scores))/(len(calibration_scores)+1)


def mixed_distribution(case,logits,classes):
    probs=mix.expert_distributions(case.topology,logits,classes)
    return tuple(sum(p[i] for p in probs)/4 for i in range(len(probs[0])))


def entropy(case,logits,classes):
    dist=mixed_distribution(case,logits,classes)
    return -sum(x*math.log(max(x,1e-200)) for x in dist)/math.log(len(dist))


def order(case,logits,classes):
    return mix.order(case,logits,classes,fixed=(1,1,1,1))


def index(calibration,known_reference,calibration_entropies):
    """Known-only split conformal OOD cutoff and independent entropy quantile.

    Calibration entropy can be supplied only from frozen model proposals;
    selection does not use calibration ground truth or source-family labels.
    """
    reference=fit_reference(known_reference)
    scores=sorted(distance(features(c),reference) for c in calibration)
    ents=sorted(calibration_entropies)
    if len(ents)!=len(calibration):raise ValueError('calibration entropy length mismatch')
    threshold=ents[max(0,math.ceil(ENTROPY_QUANTILE*len(ents))-1)]
    return {'schema':'rrc21.calibration.v1','seed':d.SEED,
            'reference_count':len(known_reference),'calibration_count':len(calibration),
            'reference_sha256':d.digest(known_reference),'calibration_sha256':d.digest(calibration),
            'ood_alpha':OOD_ALPHA,'entropy_quantile':ENTROPY_QUANTILE,
            'mean':list(reference['mean']),'scales':list(reference['scales']),
            'normalized_reference':list(map(list,reference['reference'])),
            'calibration_distances':scores,'entropy_threshold':threshold,
            'calibration_entropies':ents,
            'excludes_test_labels':True,'test_ood_families_constructed':False}


def calibrated_distance(record,case):
    reference={'mean':record['mean'],'scales':record['scales'],
               'reference':record['normalized_reference']}
    dist=distance(features(case),reference)
    return dist,conformal_pvalue(dist,record['calibration_distances'])


def decision(record,case,ood_p,measured_entropy=None):
    """True => certified symbolic fallback. OOD early exit skips neural.

    Input confidence is only consulted when OOD p is above the cutoff.
    """
    if ood_p<=record['ood_alpha']:return True,'ood'
    if measured_entropy is None:raise ValueError('entropy required for in-domain sample')
    return (measured_entropy>record['entropy_threshold'],
            'entropy' if measured_entropy>record['entropy_threshold'] else 'neural')


def cheap_work_proxy(record):
    """Nominal scalar operations for nearest-neighbor distance, separate from model MACs."""
    return 3 * len(record['normalized_reference']) * FEATURES
