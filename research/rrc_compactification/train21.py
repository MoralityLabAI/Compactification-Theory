"""RRC-21: independent known-only split calibration of frozen graph predictors.

Zero model updates. Exact class representatives and three prior checkpoints are
frozen. No held-out/negative/collision generator is reachable from fit().
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

try:
    from . import select21 as d, risk21 as r, bench19, learn18 as l, model18 as m
except ImportError:
    import select21 as d
    import risk21 as r
    import bench19
    import learn18 as l
    import model18 as m

HERE=Path(__file__).resolve().parent
MODEL_ROLES=('balanced_recurrent','skewed_recurrent','balanced_feedforward')
SOURCE_FILES=('select21.py','risk21.py','train21.py','family19.py','family20.py',
              'gate20.py','model18.py','learn18.py','certify18.py')


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def frozen_models():
    models,_,digests=bench19.checkpoints()
    return {k:models[k] for k in MODEL_ROLES},{k:digests[k] for k in MODEL_ROLES}


def fit():
    roster=d.dataset(include_test=False)
    reference=roster['reference'];calibration=roster['calibration']
    models,digests=frozen_models();classes=l.class_map()
    entropies=[]
    for case in calibration:
        logits={role:m.logits(case,models[role]) for role in MODEL_ROLES}
        entropies.append(r.entropy(case,logits,classes))
    record=r.index(calibration,reference,entropies)
    def normal_score(case):return r.distance(r.features(case),r.fit_reference(reference))
    dists=[normal_score(case) for case in calibration]
    total=sum(not (r.conformal_pvalue(score,record['calibration_distances'])>record['ood_alpha']
                  and entropy<=record['entropy_threshold'])
              for score,entropy in zip(dists,entropies))
    record.update({'source_sha256':source_hashes(),'base_model_sha256':digests,
                   'uses_only_known_family_reference_and_calibration':True,
                   'calibration_defer_count':total,
                   'calibration_defer_rate':total/len(calibration),
                   'calibration_ood_reject_count':sum(r.conformal_pvalue(score,record['calibration_distances'])<=record['ood_alpha'] for score in dists),
                   'calibration_entropy_high_count':sum(v>record['entropy_threshold'] for v in entropies),
                   'neural_updates':0,'frozen_model_updates':0,'calibration_ground_truth_used':False,
                   'full_graph_interpreter_previews_during_calibration':0})
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'results'/'rrc21_calibration.json')
    args=parser.parse_args()
    result=fit()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:result[key] for key in ('reference_count','calibration_count','calibration_defer_count','calibration_defer_rate','calibration_ood_reject_count','entropy_threshold')},indent=2))

if __name__=='__main__':main()
