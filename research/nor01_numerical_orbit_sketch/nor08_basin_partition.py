"""NOR-08C: predeclared held-out basin labels and hidden-coordinate intervention.

The original trained RRC-25 recurrent checkpoint is frozen. Public
nine-row source inputs from each of its six TEST_FAMILIES are converted
into 12D hidden states with source-native logits parity.

Each state follows a 720-step zero-input continuation and is assigned
a basin mode by distance to the TWO rigorously certified fixed-point
centers. A predeclared hidden-coordinate-0 sign-flip intervention
repeats the continuation. Mode labels are empirical FP64 indicators;
the mathematical source-to-basin certificates remain separate.

No model fitting, source future labels, hidden programs or candidate
verifier outputs enter classification or intervention selection.
"""
from __future__ import annotations

import argparse
import json
import math
from hashlib import sha256
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for directory in (ROOT,HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

from research.rrc_compactification import bench25,compound25
from research.nor01_numerical_orbit_sketch import nor06_basin_survey as survey
from research.nor01_numerical_orbit_sketch import trained_gru_cert as native
from research.nor01_numerical_orbit_sketch.invariant_ball import CENTER as P1

PAPER=HERE/"results"/"nor07_two_attractors_summary.json"
SOURCE_SHA=survey.KNOWN_SHA
EVAL_SEED_OFFSET=880088
FAMILY_MULTIPLIER=10007
SAMPLES_PER_FAMILY=12
ZERO_STEPS=720
MARGIN=0.05
PERTURBED_COORDINATE=0


def locked_model():
    models,checksums=bench25.checkpoints()
    if checksums["recurrent"]!=SOURCE_SHA:
        raise AssertionError("trained checkpoint changed")
    receipt=json.loads(PAPER.read_text())
    assert receipt["original_checkpoint_sha256"]==SOURCE_SHA
    assert receipt["two_locally_attracting_fixed_points_certified"]
    p2=tuple(float(x) for x in receipt["second_attractor"]["center_decimals"])
    return models["recurrent"],tuple(float(x) for x in P1),p2


def cohort(samples_per_family=SAMPLES_PER_FAMILY):
    if not 1<=samples_per_family<=48:
        raise ValueError("samples per family 1..48 required")
    cases=[]
    all_visibles=set()
    for ix,family in enumerate(compound25.TEST_FAMILIES):
        rng=random.Random(compound25.SEED+EVAL_SEED_OFFSET+
                          FAMILY_MULTIPLIER*ix)
        found=0
        for trial in range(200*samples_per_family):
            case=compound25.make_case(
                rng,family,f"nor08c-{family}-{trial:05d}")
            if case.visible in all_visibles:
                continue
            if not compound25.possible_headers(case):
                continue
            all_visibles.add(case.visible)
            cases.append(case)
            found+=1
            if found==samples_per_family:
                break
        if found!=samples_per_family:
            raise AssertionError("insufficient independent source cases")
    return cases


def distance(h,p):
    return math.sqrt(math.fsum((u-v)**2 for u,v in zip(h,p,strict=True)))


def predict_after_hidden(h,checkpoint,center1,center2,steps):
    for _ in range(steps):
        h=native.native_zero_step(h,checkpoint)
    d1,d2=distance(h,center1),distance(h,center2)
    if min(d1,d2)>MARGIN:
        label="unresolved"
    else:
        label="first" if d1<d2 else "second"
    return label,d1,d2


def run(samples_per_family=SAMPLES_PER_FAMILY,steps=ZERO_STEPS):
    if not 24<=steps<=2000:
        raise ValueError("steps 24..2000 required")
    checkpoint,p1,p2=locked_model()
    cases=cohort(samples_per_family)
    records=[]
    matrix={}
    interventions={"first_to_second":0,"second_to_first":0,
                   "same_mode":0,"other":0}
    for case in cases:
        h,parity,heads=survey.prefix_hidden(case,checkpoint)
        original=predict_after_hidden(h[:],checkpoint,p1,p2,steps)
        perturbed=h[:]
        perturbed[PERTURBED_COORDINATE]=-perturbed[PERTURBED_COORDINATE]
        flipped=predict_after_hidden(perturbed,checkpoint,p1,p2,steps)
        original_mode, d1,d2=original
        flip_mode,fd1,fd2=flipped
        key=case.family
        count=matrix.setdefault(key,{"first":0,"second":0,"unresolved":0})
        count[original_mode]+=1
        edge=(original_mode,flip_mode)
        if edge==("first","second"):
            interventions["first_to_second"]+=1
        elif edge==("second","first"):
            interventions["second_to_first"]+=1
        elif original_mode==flip_mode:
            interventions["same_mode"]+=1
        else:
            interventions["other"]+=1
        records.append({
            "family":key,
            "case_id_sha256":sha256(case.identifier.encode()).hexdigest(),
            "public_visible_sha256":sha256(json.dumps(case.visible,separators=(",",":")).encode()).hexdigest(),
            "native_head_parity":parity,
            "public_header_candidates":heads,
            "mode":original_mode,
            "first_center_distance":d1,
            "second_center_distance":d2,
            "flipped_coordinate":PERTURBED_COORDINATE,
            "intervention_mode":flip_mode,
            "intervention_first_distance":fd1,
            "intervention_second_distance":fd2,
        })
    total=len(records)
    diag=sum(matrix[family]["second"] for family in ("squared",))
    diag+=sum(matrix[f]["first"] for f in matrix if f!="squared")
    unresolved=sum(row["mode"]=="unresolved" for row in records)
    return {
        "schema":"nor08c.heldout.basin_partition_and_coordinate_causality.v1",
        "trained_checkpoint_sha256":SOURCE_SHA,
        "source_family_seed":"compound25.SEED+880088+10007*family_index",
        "cohort_size":total,
        "samples_per_family":samples_per_family,
        "zero_input_steps":steps,
        "mode_decision_distance_threshold":MARGIN,
        "intervention":"flip hidden coordinate 0 sign immediately after nine public input rows",
        "contingency":matrix,
        "coordinate_intervention_counts":interventions,
        "squared_versus_rest_mode_rule_accuracy":diag/total,
        "unresolved_mode_count":unresolved,
        "unconditioned_majority_family_baseline_accuracy":5/6,
        "source_future_labels_used":False,
        "new_training_performed":False,
        "exact_source_to_second_basin_capture_proved":False,
        "records":records,
        "scope":"heldout synthetic source test-family basin distribution; FP64 geometry, not formal source-entry or proof-ranking performance",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples-per-family",type=int,default=SAMPLES_PER_FAMILY)
    ap.add_argument("--steps",type=int,default=ZERO_STEPS)
    ap.add_argument("--check",action="store_true")
    ap.add_argument("--out",type=Path)
    a=ap.parse_args()
    rec=run(a.samples_per_family,a.steps)
    if a.check:
        assert rec["cohort_size"]==6*a.samples_per_family
        assert rec["trained_checkpoint_sha256"]==SOURCE_SHA
        assert not rec["source_future_labels_used"]
        assert not rec["new_training_performed"]
        assert rec["exact_source_to_second_basin_capture_proved"] is False
        assert len({r["public_visible_sha256"] for r in rec["records"]})==rec["cohort_size"]
        assert sum(rec["coordinate_intervention_counts"].values())==rec["cohort_size"]
    blob=json.dumps(rec,sort_keys=True,indent=2)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(blob)
    else:
        print(blob,end="")


if __name__=="__main__":
    main()
