"""NOR-06: source-locked basin-entry survey for the trained RRC-25 GRU.

This is a deterministic FP64 *discovery* survey for trajectories generated
from the original model's public nine-row inputs. It measures how many
zero-input recurrent steps are needed to approach the NOR-04/05 certified
fixed-point center. The survey never claims interval-certified basin
membership; all positive hits are promotion targets for an independent
rational verifier.

No training, checkpoint rewrite, secret labels, future query or paid model
inference is used to select the six public source-family cases.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import bench25, compound25, model25
from research.nor01_numerical_orbit_sketch import trained_gru_cert as cert
from research.nor01_numerical_orbit_sketch.invariant_ball import CENTER,RADIUS

SEED_OFFSET=903900
FAMILY_STEP=1009
FAMILIES=tuple(compound25.TEST_FAMILIES)
THRESHOLDS=(1e-2,1e-4,1e-6,1e-8,1e-10,1e-11)
MAX_STEPS=4800
KNOWN_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def public_cases():
    cases=[]
    seen=set()
    for ix,fam in enumerate(FAMILIES):
        rng=random.Random(compound25.SEED + SEED_OFFSET + FAMILY_STEP*ix)
        for attempt in range(32):
            case=compound25.make_case(rng,fam,f"nor03-{fam}-{attempt:02d}")
            if case.visible not in seen and compound25.possible_headers(case):
                cases.append(case)
                seen.add(case.visible)
                break
        else:
            raise AssertionError("missing public case for "+fam)
    assert len(cases)==6
    return cases


def prefix_hidden(case, checkpoint):
    headers=compound25.possible_headers(case)
    if not headers:
        raise AssertionError("public header absent")
    # Portable source implementation gate: the same recurrent reset/update
    # gate ordering as model25.logits and the earlier FHC-01C audit.
    hidden=[0.0]*model25.HIDDEN
    weights=checkpoint["state_dict"]
    for row in compound25.features(case,headers[0]):
        a=model25.affine(weights["rnn.weight_ih_l0"],weights["rnn.bias_ih_l0"],row)
        v=model25.affine(weights["rnn.weight_hh_l0"],weights["rnn.bias_hh_l0"],hidden)
        next_state=[]
        for i in range(model25.HIDDEN):
            reset=model25.sigmoid(a[i]+v[i])
            update=model25.sigmoid(a[i+model25.HIDDEN]+v[i+model25.HIDDEN])
            new=math.tanh(a[i+2*model25.HIDDEN]+reset*v[i+2*model25.HIDDEN])
            next_state.append((1-update)*new+update*hidden[i])
        hidden=next_state
    expected=model25.logits(case,headers[0],checkpoint)
    observed=model25.affine(weights["head.weight"],weights["head.bias"],hidden)
    parity=max(abs(x-y) for x,y in zip(expected,observed,strict=True))
    assert parity<1e-11,(case.family,parity)
    return hidden,parity,len(headers)


def point_distance(state):
    return math.sqrt(math.fsum((h-float(c))**2 for h,c in zip(state,CENTER,strict=True)))


def survey_one(case, checkpoint, max_steps):
    h,parity,header_count=prefix_hidden(case,checkpoint)
    start=point_distance(h)
    prev=start
    hits={format(x,".0e"):None for x in THRESHOLDS}
    every12=[]
    first=None
    min_distance=start
    min_step=0
    for t in range(max_steps+1):
        d=point_distance(h)
        if d<min_distance:
            min_distance,min_step=d,t
        for threshold in THRESHOLDS:
            key=format(threshold,".0e")
            if hits[key] is None and d<=threshold:
                hits[key]=t
        if t%12==0:
            # Important: the norm is compared to the rational *center*,
            # whose distance to the actual fixed point is <=1.6534e-14.
            every12.append({"t":t,"center_distance":d,
                            "block_ratio_to_prior":d/prev if t else None})
            prev=d
        if first is None and d<float(RADIUS):
            first=t
        if t<max_steps:
            h=cert.native_zero_step(h,checkpoint)
    return {
        "family":case.family,
        "public_visible_sha256":sha256(json.dumps(case.visible,separators=(",",":")).encode()).hexdigest(),
        "case_identifier_sha256":sha256(case.identifier.encode()).hexdigest(),
        "public_header_candidate_count":header_count,
        "source_native_logits_parity":parity,
        "initial_center_distance":start,
        "first_measured_ball_entry_step":first,
        "threshold_hits":hits,
        "smallest_measured_center_distance":min_distance,
        "smallest_measured_center_distance_step":min_step,
        "last_center_distance":point_distance(h),
        "last_12_step_block_ratio":every12[-1]["block_ratio_to_prior"],
        "trajectory_stride":12,
        "early_strided_trace":every12[:min(13,len(every12))],
        "final_strided_trace":every12[-min(8,len(every12)):],
        "distance_to_actual_fixed_point_note":"center is within <=1.6534e-14 of certified fixed point; float path has no interval shadowing certificate",
    }


def run(max_steps=MAX_STEPS):
    if max_steps<24 or max_steps>12000:
        raise ValueError("max_steps requires 24..12000")
    trained,digests=bench25.checkpoints()
    ck=trained["recurrent"]
    assert ck["parameters"]==1000 and digests["recurrent"]==KNOWN_SHA
    cases=public_cases()
    rows=[survey_one(case,ck,max_steps) for case in cases]
    return {
        "schema":"nor06.rrc25.input_conditioned.basin_discovery.v1",
        "mode":"FP64 discovery only; basin membership remains unverified",
        "checkpoint_sha256":digests["recurrent"],
        "source_hashes_verified":ck["source_sha256"],
        "family_seeds":"compound25.SEED+903900+1009*family_index; first case with public header",
        "family_count":len(rows),
        "warmup_max_steps":max_steps,
        "fixed_point_center_decimals":[str(c) for c in CENTER],
        "certified_center_ball_radius_fraction":str(RADIUS),
        "found_measured_entries":sum(x["first_measured_ball_entry_step"] is not None for x in rows),
        "rows":rows,
        "limits":{
            "native_input_features":"nine public source feature rows, exact existing RRC25 header decoder",
            "input_truth_or_future_used":False,
            "rounding":"Python double precision",
            "certified_entry":False,
            "proposed_next_gate":"source-locked rational or interval-recentered shadowing of each claimed entry",
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--max-steps",type=int,default=MAX_STEPS)
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=run(args.max_steps)
    if args.check:
        assert result["family_count"]==6
        assert result["checkpoint_sha256"]==KNOWN_SHA
        assert not result["limits"]["certified_entry"]
        assert all(z["source_native_logits_parity"]<1e-11 for z in result["rows"])
        assert len(set(z["public_visible_sha256"] for z in result["rows"]))==6
    body=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(body)
    else:
        print(body,end="")


if __name__=="__main__":
    main()
