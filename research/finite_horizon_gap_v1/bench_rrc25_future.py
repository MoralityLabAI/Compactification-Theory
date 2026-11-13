"""FHC-01C: trained source-locked RRC-25 GRU future-logit geometry pilot.

Uses the committed recurrent GRU checkpoint in rrc_compactification/results,
verifies its original source/train hashes, independently recomputes the
nine-step GRU state, and then applies a *common zero-input continuation*
to expose a measured future-observation metric.

PCA uses only dev cases. Tests are separate original source-generator
families. This is a retrospective neural-state analysis, NOT a measure
of native proof-ranking accuracy at repeated zero inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.rrc_compactification import bench25 as receipt_gate
from research.rrc_compactification import compound25 as source
from research.rrc_compactification import model25 as network
from research.finite_horizon_gap_v1.bench_future_metric import (
    calibration_scale, distortion_metrics,
)

SCHEMA = "fhc01c.rrc25_trained_future_logits.v1"
SEED = 940199
HORIZONS = (1, 2, 4, 8, 16)
WIDTHS = (2, 4, 8, 12, 16)
TRAIN_PER, DEV_PER, TEST_PER = 1, 12, 8


def source_header(case):
    """Use only public observations; select first positive public header."""
    candidates = source.possible_headers(case)
    if not candidates:
        raise RuntimeError("source case has no publicly inferred header")
    return candidates[0], len(candidates)


def gru_step(h: list[float], row: list[float], checkpoint: dict) -> list[float]:
    s = checkpoint["state_dict"]
    a = network.affine(s["rnn.weight_ih_l0"], s["rnn.bias_ih_l0"], row)
    v = network.affine(s["rnn.weight_hh_l0"], s["rnn.bias_hh_l0"], h)
    result = []
    for i in range(network.HIDDEN):
        reset = network.sigmoid(a[i] + v[i])
        update = network.sigmoid(a[i + network.HIDDEN] + v[i + network.HIDDEN])
        new = math.tanh(a[i + 2*network.HIDDEN] + reset*v[i + 2*network.HIDDEN])
        result.append((1 - update)*new + update*h[i])
    return result


def logits(h: list[float], checkpoint: dict) -> list[float]:
    s=checkpoint["state_dict"]
    return network.affine(s["head.weight"], s["head.bias"], h)


def native_case(case, checkpoint: dict, max_horizon: int):
    header, head_count = source_header(case)
    inputs = source.features(case, header)
    if len(inputs) != 9 or any(len(row) != 8 for row in inputs):
        raise RuntimeError("original RRC25 public features changed shape")
    h = [0.] * network.HIDDEN
    for row in inputs:
        h = gru_step(h, row, checkpoint)
    direct = network.logits(case, header, checkpoint)
    recovered = logits(h, checkpoint)
    parity = max(abs(a-b) for a,b in zip(direct, recovered))
    if parity > 1e-11:
        raise AssertionError("trained RRC25 native readout parity failed")
    hidden = h[:]
    future = []
    zero = [0.] * 8
    for _ in range(max_horizon):
        h = gru_step(h, zero, checkpoint)
        future.extend(logits(h, checkpoint))
    if not all(math.isfinite(v) for v in hidden + future):
        raise RuntimeError("nonfinite trained recurrent state or logits")
    return hidden, future, parity, head_count


def _stack_cases(cases: list, checkpoint: dict, max_horizon: int):
    hidden, future = [], []
    parity_max = 0.0
    multicandidate = 0
    for case in cases:
        h, f, parity, n = native_case(case, checkpoint, max_horizon)
        hidden.append(h)
        future.append(f)
        parity_max = max(parity_max, parity)
        multicandidate += int(n != 1)
    return (np.asarray(hidden, dtype=np.float64),
            np.asarray(future, dtype=np.float64),
            parity_max, multicandidate)


def _fit_methods(train, test, hidden_train, hidden_test, T: int, d_list,
                 seed: int) -> list[tuple]:
    dims=16*T
    methods=[
        ("native_hidden_12", hidden_train, hidden_test, {
            "coordinates": network.HIDDEN,
            "stored_state_bits_fp32": network.HIDDEN*32,
            "native_recurrent_decoder": True,
            "trained": "frozen RRC25 original training",
            "oracle_future_queries_to_form_code": 0,
        }),
        ("full_trace_identity", train.copy(), test.copy(), {
            "coordinates":dims,
            "oracle_future_queries_to_form_code":T,
            "projection_parameters":0,
            "trained":False,
        }),
    ]
    center = np.mean(train,axis=0,keepdims=True)
    _,_,vt=np.linalg.svd(train-center, full_matrices=False)
    for d in d_list:
        if d > dims:
            continue
        if d <= len(vt):
            matrix=vt[:d].T
            methods.append((f"pca_{d}",(train-center)@matrix,
                            (test-center)@matrix,{
                "coordinates":d,"trained":"PCA on dev cases only",
                "fit_cases":len(train),
                "projection_parameters":dims*d,
                "mean_parameters":dims,
                "oracle_future_queries_to_form_code":T,
            }))
        rng=np.random.default_rng(seed+1000+T*101+d)
        matrix=rng.standard_normal((dims,d))/math.sqrt(d)
        methods.append((f"gaussian_{d}",(train-center)@matrix,
                        (test-center)@matrix,{
            "coordinates":d,"trained":False,
            "projection_parameters":dims*d,
            "oracle_future_queries_to_form_code":T,
        }))
    return methods


def run(seed: int = SEED, horizons=HORIZONS, widths=WIDTHS) -> dict:
    if (not horizons or not widths or any(t < 1 or t > 16 for t in horizons)
            or any(d < 1 or d > 128 for d in widths)):
        raise ValueError("invalid frozen horizon or projection widths")
    trained,digests=receipt_gate.checkpoints()
    checkpoint=trained["recurrent"]
    if checkpoint["parameters"] != 1000:
        raise AssertionError("source recurrent checkpoint parameter count changed")
    roster=source.roster(include_test=True,train_per=TRAIN_PER,
                         dev_per=DEV_PER,test_per=TEST_PER)
    calibration=roster["dev"]
    heldout=[x for fam in source.TEST_FAMILIES for x in roster[fam]]
    if len(calibration)!=3*DEV_PER or len(heldout)!=6*TEST_PER:
        raise RuntimeError("RRC25 source roster lengths changed")
    if set(x.identifier for x in calibration)&set(x.identifier for x in heldout):
        raise AssertionError("RRC25 source identities leaked across splits")

    max_horizon=max(horizons)
    htr, ftr, ptrain, ctrain=_stack_cases(calibration,checkpoint,max_horizon)
    hte, fte, ptest, ctest=_stack_cases(heldout,checkpoint,max_horizon)
    cases=[]
    family_order=[x.family for x in heldout]
    for T in horizons:
        dim=16*T
        train=ftr[:,:dim]
        test=fte[:,:dim]
        methods=_fit_methods(train,test,htr,hte,T,widths,seed)
        scored=[]
        for name,code_train,code_test,ledger in methods:
            scale=calibration_scale(train,code_train)
            ids=np.arange(len(heldout),dtype=np.int64)
            score=distortion_metrics(test,code_test,scale,ids)
            mapped={}
            if "worst_pair" in score:
                mapped["worst_pair_case_ids"]=[
                    heldout[index].identifier for index in score["worst_pair"]]
            for k,witness in score["violation_witnesses"].items():
                witness["case_a_id"]=heldout[witness["state_a"]].identifier
                witness["case_b_id"]=heldout[witness["state_b"]].identifier
            by_family={}
            for fam in source.TEST_FAMILIES:
                mask=np.array([i for i,f in enumerate(family_order) if f==fam],
                              dtype=np.int64)
                try:
                    sub=distortion_metrics(test[mask],code_test[mask],
                                           scale,mask)
                    by_family[fam]={
                        "pairs":sub["pairs_distinct"],
                        "D2_calibrated_violations":
                            sub["violation_counts_calibration_scale"]["2.0"],
                        "test_optimal_D":
                            sub["optimal_rescaling_distortion_heldout"]}
                except ValueError:
                    by_family[fam]={"reason":"no_distinct_future_pairs"}
            scored.append({
                "method":name,
                "dimension":int(code_train.shape[1]),
                "calibration_scale":scale,
                "ledger":ledger,
                "heldout":score,
                "by_family":by_family,
                **mapped,
            })
        cases.append({"horizon":T,"future_feature_dimension":dim,
                      "methods":scored})
    path=Path(__file__)
    return {
        "schema":SCHEMA,
        "status":"FROZEN_TRAINED_RRC25_DEVELOPMENT",
        "neural_model":"RRC25 portable trained 1000-parameter GRU",
        "checkpoint_sha256":digests["recurrent"],
        "checkpoint_source_hashes":checkpoint["source_sha256"],
        "experiment_source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_reuse":"bench25.checkpoints original SHA-256 source gate",
        "calibration_roster_sha256":source.fingerprint(calibration),
        "heldout_roster_sha256":source.fingerprint(heldout),
        "train_per":TRAIN_PER,
        "dev_per":DEV_PER,
        "test_per":TEST_PER,
        "seed":seed,
        "hidden_width":network.HIDDEN,
        "network_parameters":checkpoint["parameters"],
        "native_head_parity_max_abs_error":max(ptrain,ptest),
        "public_header_multiple_candidates":{
            "calibration":ctrain,"heldout":ctest},
        "calibration_case_ids":[x.identifier for x in calibration],
        "heldout_cases":[{"id":x.identifier,"family":x.family}
                         for x in heldout],
        "continuation":"frozen GRU with 8-vector zeros after nine native feature rows",
        "continuation_is_native_task_training_distribution":False,
        "horizons":list(horizons),
        "widths":list(widths),
        "cases":cases,
        "full_space_geometry_certificate":False,
        "source_family_098_infinite_witness_tested":False,
    }


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--horizons",type=int,nargs="+",default=list(HORIZONS))
    p.add_argument("--widths",type=int,nargs="+",default=list(WIDTHS))
    a=p.parse_args(argv)
    report=run(seed=a.seed,horizons=tuple(a.horizons),widths=tuple(a.widths))
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(report,indent=2,sort_keys=True,
                                     allow_nan=False)+"\n")
    compact=[]
    for row in report["cases"]:
        out={"T":row["horizon"],"methods":{}}
        for method in row["methods"]:
            if method["method"] in (
                "native_hidden_12","full_trace_identity",
                "pca_4","pca_8","pca_12","gaussian_4","gaussian_8",
                "gaussian_12"
            ):
                out["methods"][method["method"]]={
                    "heldout_D":method["heldout"]["optimal_rescaling_distortion_heldout"],
                    "calibrated_D2_fails":
                        method["heldout"]["violation_counts_calibration_scale"]["2.0"],
                    "pair":method.get("worst_pair_case_ids"),
                }
        compact.append(out)
    print(json.dumps({"status":report["status"],
                      "checkpoint_sha256":report["checkpoint_sha256"],
                      "max_native_logit_parity_error":
                          report["native_head_parity_max_abs_error"],
                      "cases":compact},indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
