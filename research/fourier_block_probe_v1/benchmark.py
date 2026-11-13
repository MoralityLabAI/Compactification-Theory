"""FPB-01: held-out finite Fourier probes for a parity-lifted hidden mode.

Synthetic source: 17-bit Rubinstein+sentinel h; observed inputs are two
independently uniform 17-bit registers x,y and target (-1)^h(x xor y).
A first-order Fourier probe sees singleton sign features. Paired Fourier
probes see sign(x_i)sign(y_i), or their predeclared two-coordinate products.
All probe coefficients are calibrated on train samples only.

This program does not measure trained neural model modes, cognition, or
sentience. It tests a diagnostic intervention/representation pipeline.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import random
from pathlib import Path

try:
    from .reference import (
        BASE_N, HIDDEN_N, SENTINEL, hidden_fourier, lifted_sign,
        exact_base_rejection_count,
    )
except ImportError:
    from reference import (
        BASE_N, HIDDEN_N, SENTINEL, hidden_fourier, lifted_sign,
        exact_base_rejection_count,
    )

SEED = 940132192
TRAIN = 4096
TEST = 4096
PAIRS = tuple((4*b+i,4*b+j) for b in range(4)
              for i in range(4) for j in range(i+1,4))
ALL_PAIRS = tuple((1<<i,1<<i) for i in range(HIDDEN_N))
QUADRATIC_PAIRS = tuple((1<<i|1<<j,1<<i|1<<j) for i,j in PAIRS)
RAW_SINGLETONS = tuple((1<<i,0) for i in range(HIDDEN_N)) + tuple(
    (0,1<<i) for i in range(HIDDEN_N))
SENTINEL_PAIR = ((SENTINEL,SENTINEL),)


def draw(seed: int, count: int):
    if type(seed) is not int or type(count) is not int or count<1:
        raise ValueError("positive count and integer seed required")
    rng=random.Random(seed)
    return [(rng.getrandbits(HIDDEN_N),rng.getrandbits(HIDDEN_N))
            for _ in range(count)]


def signed_feature(left: int, right: int, masks: tuple[int,int]) -> int:
    a,b=masks
    return -1 if ((left&a).bit_count()+(right&b).bit_count())%2 else 1


def estimate_coefficients(data, features):
    if not data:
        raise ValueError("empty training dataset")
    denom=len(data)
    labels=[lifted_sign(x,y) for x,y in data]
    mean=Fraction(sum(labels),denom)
    coefficients=[
        Fraction(sum(label*signed_feature(x,y,masks)
                     for label,(x,y) in zip(labels,data)),denom)
        for masks in features
    ]
    return mean,coefficients


def score_model(data, mean, coeffs, features):
    if not data:
        raise ValueError("empty test dataset")
    if len(coeffs)!=len(features):
        raise ValueError("coefficient count mismatch")
    n=len(data)
    total_mse=Fraction()
    total_correct=0
    for x,y in data:
        actual=lifted_sign(x,y)
        predicted=mean+sum(
            (coef*signed_feature(x,y,masks)
             for coef,masks in zip(coeffs,features)),Fraction())
        total_mse+=(actual-predicted)**2
        guessed=1 if predicted>=0 else -1
        total_correct+=int(guessed==actual)
    return {
        "mse":str(total_mse/n),
        "mse_float":float(total_mse/n),
        "classification_accuracy":str(Fraction(total_correct,n)),
        "classification_accuracy_float":total_correct/n,
        "correct":total_correct,"evaluated":n,
    }


def registered_methods():
    return {
        "bias_only":(),
        "raw_34_singletons":RAW_SINGLETONS,
        "paired_sentinel":SENTINEL_PAIR,
        "paired_all_17":ALL_PAIRS,
        "paired_17_plus_block_fourth_order":
            ALL_PAIRS+QUADRATIC_PAIRS,
    }


def exact_population():
    reject=Fraction(exact_base_rejection_count(),1<<BASE_N)
    mu=hidden_fourier(0)
    pair=hidden_fourier(SENTINEL)
    if pair!=reject or mu!=reject-1:
        raise AssertionError("exact source prior/paired coefficient mismatch")
    baseline_accuracy=1-reject/2
    paired_accuracy=(1+reject)/2
    fourth_coefficients=[hidden_fourier((1<<i)|(1<<j)) for i,j in PAIRS]
    if len(fourth_coefficients)!=24 or set(map(abs,fourth_coefficients))!={Fraction(343,4096)}:
        raise AssertionError("exact fourth-order feature basis changed")
    fourth_population_mse=(1-mu**2-pair**2-
                           sum((c*c for c in fourth_coefficients),Fraction()))
    return {
        "mean_sign":str(mu),
        "singleton_coefficient_max_abs":"0",
        "paired_sentinel":str(pair),
        "baseline_mse":str(1-mu**2),
        "paired_sentinel_mse":str(1-mu**2-pair**2),
        "exact_pair_mse_gain":str(pair**2),
        "constant_mode_accuracy":str(baseline_accuracy),
        "paired_sentinel_accuracy":str(paired_accuracy),
        "paired_classification_accuracy_gain":
            str(paired_accuracy-baseline_accuracy),
        "fourth_order_pair_coefficient_0_1":str(hidden_fourier(3)),
        "fourth_order_pair_count":len(fourth_coefficients),
        "oracle_41_feature_mse":str(fourth_population_mse),
    }


def run(seed=SEED,train_count=TRAIN,test_count=TEST):
    if type(seed) is not int or train_count<100 or test_count<100:
        raise ValueError("integer seed, at least 100 calibration and 100 test")
    training=draw(seed,train_count)
    testing=draw(seed+1,test_count)
    if training==testing:
        raise AssertionError("train/test generated identical input sequence")
    methods={}
    for name,features in registered_methods().items():
        mean,coeffs=estimate_coefficients(training,features)
        metrics=score_model(testing,mean,coeffs,features)
        methods[name]={
            "features":len(features),
            "calibration_mean":str(mean),
            "calibration_coefficients":[str(v) for v in coeffs],
            "heldout":metrics,
        }
    exact=exact_population()
    report={
        "schema":"fpb01.paired_fourier_probe.v1",
        "status":"FROZEN_SYNTHETIC_PARITY_MODE",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256":hashlib.sha256(
            Path(__file__).with_name("reference.py").read_bytes()).hexdigest(),
        "seed":seed,"train_count":train_count,"test_count":test_count,
        "source_hidden_bits":HIDDEN_N,"observed_input_bits":2*HIDDEN_N,
        "train_data_sha256":hashlib.sha256(json.dumps(training).encode()).hexdigest(),
        "test_data_sha256":hashlib.sha256(json.dumps(testing).encode()).hexdigest(),
        "label_definition":"(-1)^h17(x xor y)",
        "linear_readout_estimator":"sample Fourier coefficients + calibration intercept",
        "full_population_exact":exact,
        "methods":methods,
        "evaluates_trained_neural_checkpoint":False,
        "observes_internal_chain_of_thought":False,
        "synthetic_hidden_mode":True,
    }
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--train",type=int,default=TRAIN)
    p.add_argument("--test",type=int,default=TEST)
    p.add_argument("--out",type=Path)
    args=p.parse_args(argv)
    report=run(args.seed,args.train,args.test)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":report["status"],
        "seed":report["seed"],
        "population":report["full_population_exact"],
        "heldout":{k:{"mse":v["heldout"]["mse_float"],
                      "accuracy":v["heldout"]["classification_accuracy_float"],
                      "n_features":v["features"]}
                   for k,v in report["methods"].items()},
    },indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
