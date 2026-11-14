"""KSR-01B: reproducible small supervised count monitor for noisy trees.

Fresh 15-parameter multiclass softmax model trained only on synthetic
root/leaf-count examples. All generation follows the exact q-state
symmetric broadcast law, with roots and entire trees split by seed.
Python standard library only; no pretrained neural checkpoint.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import struct

try:
    from .reference import (
        bayes_floor, bit_budget, exact_broadcast, sparse_schedule,
    )
except ImportError:
    from reference import bayes_floor, bit_budget, exact_broadcast, sparse_schedule


SCHEMA = "ksr01b.learned_count_monitor.v1"
STATUS = "NEWLY_TRAINED_SYNTHETIC_LOGISTIC_TREE_MONITOR"
Q = 3
B = 2
LAM = Fraction(4,5)
DATA_SEED = 940229236
SEEDS = (11,19)
EPOCHS = 6
LR = 0.15
TRAIN = 4096
TEST = 4096
CELLS = (
    ("complete_depth3",3,Fraction(1)),
    ("half_depth2",2,Fraction(1,2)),
    ("sparse_depth2",2,Fraction(625,1024)),
)
CONDITIONS = ("supervised", "random_legal_labels")


def draw_tree(rng:random.Random, q:int, b:int,
              lam:Fraction,depth:int,retention:Fraction):
    if not 0 < lam <=1 or not 0 < retention <=1 or depth<0:
        raise ValueError("learner requires positive eigenvalue and retention")
    root=rng.randrange(q)
    layer=[root]
    for _ in range(depth):
        next_layer=[]
        for parent in layer:
            for _ in range(b):
                child=(parent if rng.random()<float(lam)
                       else rng.randrange(q))
                next_layer.append(child)
        layer=next_layer
    counts=[0]*q
    for leaf in layer:
        if rng.random()<float(retention):
            counts[leaf]+=1
    return {"root":root,"counts":counts}


def dataset(seed:int,count:int,depth:int,retention:Fraction)->list[dict]:
    if type(seed) is not int or type(count) is not int or count<1:
        raise ValueError("integer seed and count>=1 required")
    rng=random.Random(seed)
    return [draw_tree(rng,Q,B,LAM,depth,retention) for _ in range(count)]


def features(counts:list[int],leaves:int)->tuple[float,...]:
    if len(counts)!=Q or leaves<1 or any(type(c) is not int or c<0 for c in counts):
        raise ValueError("invalid count vector")
    if sum(counts)>leaves:
        raise ValueError("more observed leaves than possible")
    return (*[x/leaves for x in counts],sum(counts)/leaves,1.0)


def softmax(logits:list[float])->list[float]:
    maximum=max(logits)
    shifted=[math.exp(x-maximum) for x in logits]
    denom=sum(shifted)
    return [x/denom for x in shifted]


def predict(weights:list[list[float]], counts:list[int],leaves:int):
    feats=features(counts,leaves)
    logits=[sum(w*v for w,v in zip(row,feats)) for row in weights]
    ps=softmax(logits)
    return max(range(Q),key=lambda k:(ps[k],-k)),ps


def model_hash(weights:list[list[float]])->str:
    h=hashlib.sha256()
    for row in weights:
        for v in row:
            h.update(struct.pack("!d",v))
    return h.hexdigest()


def train(rows, *,leaves:int,seed:int,condition:str,
          epochs:int=EPOCHS,learning_rate:float=LR):
    if condition not in CONDITIONS or type(seed) is not int:
        raise ValueError("invalid training condition or seed")
    if type(epochs) is not int or epochs<1 or learning_rate<=0:
        raise ValueError("invalid optimizer budget")
    if not rows:
        raise ValueError("empty training examples")
    rng=random.Random(DATA_SEED+seed+919)
    y=[row["root"] for row in rows]
    if condition=="random_legal_labels":
        labels_random=random.Random(DATA_SEED+seed+193)
        y=[labels_random.randrange(Q) for _ in rows]
    W=[[0.0]*(Q+2) for _ in range(Q)]
    order=list(range(len(rows)))
    train_examples=[features(r["counts"],leaves) for r in rows]
    final_loss=0.0
    for _ in range(epochs):
        rng.shuffle(order)
        total_loss=0.0
        for idx in order:
            x=train_examples[idx]
            logits=[sum(w*z for w,z in zip(row,x)) for row in W]
            probabilities=softmax(logits)
            total_loss-=math.log(max(probabilities[y[idx]],1e-300))
            for cls in range(Q):
                grad=probabilities[cls]-(cls==y[idx])
                for j,z in enumerate(x):
                    W[cls][j]-=learning_rate*grad*z
        final_loss=total_loss/len(rows)
    train_correct=sum(predict(W,r["counts"],leaves)[0]==r["root"] for r in rows)
    return W, {
        "condition":condition,"seed":seed,
        "num_parameters":Q*(Q+2),
        "epochs":epochs,"train_examples":len(rows),
        "gradient_steps":len(rows)*epochs,
        "learning_rate":learning_rate,
        "model_sha256":model_hash(W),
        "final_train_label_loss":final_loss,
        "train_true_root_accuracy":train_correct/len(rows),
    }


def measure(weights,rows,leaves:int):
    correct=0;loss=0.0
    for row in rows:
        pred,p=predict(weights,row["counts"],leaves)
        correct+=int(pred==row["root"])
        loss-=math.log(max(p[row["root"]],1e-300))
    return {"correct":correct,"n":len(rows),
            "accuracy":correct/len(rows),
            "mean_log_loss":loss/len(rows)}


def plurality(rows):
    if not rows:
        raise ValueError("no observations")
    correct=0
    for row in rows:
        counts=row["counts"]
        pred=max(range(Q),key=lambda j:(counts[j],-j))
        correct+=int(pred==row["root"])
    return {"correct":correct,"n":len(rows),"accuracy":correct/len(rows)}


def data_hash(rows)->str:
    return hashlib.sha256(json.dumps(
        rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def run(train_count:int=TRAIN,test_count:int=TEST,
        seeds=SEEDS,conditions=CONDITIONS,epochs:int=EPOCHS):
    if (type(train_count) is not int or type(test_count) is not int
            or train_count<64 or test_count<64):
        raise ValueError("cohort sizes must be >=64")
    results=[]
    for i,(name,depth,r) in enumerate(CELLS):
        leaves=B**depth
        train_rows=dataset(DATA_SEED+i*1011,train_count,depth,r)
        test_rows=dataset(DATA_SEED+1+i*1011,test_count,depth,r)
        if train_rows==test_rows:
            raise AssertionError("whole-tree train/test leakage")
        oracle=exact_broadcast(Q,B,LAM,depth,r)
        baseline=plurality(test_rows)
        model_results=[]
        for seed in seeds:
            for condition in conditions:
                weights,receipt=train(train_rows,leaves=leaves,seed=seed,
                                      condition=condition,epochs=epochs)
                test_metric=measure(weights,test_rows,leaves)
                model_results.append({"train":receipt,"heldout":test_metric})
        results.append({
            "cell_id":name,"depth":depth,"retention":str(r),
            "leaves":leaves,
            "expected_observed_leaves":str(leaves*r),
            "counter_memory":bit_budget(Q,B,depth,r),
            "train_data_sha256":data_hash(train_rows),
            "test_data_sha256":data_hash(test_rows),
            "independent_tree_train_count":train_count,
            "independent_tree_test_count":test_count,
            "exact_count_bayes_lower":oracle["count_lower_bound"],
            "exact_count_bayes_population_MAP":oracle["count_MAP"],
            "exact_full_pattern_population_MAP":oracle["full_leaf_MAP"],
            "fixed_plurality":baseline,
            "learned_arms":model_results,
        })
    return {
        "schema":SCHEMA,
        "status":STATUS,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256":hashlib.sha256(
            Path(__file__).with_name("reference.py").read_bytes()).hexdigest(),
        "data_seed":DATA_SEED,"fit_seeds":list(seeds),
        "conditions":list(conditions),
        "training_epochs":epochs,
        "source_model":"freshly trained multinomial logistic on synthetic tree counts",
        "weights_pretrained":False,"reads_raw_tree_internal_states_at_test":False,
        "counter_bits_are_dynamic_memory_only":True,
        "q":Q,"b":B,"lambda":str(LAM),
        "cases":results,
    }


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=int,default=TRAIN)
    p.add_argument("--test",type=int,default=TEST)
    p.add_argument("--epochs",type=int,default=EPOCHS)
    p.add_argument("--out",type=Path)
    args=p.parse_args(argv)
    receipt=run(args.train,args.test,epochs=args.epochs)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":receipt["status"],
        "cells":[{
            "name":c["cell_id"],"depth":c["depth"],
            "retention":c["retention"],
            "population_count_MAP":float(Fraction(c["exact_count_bayes_population_MAP"])),
            "population_bound":float(Fraction(c["exact_count_bayes_lower"])),
            "heldout_plurality":c["fixed_plurality"]["accuracy"],
            "trained":[{"condition":arm["train"]["condition"],
                        "seed":arm["train"]["seed"],
                        "accuracy":arm["heldout"]["accuracy"],
                        "n":arm["heldout"]["n"]}
                       for arm in c["learned_arms"]]
        } for c in receipt["cases"]]},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
