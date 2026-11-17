"""Matched compute, tiny neural recurrent curriculum arms for SIML-01N.

Development metrics can influence task allocation. Final T1 word-compositions
and T4 pair holdouts are accessed only *after* all training updates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import torch

from microlearner import canonical, digest
from neural_student import (NODES, WEIGHTS, RecurrentStudent, encode_batch,
                            frozen_pool, measured_cost, triple_transfer_accuracy)

ARMS=("fixed", "random", "learning_progress", "introspective_naive",
      "introspective", "introspective_shuffled")
TASK_ALLOCATION = ("T0",)*5+("T1",)*9+("T4",)*6


@dataclass(frozen=True)
class Config:
    seeds: tuple[int,...]=(0,1)
    steps: int=160
    batch: int=32
    width: int=48
    diagnostic_interval: int=16
    successor_weight: float=.35
    learning_rate: float=.003
    arms: tuple[str,...]=ARMS

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(type(s)!=int or s<0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if self.steps<1 or self.batch<1 or self.width<8 or self.diagnostic_interval<1:
            raise ValueError("nonpositive budget or insufficient model width")
        if any(a not in ARMS for a in self.arms) or len(set(self.arms)) != len(self.arms):
            raise ValueError("invalid arm list")
        if not (0 <= self.successor_weight <= 10) or not (0 < self.learning_rate < 1):
            raise ValueError("invalid training hyperparameters")


def checksum_model(model:RecurrentStudent):
    hasher=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode()+b"\x00")
        raw=tensor.detach().cpu().contiguous().numpy().tobytes()
        hasher.update(len(raw).to_bytes(8,"little"))
        hasher.update(raw)
    return hasher.hexdigest()


def dev_diagnostics(model:RecurrentStudent,dev_pools):
    # Identical diagnostic work for all arms at all inspection rounds.
    scores={n:model.accuracy(dev_pools[n]) for n in NODES}
    residuals={n:(model.closure_residual(dev_pools[n][:64]) if n!="T0" else 0.0)
               for n in NODES}
    return {"accuracy":scores,"closure_residual":residuals}


def choose_task(arm:str, step:int, rng:random.Random, metrics:dict,
                prev_metrics:dict | None) -> str:
    if arm=="fixed":
        return TASK_ALLOCATION[step%len(TASK_ALLOCATION)]
    if arm=="random":
        return rng.choices(NODES,weights=[WEIGHTS[n] for n in NODES],k=1)[0]
    if arm=="learning_progress":
        # Standard behavior-only curriculum. First round allocates by unmet need.
        scores={n:(1-metrics["accuracy"][n])*.12 +
                (max(0.,metrics["accuracy"][n]-(prev_metrics or metrics)["accuracy"][n])*.25)
                for n in NODES}
        return max(NODES,key=lambda n:(scores[n],WEIGHTS[n]))
    if arm=="introspective_naive":
        # Deliberately unguarded greedy policy, exposing curriculum starvation.
        scores={n:(1-metrics["accuracy"][n])*WEIGHTS[n] +
                .35*min(metrics["closure_residual"][n],1.)+
                (.03 if n=="T0" else 0.) for n in NODES}
        return max(NODES,key=lambda n:(scores[n],WEIGHTS[n]))
    if arm in ("introspective","introspective_shuffled"):
        # Conservatively blend the fixed family distribution with a diagnostic
        # preference, ensuring each family retains selection probability >=.65*w.
        # This is a heuristic policy, not a confidence-qualified improvement.
        closure=metrics["closure_residual"]
        if arm=="introspective_shuffled":
            closure={"T0":closure["T4"],"T1":closure["T0"],"T4":closure["T1"]}
        raw={n:WEIGHTS[n]*(.25+1-metrics["accuracy"][n]+.5*min(closure[n],1.)) for n in NODES}
        total=sum(raw.values())
        weights=[.65*WEIGHTS[n]+.35*raw[n]/total for n in NODES]
        return rng.choices(NODES,weights=weights,k=1)[0]
    raise ValueError("unregistered arm")


def run_arm(config:Config, seed:int, arm:str):
    if arm not in ARMS:
        raise ValueError("unknown curriculum arm")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model=RecurrentStudent(config.width)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=0)
    data={n:frozen_pool(n,"train") for n in NODES}
    dev={n:frozen_pool(n,"dev") for n in NODES}
    # Independent per-task streams use stable hashes across processes.
    streams={n:random.Random(int.from_bytes(hashlib.sha256(f"siml01n.data:{seed}:{n}".encode()).digest()[:8],"big")) for n in NODES}
    selector_rng=random.Random(1_000_000+seed)
    initial_checkpoint=checksum_model(model)
    counts=Counter()
    snapshots=[]
    prev=None
    metrics=dev_diagnostics(model,dev)
    t0=time.perf_counter()
    for i in range(config.steps):
        if i % config.diagnostic_interval==0:
            if i:
                prev=metrics
                metrics=dev_diagnostics(model,dev)
            snapshots.append({"after_updates":i,"dev":metrics})
        task=choose_task(arm,i,selector_rng,metrics,prev)
        population=data[task]
        chosen=[population[streams[task].randrange(len(population))] for _ in range(config.batch)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=model.batch_loss(encode_batch(chosen),successor_weight=config.successor_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("invalid loss")
        loss.backward()
        optimizer.step()
        counts[task]+=1
    # Freeze checkpoint before looking at the test collection.
    final_checkpoint=checksum_model(model)
    final_dev=dev_diagnostics(model,dev)
    frozen_test={n:frozen_pool(n,"test") for n in NODES}
    final_test={n:model.accuracy(frozen_test[n]) for n in NODES}
    secondary_triple=triple_transfer_accuracy(model)
    elapsed=time.perf_counter()-t0
    resources=measured_cost(model,steps=config.steps,batch=config.batch,
                            diagnostics=len(snapshots)+1)
    resources["scheduler_calls"]=config.steps
    resources["wall_seconds_observed"]=round(elapsed,4)
    resources["estimated_training_recurrent_step_calls"] = config.steps * config.batch * 2
    return {"arm":arm,"seed":seed,"initial_checkpoint":initial_checkpoint,
            "trained_checkpoint":final_checkpoint,"updates_by_node":dict(sorted(counts.items())),
            "initial_diagnostics":snapshots[0]["dev"],"final_development":final_dev,
            "final_test":final_test,"secondary_unseen_triples":secondary_triple,
            "weighted_test_score":sum(final_test[n]*WEIGHTS[n] for n in NODES),
            "test_counts":{n:len(frozen_test[n]) for n in NODES},
            "resources":resources,
            "inspection_trace":snapshots}


def run_experiment(config:Config):
    started=time.perf_counter()
    outcomes=[]
    for seed in config.seeds:
        for arm in config.arms:
            outcomes.append(run_arm(config,seed,arm))
    # Means are descriptive across initializations, not confidence-qualified.
    means={arm:sum(r["weighted_test_score"] for r in outcomes if r["arm"]==arm)/len(config.seeds)
           for arm in config.arms}
    sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
             for name in ("neural_student.py","neural_curriculum.py","microlearner.py","gate_math.py","local_proposer.py")}
    return {"schema":"siml01.neural-curriculum.v0.2",
            "source_sha256":sources,
            "evidence":"exploratory_CPU_training_with_new_unseen_length3_holdout",
            "torch":torch.__version__, "python":platform.python_version(),
            "graph":{"nodes":list(NODES),"weights":WEIGHTS,
                     "T1_final_words":[list(x) for x in __import__("neural_student").T1_WORDS],
                     "T4_split":"SHA256(siml01n.t4.v1:a,b) buckets 70/15/15",
                     "length3_secondary_words":[list(w) for w in __import__("neural_student").TRIPLE_WORDS]},
            "configuration":asdict(config),
            "task_pool_counts":{n:{s:len(frozen_pool(n,s)) for s in ("train","dev","test")}
                                for n in NODES},
            "total_time_seconds":round(time.perf_counter()-started,4),
            "comparison_means":means,"outcomes":outcomes,
            "limits":["Small finite symbolic world with a reused task grammar",
                      "Test pairs are disjoint in T4 but held-out T1 words share the same eight states",
                      "The three-action holdout uses known primitives in a new execution length",
                      "The proposer is deterministic Python, not a loaded 27B model",
                      "The scheduler's development outcomes are allowed for curriculum choice",
                      "No statistical superiority certificate; training seeds are independent comparison units",
                      "Only a trained recurrent student and neural diagnostic are demonstrated"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",default="0,1")
    parser.add_argument("--steps",type=int,default=160)
    parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--width",type=int,default=48)
    parser.add_argument("--interval",type=int,default=16)
    parser.add_argument("--arms",default=",".join(ARMS))
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    config=Config(seeds=tuple(int(x) for x in args.seeds.split(",")),
                  steps=args.steps,batch=args.batch,width=args.width,
                  diagnostic_interval=args.interval,arms=tuple(args.arms.split(",")))
    result=run_experiment(config)
    file=Path(args.output)
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_bytes(canonical(result)+b"\n")
    print(json.dumps({"output":str(file),"means":result["comparison_means"],
                      "pool_sizes":result["task_pool_counts"]},indent=2))


if __name__=="__main__":
    main()
