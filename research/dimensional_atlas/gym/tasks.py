"""Exact source/verifier adapters for SCM-15.

The self-contained finite-ring tasks are SYNTHETIC plumbing tests. Real
formal-series assessment must import the pinned independent EvalMath oracle.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import importlib.util
import json
import math
import random
from typing import Any

OPS = ("I", "N", "S", "R")
PROGRAMS = tuple(product(OPS, repeat=3))
TOPOLOGIES = ("fork", "cascade", "additive", "hadamard")
MOD = 17
DEGREE = 12
EXPECTED_EVALMATH_SHA256 = "c0df77faae6351e522f35d36fa6031d6270ecced2fd92d0477cb5e08992ccb0b"


def _canonical(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class MathTask:
    case_id: str
    family: str
    topology: str
    demos: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    query: tuple[int, ...]
    hidden_target: tuple[int, ...]
    source_graph: tuple[str, str, str]
    backend: str = "synthetic_ring"
    oracle_case: Any = None

    @property
    def public(self) -> dict:
        return {"case_id": self.case_id, "topology": self.topology,
                "demos": [{"input": list(x), "output": list(y)} for x, y in self.demos],
                "query": list(self.query), "backend": self.backend}

    @property
    def signature(self) -> str:
        return sha256(_canonical({k:v for k,v in self.public.items() if k != "case_id"}).encode()).hexdigest()


def _toy_unary(x: tuple[int, ...], op: str) -> tuple[int, ...]:
    if op == "I": return x
    if op == "N": return tuple(-a % MOD for a in x)
    if op == "S": return (x[-1],) + x[:-1]
    if op == "R": return x[::-1]
    raise ValueError(op)


def toy_execute(x: tuple[int, ...], topo: str, graph: tuple[str,...]) -> tuple[int,...]:
    if topo not in TOPOLOGIES or tuple(graph) not in PROGRAMS:
        raise ValueError("invalid synthetic program")
    a = _toy_unary(x, graph[0]); b = _toy_unary(a if topo == "cascade" else x, graph[1])
    if topo == "fork":
        out = tuple(sum(a[k]*b[(i-k)%len(a)] for k in range(len(a))) % MOD for i in range(len(a)))
    elif topo == "cascade":
        out = tuple((a[i] + 3*b[i]) % MOD for i in range(len(a)))
    elif topo == "additive":
        out = tuple((a[i] + b[i]) % MOD for i in range(len(a)))
    else:
        out = tuple(a[i]*b[i] % MOD for i in range(len(a)))
    return _toy_unary(out, graph[2])


def make_toy(seed: int, *, case_id: str, family: str = "iid",
             topology: str | None = None, graph: tuple[str,...] | None = None) -> MathTask:
    rng = random.Random(seed)
    topo = topology or rng.choice(TOPOLOGIES)
    program = tuple(graph or rng.choice(PROGRAMS))
    def series():
        if family == "iid":
            return tuple(rng.randrange(MOD) for _ in range(DEGREE+1))
        if family == "sparse":
            return tuple(rng.randrange(MOD) if rng.randrange(5) == 0 else 0 for _ in range(DEGREE+1))
        if family == "periodic":
            pattern = [rng.randrange(MOD) for _ in range(3)]
            return tuple(pattern[i%3] for i in range(DEGREE+1))
        raise ValueError("unknown family")
    inputs = [series() for _ in range(3)]
    demos = tuple((x, toy_execute(x,topo,program)) for x in inputs[:2])
    return MathTask(case_id, family, topo, demos, inputs[2],
                    toy_execute(inputs[2],topo,program), program)


def load_evalmath(source_file: Path):
    """Fail closed on source/version drift; never replace the independent oracle."""
    if not source_file.is_file():
        raise FileNotFoundError("EvalMath source checkout required")
    digest = sha256(source_file.read_bytes()).hexdigest()
    if digest != EXPECTED_EVALMATH_SHA256:
        raise ValueError(f"EvalMath exact oracle source hash mismatch: {digest}")
    spec = importlib.util.spec_from_file_location("_scm15_pinned_evalmath_oracle", source_file)
    if spec is None or spec.loader is None: raise RuntimeError("unloadable EvalMath oracle")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves its defining module from sys.modules.
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if tuple(mod.PROGRAMS) != PROGRAMS or tuple(mod.TOPOLOGIES) != TOPOLOGIES:
        raise ValueError("EvalMath DSL schema mismatch")
    return mod


def make_evalmath_case(oracle, seed: int, *, case_id: str, family: str,
                       topology: str | None = None, graph: tuple[str,...] | None = None) -> MathTask:
    rng = random.Random(seed)
    topo = topology or rng.choice(TOPOLOGIES)
    program = tuple(graph or rng.choice(PROGRAMS))
    case = oracle.make_case(rng, split="scm15_discovery", family=family, topology=topo,
                            graph=program, identifier=case_id)
    return MathTask(case_id, family, topo,
                    tuple((tuple(a[:13]),tuple(b[:13])) for a,b in case.demos),
                    tuple(case.query[:13]), tuple(case.future), program,
                    backend="evalmath_arg01", oracle_case=case)


def check_candidate(task:MathTask, graph:tuple[str,...], oracle=None) -> dict:
    graph=tuple(graph)
    if graph not in PROGRAMS: raise ValueError("bad graph")
    if task.backend == "evalmath_arg01":
        if oracle is None: raise RuntimeError("independent EvalMath oracle required")
        return oracle.check_candidate(task.oracle_case, graph)
    errors=[]
    for x,y in task.demos:
        output=toy_execute(x,task.topology,graph)
        errors.append(next((i for i,(v,w) in enumerate(zip(output,y)) if v!=w),None))
    return {"accepted":all(x is None for x in errors),"first_faults":errors}


def verify_order(task:MathTask, order:list[int], *, budget:int=12, oracle=None) -> dict:
    """Only demo-prefix verification occurs before a single future grade."""
    checked=[]; winner=None; faults=[]
    for index in dict.fromkeys(order):
        if len(checked)>=budget: break
        graph=PROGRAMS[int(index)]
        q=check_candidate(task,graph,oracle)
        checked.append(int(index)); faults.append(q["first_faults"])
        if q["accepted"]:
            winner=int(index); break
    # Holdout future grading happens after selection; it is never model input.
    if winner is None: future=False
    elif task.backend=="evalmath_arg01":
        future=bool(oracle.grade(task.oracle_case,{"graph":PROGRAMS[winner]})["future_correct"])
    else:
        future=toy_execute(task.query,task.topology,PROGRAMS[winner])==task.hidden_target
    return {"winner":winner,"queries":len(checked),"checked":checked,
            "faults":faults,"accepted":winner is not None,
            "future_correct":bool(future),
            "false_future_acceptance":winner is not None and not future}


def encode(task:MathTask):
    """4 x 30 public feature tensor; hidden program/future never accessed."""
    import torch
    def compress(x):
        if task.backend=="synthetic_ring":
            return [(v-8)/8 for v in x]
        return [math.copysign(min(1.,math.log1p(abs(v))/9),v) for v in x]
    rows=[]
    for pos in range(4):
        onehot=[float(i==TOPOLOGIES.index(task.topology)) for i in range(4)]
        if pos<2:
            x,y=task.demos[pos]; vector=compress(x)+compress(y)
        elif pos==2:
            vector=compress(task.query)+[0.]*13
        else:
            vector=[0.]*26
        rows.append(onehot+vector)
    return torch.tensor(rows,dtype=torch.float32)


def fresh_pool(kind:str, seeds:list[int], *, family:str="iid", oracle=None,
               graph:tuple[str,...]|None=None, topology:str|None=None,
               forbidden:set[str]|None=None) -> list[MathTask]:
    seen=set(forbidden or ()); out=[]
    for seed in seeds:
        task=(make_evalmath_case(oracle,seed,case_id=f"{kind}:{seed}",family=family,
                                 topology=topology,graph=graph) if oracle is not None else
              make_toy(seed,case_id=f"{kind}:{seed}",family=family,
                       topology=topology,graph=graph))
        if task.signature in seen: continue
        seen.add(task.signature); out.append(task)
    return out


def select_frontier(failures:list[dict], *, limit:int=4) -> list[dict]:
    """Maxima of a declared discovery proxy over a finite cell-neighborhood.

    Input rows contain only *discovery* candidate confidence and first-prefix
    fault. No future gold, heldout labels or open test scores can enter.
    """
    scored=[]
    for row in failures:
        if row.get("split")!="discovery": raise ValueError("frontier uses discovery only")
        fault=row.get("first_fault")
        if fault is None: continue
        score=float(row["confidence"])*(1+int(fault))/(DEGREE+1)
        scored.append({**row,"frontier_proxy":score})
    # Family/graph/topology cells define a finite neighborhood. Pick its local
    # maximum, with deterministic case-ID tie breaks. These are *proxy peaks*.
    peaks={}
    for row in scored:
        key=(row["family"],row["topology"],tuple(row["graph"]))
        if key not in peaks or (-row["frontier_proxy"],row["case_id"]) < (-peaks[key]["frontier_proxy"],peaks[key]["case_id"]):
            peaks[key]=row
    return sorted(peaks.values(),key=lambda x:(-x["frontier_proxy"],x["case_id"]))[:limit]


def seal_new_evals(peaks:list[dict], *, seed_base:int=900000, per_peak:int=2,
                   oracle=None, forbidden:set[str]|None=None) -> tuple[list[dict],list[dict]]:
    """Fresh generator seeds; model sees only public, private sidecar stays sealed."""
    public=[]; private=[]; used=set(forbidden or ())
    for pi,peak in enumerate(peaks):
        for j in range(per_peak):
            seed=seed_base+pi*1000+j
            family=peak["family"]; topo=peak["topology"]; graph=tuple(peak["graph"])
            task=(make_evalmath_case(oracle,seed,case_id=f"scm15:{pi}:{j}",family=family,
                                     topology=topo,graph=graph) if oracle is not None else
                  make_toy(seed,case_id=f"scm15:{pi}:{j}",family=family,topology=topo,graph=graph))
            if task.signature in used: raise ValueError("generated task duplicates existing public signature")
            used.add(task.signature)
            public.append({**task.public,"public_sha256":task.signature,"generation_source":"discovery_proxy_cell"})
            private.append({"case_id":task.case_id,"public_sha256":task.signature,
                            "source_graph":list(task.source_graph),"future":list(task.hidden_target),
                            "source_seed":seed})
    return public,private
