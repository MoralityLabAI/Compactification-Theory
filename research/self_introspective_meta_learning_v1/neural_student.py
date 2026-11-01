"""SIML-01N: small *trained* tied-weight recurrent student and frozen task graph.

Torch is optional, deliberately isolated from the standard-library SIML-01
reference. T1 final evaluation words are excluded from every training and
scheduler diagnostic. T4 final state-pairs are disjoint from train/dev pairs.
This is a research pilot, not a theorem about unseen task semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
import random
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from microlearner import ACTIONS, T1_WORDS, apply_bit_action, all_bit_states, digest

N_CLASSES = 31
FEATURES = 72
OPERATIONS = tuple(ACTIONS) + ("shear", "nop")
OP_INDEX = {a: i for i, a in enumerate(OPERATIONS)}
BIT2 = tuple((x, (a, b)) for x in all_bit_states()
             for a in ACTIONS for b in ACTIONS if (a,b) not in T1_WORDS)
# Five development word types, excluded from optimization and final two-word test.
DEV_WORDS = (("reveal_c", "flip_b"), ("flip_b", "flip_y"),
             ("identity", "reveal_b"), ("reveal_b", "flip_y"),
             ("flip_y", "identity"))
TRAIN_WORDS = tuple((a, b) for a in ACTIONS for b in ACTIONS
                    if (a, b) not in T1_WORDS and (a, b) not in DEV_WORDS)
NODES = ("T0", "T1", "T4")
# Secondary, unseen *three-action* semantic compositions, frozen after v0 pilot.
TRIPLE_WORDS = (("flip_b","flip_y","reveal_c"),
                ("reveal_c","flip_b","flip_y"),
                ("flip_y","reveal_b","flip_y"))
WEIGHTS = {"T0": 0.25, "T1": 0.45, "T4": 0.30}


def apply_word(x: tuple[int, int, int], word: Sequence[str]):
    for action in word:
        x = apply_bit_action(x, action)
    return x


def split_shear(a: int, b: int) -> str:
    # A fixed, label-independent split. No model seed or outcome influences it.
    n = int.from_bytes(sha256(f"siml01n.t4.v1:{a},{b}".encode()).digest()[:4], "big") % 100
    return "train" if n < 70 else "dev" if n < 85 else "test"


@dataclass(frozen=True)
class Example:
    task: str
    state: tuple[int, ...]
    actions: tuple[str, str]
    target: int
    successor_bits: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None


def make_case(task: str, state: tuple[int, ...], actions=None) -> Example:
    if task == "T0":
        return Example(task, state, ("nop", "nop"), state[0])
    if task == "T1":
        if actions is None or len(actions) != 2 or any(a not in ACTIONS for a in actions):
            raise ValueError("T1 requires two registered primitive actions")
        m = apply_bit_action(state, actions[0])
        n = apply_bit_action(m, actions[1])
        return Example(task, state, tuple(actions), n[0], (m, n))
    if task == "T4":
        a, b = state
        return Example(task, state, ("shear", "nop"), (2*a+b) % 31)
    raise ValueError("unregistered task")


def frozen_pool(task: str, split: str) -> list[Example]:
    if split not in ("train", "dev", "test"):
        raise ValueError("bad split")
    if task == "T0":
        return [make_case(task,x) for x in all_bit_states()]
    if task == "T1":
        words = TRAIN_WORDS if split == "train" else DEV_WORDS if split == "dev" else T1_WORDS
        return [make_case(task,x,w) for w in words for x in all_bit_states()]
    if task == "T4":
        return [make_case(task,(a,b)) for a in range(31) for b in range(31)
                if split_shear(a,b) == split]
    raise ValueError("bad task")


def encode_batch(cases: Sequence[Example], device="cpu"):
    """Public inputs: raw bit state or one-hot modular coordinates + task tag."""
    inputs = torch.zeros((len(cases), FEATURES), dtype=torch.float32, device=device)
    operations = torch.empty((len(cases), 2), dtype=torch.long, device=device)
    targets = torch.empty((len(cases),), dtype=torch.long, device=device)
    states = torch.zeros((len(cases),2,3), dtype=torch.float32, device=device)
    is_bit = torch.zeros(len(cases), dtype=torch.bool, device=device)
    for i,c in enumerate(cases):
        if c.task in ("T0", "T1"):
            inputs[i, :3] = torch.tensor(c.state, device=device, dtype=torch.float32)
            inputs[i, 65] = 1.
            if c.successor_bits:
                states[i] = torch.tensor(c.successor_bits, device=device, dtype=torch.float32)
                is_bit[i] = True
        else:
            a,b=c.state
            # Engineered joint (present sum, hidden a) phase features from the
            # exact shear construction; an explicit, charged inductive bias.
            s=(a+b)%31
            inputs[i,3+s] = 1.
            inputs[i,34+a] = 1.
            inputs[i,66] = 1.
            inputs[i,67:71] = torch.tensor([math.cos(2*math.pi*s/31),math.sin(2*math.pi*s/31),
                                           math.cos(2*math.pi*a/31),math.sin(2*math.pi*a/31)],device=device)
        # index 71 reserved for a future interface mode
        operations[i] = torch.tensor([OP_INDEX[a] for a in c.actions],device=device)
        targets[i] = c.target
    return inputs,operations,targets,states,is_bit


class RecurrentStudent(nn.Module):
    """Unshared input encoder, tied recurrent transition and fixed shared head."""
    def __init__(self, width: int = 48):
        super().__init__()
        if width < 8:
            raise ValueError("width must be >= 8")
        self.width=width
        self.encoder=nn.Sequential(nn.Linear(FEATURES,width),nn.Tanh())
        self.operations=nn.Embedding(len(OPERATIONS),width)
        self.step=nn.GRUCell(width,width)
        self.readout=nn.Linear(width,N_CLASSES)
        self.probe=nn.Linear(width,3)

    def forward(self, inputs, op_ids):
        h=self.encoder(inputs)
        states=[]
        if op_ids.ndim != 2 or op_ids.shape[1] < 1:
            raise ValueError("action sequence must have positive length")
        for t in range(op_ids.shape[1]):
            h=self.step(self.operations(op_ids[:,t]),h)
            states.append(h)
        return self.readout(h),torch.stack(states,dim=1)

    def batch_loss(self, batch, *, successor_weight: float=0.0):
        inputs,op_ids,target,successors,mask=batch
        logits,states=self(inputs,op_ids)
        bit=inputs[:,65]>0
        # Task-specific support: binary for T0/T1, 31-way for T4.
        binary_loss=F.cross_entropy(logits[bit,:2],target[bit]) if bit.any() else logits.sum()*0
        modular_loss=F.cross_entropy(logits[~bit],target[~bit]) if (~bit).any() else logits.sum()*0
        n_bit=int(bit.sum().item())
        n_mod=int((~bit).sum().item())
        loss=(binary_loss*n_bit+modular_loss*n_mod)/max(1,n_bit+n_mod)
        if successor_weight and mask.any():
            # Privileged exact microstate is supplied only for sampled training cases.
            aux=F.binary_cross_entropy_with_logits(self.probe(states[mask]),successors[mask])
            loss=loss+successor_weight*aux
        return loss

    @torch.no_grad()
    def classify(self,cases: Sequence[Example],chunk=256):
        self.eval()
        answers=[]
        for offset in range(0,len(cases),chunk):
            batch=encode_batch(cases[offset:offset+chunk])
            logits,_=self(batch[0],batch[1])
            bit=batch[0][:,65]>0
            if bit.any(): logits[bit,2:]=-float("inf")
            answers.extend(logits.argmax(dim=-1).tolist())
        return answers

    @torch.no_grad()
    def accuracy(self,cases:Sequence[Example]):
        predicted=self.classify(cases)
        return sum(y==c.target for y,c in zip(predicted,cases))/len(cases)

    @torch.no_grad()
    def closure_residual(self, cases: Sequence[Example], chunk=128):
        """Mean squared discrepancy with a freshly encoded known successor.

        These are *development* states only; no sealed test words are read.
        Algebraic closure is approximate and scale-dependent for neural states.
        """
        self.eval()
        errors=[]
        for offset in range(0,len(cases),chunk):
            batch_cases=cases[offset:offset+chunk]
            batch=encode_batch(batch_cases)
            inp,ops=batch[0],batch[1]
            init=self.encoder(inp)
            nxt=self.step(self.operations(ops[:,0]),init)
            succ=[]
            for c in batch_cases:
                if c.task=="T1":
                    succ.append(make_case("T0",apply_bit_action(c.state,c.actions[0])))
                elif c.task=="T4":
                    a,b=c.state
                    succ.append(make_case("T4",(a,(a+b)%31)))
                else:
                    succ.append(make_case("T0",c.state))
            encoded=self.encoder(encode_batch(succ)[0])
            errors.extend(((encoded-nxt)**2).mean(dim=1).tolist())
        return math.fsum(errors)/len(errors)


@torch.no_grad()
def triple_transfer_accuracy(model:RecurrentStudent):
    """Unseen 3-action words, assessed only after training and candidate freeze."""
    cases=[]
    targets=[]
    for word in TRIPLE_WORDS:
        for x in all_bit_states():
            cases.append(make_case("T1",x,word[:2]))
            targets.append(apply_word(x,word)[0])
    inputs,_,_,_,_=encode_batch(cases)
    ops=torch.tensor([[OP_INDEX[x] for x in word] for word in TRIPLE_WORDS for _ in all_bit_states()], dtype=torch.long)
    model.eval()
    logits,_=model(inputs,ops)
    predicted=logits[:,:2].argmax(dim=-1).tolist()
    return sum(a==b for a,b in zip(predicted,targets))/len(targets)


def measured_cost(model:RecurrentStudent, *,steps:int,batch:int,diagnostics:int):
    return {"parameters":sum(p.numel() for p in model.parameters()),
            "training_updates":steps,"examples_drawn":steps*batch,
            "recurrent_steps_per_example":2,
            "recurrent_step_examples":steps*batch*2,
            "diagnostic_rounds":diagnostics,
            "device":"cpu", "dtype":"torch.float32"}
