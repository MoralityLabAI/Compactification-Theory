"""Small CPU training pilot for a learned recurrent pointer-machine readout.

Toy PyTorch model, with explicit multiplicative features state_i * edge_ij.
It does NOT use LOTUS, Qwen, Qwenputer, Circom compiler, or a gradient from R1CS.
Uses one-step supervised labels; evaluates longer shared-weight recurrence.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .spec import build, case, digest, witness_for, witness_for_trace


def train_and_evaluate(seed: int = 20261103, n: int = 6,
                       updates: int = 250, batch: int = 128, cases_per_family: int = 32):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    if n < 2 or n > 12 or updates < 1 or batch < 1 or cases_per_family < 1:
        raise ValueError("invalid training configuration")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = "cpu"

    class Readout(nn.Module):
        def __init__(self, n):
            super().__init__()
            # The feature plane is the same quadratic term as the circuit's prod[i,j].
            self.weight = nn.Parameter(torch.randn(n * n, n) * 0.01)
            self.bias = nn.Parameter(torch.zeros(n))

        def forward(self, state, edges):
            features = (state.unsqueeze(-1) * edges).flatten(1)
            return features @ self.weight + self.bias

    model = Readout(n).to(device)
    # Frozen randomly initialized readout; same bilinear inputs and parameter count.
    random_model = Readout(n).to(device)
    random_model.load_state_dict({k:v.detach().clone() for k,v in model.state_dict().items()})
    random_model.eval()
    optim = torch.optim.AdamW(model.parameters(), lr=0.07, weight_decay=0.0)
    t0 = time.perf_counter()
    for step in range(updates):
        # Training ONLY from random_function family, and one transition at a time.
        maps = torch.tensor([[rng.randrange(n) for _ in range(n)] for _ in range(batch)],
                            dtype=torch.long)
        state_ids = torch.tensor([rng.randrange(n) for _ in range(batch)], dtype=torch.long)
        edges = F.one_hot(maps, num_classes=n).float()
        state = F.one_hot(state_ids, num_classes=n).float()
        target = maps.gather(1, state_ids[:, None]).flatten()
        logits = model(state, edges)
        loss = F.cross_entropy(logits, target)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    train_seconds = time.perf_counter() - t0

    results = []
    circuit_cache = {}
    model.eval()
    with torch.inference_mode():
        for family, split in (("permutation", "calibration"),
                              ("long_cycle", "heldout"), ("sink", "heldout")):
            for depth in (1, 2, 4, 8, 12, 16):
                correct_soft = correct_hard = exact_valid = random_correct = 0
                total_defect = 0.0
                violations_soft = 0
                for idx in range(cases_per_family):
                    task = case(family, seed + 1000, idx, n=n, steps=depth)
                    edges = F.one_hot(torch.tensor(task.successors, dtype=torch.long), n).float()[None]
                    state_soft = F.one_hot(torch.tensor([task.initial]), n).float()
                    state_hard = state_soft.clone()
                    state_random = state_soft.clone()
                    trajectory = [task.initial]
                    for _ in range(depth):
                        previous_soft = int(state_soft.argmax(-1).item())
                        logits_soft = model(state_soft, edges)
                        logits_hard = model(state_hard, edges)
                        state_soft = F.softmax(logits_soft, -1)
                        next_idx = logits_hard.argmax(-1)
                        state_hard = F.one_hot(next_idx, n).float()
                        random_idx = random_model(state_random,edges).argmax(-1)
                        state_random = F.one_hot(random_idx, n).float()
                        trajectory.append(int(next_idx.item()))
                        # Diagnostic uses reference; never fed to neural model.
                        target_step = int(task.successors[previous_soft])
                        total_defect += float((state_soft - F.one_hot(
                            torch.tensor([target_step]), n).float()).square().sum().item())
                    gold = task.trace()[-1]
                    correct_soft += int(int(state_soft.argmax(-1).item()) == gold)
                    correct_hard += int(trajectory[-1] == gold)
                    random_correct += int(int(state_random.argmax(-1).item()) == gold)
                    ck = (task.n, task.steps)
                    if ck not in circuit_cache:
                        circuit_cache[ck] = build(task)
                    checked_circuit = circuit_cache[ck]
                    candidate_witness, candidate_public = witness_for_trace(task, checked_circuit, trajectory)
                    is_verified = checked_circuit.check(candidate_witness, candidate_public)["valid"]
                    if is_verified != (trajectory == task.trace()):
                        raise AssertionError("constraint verifier disagrees with trace oracle")
                    exact_valid += int(is_verified)
                    # Exact circuit verifier should reject an erroneous *claimed* trace.
                    # Keeping the circuit in the audit path, rather than runtime, is intentional.
                    if trajectory != task.trace():
                        violations_soft += 1
                results.append({"split": split, "family": family, "depth": depth,
                                "n": cases_per_family,
                                "soft_accuracy": correct_soft / cases_per_family,
                                "hard_accuracy": correct_hard / cases_per_family,
                                "random_untrained_accuracy":random_correct / cases_per_family,
                                "whole_trace_valid_rate": exact_valid / cases_per_family,
                                "invalid_claimed_traces": violations_soft,
                                "mean_relaxed_defect_sum": total_defect / cases_per_family})
    # Verify one full witness with the actual R1CS checker on a test-family example.
    audit_task = case("long_cycle", seed + 1000, 0, n=n, steps=8)
    circuit = build(audit_task)
    witness, public = witness_for(audit_task, circuit)
    checked = circuit.check(witness, public)
    if not checked["valid"]:
        raise RuntimeError(f"independent R1CS oracle fails: {checked}")
    return {
        "class": "toy_cpu_torch_supervised_proxy",
        "seed": seed, "n_nodes": n, "updates": updates,
        "batch": batch, "n_train_examples": updates * batch,
        "train_family": "random_function", "train_depth": 1,
        "calibration_family": "permutation", "heldout_families": ["long_cycle", "sink"],
        "train_seconds": round(train_seconds, 3),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_weight_sha256": digest({k: v.detach().cpu().tolist() for k,v in model.state_dict().items()}),
        "notes": "Heldout families differ from train, yet N=constant and quadratic product features are explicitly provided; this tests trained readout and iteration, not autonomous algorithm discovery.",
        "r1cs_oracle_example": {"n_signals": len(circuit.names),
                                "n_constraints": len(circuit.constraints),
                                "verification": checked},
        "results": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--updates", type=int, default=250)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--cases", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("runs/algernon/circuit_mirror/pilot.json"))
    a = p.parse_args()
    record = train_and_evaluate(a.seed, a.n, a.updates, a.batch, a.cases)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(f"TRAINED toy CPU readout; {record['parameter_count']} parameters; "
          f"{len(record['results'])} evaluation cells -> {a.out}")
    for r in record["results"]:
        if r["depth"] in (1, 8, 16):
            print(f"{r['family']} depth={r['depth']}: soft={r['soft_accuracy']:.3f}, "
                  f"hard={r['hard_accuracy']:.3f}, valid_trace={r['whole_trace_valid_rate']:.3f}")


if __name__ == "__main__":
    main()
