"""CT-02 offline, CPU-first compression/growth trajectory pilot.

Synthetic experiment: no external checkpoints, hosted inference, or GPU side effects.
Requires PyTorch; all model and data RNGs are seeded. Evidence is synthetic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

INPUT_DIM, N_CLASSES = 12, 6
TEACHER_DEPTH, TEACHER_WIDTH = 8, 64
TARGET_DEPTH, TARGET_WIDTH = 4, 32


class ResidualMLP(nn.Module):
    def __init__(self, depth: int, width: int):
        super().__init__()
        self.depth, self.width = depth, width
        self.embed = nn.Linear(INPUT_DIM, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.out = nn.Linear(width, N_CLASSES)
        self.register_buffer("gains", torch.ones(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.embed(x))
        for layer, gain in zip(self.blocks, self.gains):
            h = h + 0.35 * gain * torch.tanh(layer(h))
        return self.out(h)


def make_data(count: int, seed: int, shift: bool = False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(count, INPUT_DIM, generator=g)
    phase = torch.randint(0, 3, (count,), generator=g)
    x[:, :3] = F.one_hot(phase, 3).float() * 2 - 1
    if shift:
        x[:, 3:] *= 1.5
    scores = torch.stack((
        x[:, 3] * x[:, 4] + 0.55 * x[:, 5] - 0.25 * x[:, 6] * x[:, 7],
        x[:, 4] - 0.70 * x[:, 5] * x[:, 6] + 0.4 * torch.sin(x[:, 7]),
        x[:, 8] * x[:, 9] + 0.2 * x[:, 10] - 0.4 * x[:, 3] * x[:, 4],
    ), dim=1)
    y = 2 * phase + (scores.gather(1, phase[:, None]).squeeze(1) > 0).long()
    return x, y, phase


def allowed(phase: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    return (predictions // 2) == phase


def enforce_gate(phase: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Toy analogue of an external deterministic INSPECT/ACT/VERIFY gate."""
    return torch.where(allowed(phase, predictions), predictions, 2 * phase)


def uniform_indices(old: int, new: int) -> List[int]:
    if not 1 <= new <= old:
        raise ValueError("new depth must be in [1, old depth]")
    if new == 1:
        return [old - 1]
    return [round(i * (old - 1) / (new - 1)) for i in range(new)]


def resize(source: ResidualMLP, depth: int, width: int, indexes=None):
    """Structured top-left width crop and selected-layer depth crop."""
    if width > source.width or depth > source.depth:
        raise ValueError("resize can only shrink")
    idx = uniform_indices(source.depth, depth) if indexes is None else list(indexes)
    if len(idx) != depth or len(set(idx)) != depth or any(i < 0 or i >= source.depth for i in idx):
        raise ValueError("layer selection must be distinct valid indices")
    result = ResidualMLP(depth, width)
    with torch.no_grad():
        result.embed.weight.copy_(source.embed.weight[:width, :])
        result.embed.bias.copy_(source.embed.bias[:width])
        for dest, i in zip(result.blocks, idx):
            src = source.blocks[i]
            dest.weight.copy_(src.weight[:width, :width])
            dest.bias.copy_(src.bias[:width])
        result.gains.copy_(source.gains[idx])
        result.out.weight.copy_(source.out.weight[:, :width])
        result.out.bias.copy_(source.out.bias)
    return result


@torch.no_grad()
def rank_layers(teacher: ResidualMLP, x: torch.Tensor, keep: int):
    """Calibration-only output perturbation after zeroing each residual block."""
    baseline = teacher(x)
    scores = []
    for i, layer in enumerate(teacher.blocks):
        saved = layer.weight.detach().clone(), layer.bias.detach().clone()
        layer.weight.zero_(); layer.bias.zero_()
        score = (teacher(x) - baseline).square().mean().item()
        layer.weight.copy_(saved[0]); layer.bias.copy_(saved[1])
        scores.append((score, i))
    return sorted(i for _, i in sorted(scores, reverse=True)[:keep])


def macs(model: ResidualMLP) -> int:
    w = model.width
    return INPUT_DIM * w + model.depth * w * w + w * N_CLASSES


def train_segment(model, x, y, soft, stream, start, stop, kd, lr):
    # Fresh optimizer at every segment for ALL arms, including one-shot and scratch.
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    for i in range(start, stop):
        j = stream[i]
        logits = model(x[j])
        loss = F.cross_entropy(logits, y[j])
        if kd:
            temperature = 2.0
            q = F.softmax(soft[j] / temperature, dim=-1)
            divergence = F.kl_div(F.log_softmax(logits / temperature, dim=-1), q, reduction="batchmean")
            loss = 0.5 * loss + 0.5 * temperature**2 * divergence
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    return model


@torch.no_grad()
def evaluate(model, teacher, data):
    x, y, phase = data
    model.eval()
    pred_logits = model(x)
    pred = pred_logits.argmax(1)
    teacher_logits = teacher(x)
    invalid = (~allowed(phase, pred)).float().mean().item()
    gated = enforce_gate(phase, pred)
    return {
        "accuracy": round((pred == y).float().mean().item(), 6),
        "ce": round(F.cross_entropy(pred_logits, y).item(), 6),
        "teacher_kl": round(F.kl_div(F.log_softmax(pred_logits, -1), F.softmax(teacher_logits, -1), reduction="batchmean").item(), 6),
        "pre_gate_violation": round(invalid, 6),
        "post_gate_violation": int((~allowed(phase, gated)).sum().item()),
    }


def grow(source: ResidualMLP, variant: str, seed: int):
    """4 to 8 untied blocks with repeat, scaled-repeat, matched random, or identity insertions."""
    if source.depth != 4:
        raise ValueError("growth requires depth 4")
    torch.manual_seed(seed)
    result = ResidualMLP(8, source.width)
    mapping = [0, 1, 2, 1, 2, 1, 2, 3]
    with torch.no_grad():
        result.embed.load_state_dict(source.embed.state_dict())
        result.out.load_state_dict(source.out.state_dict())
        for j, i in enumerate(mapping):
            old_anchor = j in (0, 1, 2, 7)
            result.gains[j] = source.gains[i]
            if variant in ("copy", "copy_scaled") or old_anchor:
                result.blocks[j].load_state_dict(source.blocks[i].state_dict())
            if variant == "identity" and not old_anchor:
                result.blocks[j].weight.zero_()
                result.blocks[j].bias.zero_()
            if variant in ("copy_scaled", "random_scaled") and i in (1, 2):
                result.gains[j] /= 3
    return result


@torch.no_grad()
def interpolation_barrier(a: ResidualMLP, b: ResidualMLP, data) -> float:
    """Linear-mode-connectivity diagnostic on the same final architecture.

    Positive values indicate a validation CE barrier above BOTH endpoints.
    Parameter permutations are not aligned; zero is not a basin theorem.
    """
    if (a.depth, a.width) != (b.depth, b.width):
        raise ValueError("endpoint architectures differ")
    states = a.state_dict(), b.state_dict()
    x, y, _ = data
    endpoint_ce = [F.cross_entropy(m(x), y).item() for m in (a, b)]
    model = ResidualMLP(a.depth, a.width)
    losses = []
    for alpha in (0.25, 0.5, 0.75):
        model.load_state_dict({k: torch.lerp(states[0][k], states[1][k], alpha) for k in states[0]})
        losses.append(F.cross_entropy(model(x), y).item())
    return round(max(0.0, max(losses) - max(endpoint_ce)), 7)


def run_seed(seed=0, teacher_steps=80, student_steps=60, batch=128, budget="tokens", growth=False):
    if teacher_steps < 1 or student_steps < 2 or batch < 1:
        raise ValueError("steps and batch must be positive (student steps >= 2)")
    torch.manual_seed(seed)
    train_x, train_y, train_p = make_data(2048, 2000 + seed)
    validation = make_data(1024, 3000 + seed)
    shifted = make_data(1024, 4000 + seed, shift=True)
    teacher = ResidualMLP(TEACHER_DEPTH, TEACHER_WIDTH)
    teacher_stream = torch.randint(0, len(train_x), (teacher_steps, batch), generator=torch.Generator().manual_seed(5000 + seed))
    train_segment(teacher, train_x, train_y, None, teacher_stream, 0, teacher_steps, kd=False, lr=0.008)
    teacher.eval()
    with torch.no_grad():
        train_soft = teacher(train_x)
    saliency_idx = rank_layers(teacher, train_x[:128], keep=TARGET_DEPTH)
    plan = [
        ("one_shot_ce", "one_shot", False),
        ("one_shot_kd", "one_shot", True),
        ("depth_first_kd", "depth_first", True),
        ("width_first_kd", "width_first", True),
        ("joint_kd", "joint", True),
        ("saliency_kd", "saliency", True),
        ("scratch_kd", "scratch", True),
    ]
    result = {
        "schema": "ct02.synth.v1", "seed": seed,
        "torch_version": torch.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task": "six-class phase-conditioned nonlinear oracle; deterministic external phase gate",
        "teacher_steps": teacher_steps, "student_step_budget": student_steps,
        "budget": budget, "batch": batch, "train_count": len(train_x),
        "teacher_accuracy": evaluate(teacher, teacher, validation)["accuracy"],
        "saliency_selected": saliency_idx,
        "arms": {},
    }
    batch_stream = torch.randint(0, len(train_x), (student_steps, batch), generator=torch.Generator().manual_seed(6000 + seed))
    base_for_growth = None
    finished_models = {}
    final_macs = macs(ResidualMLP(TARGET_DEPTH, TARGET_WIDTH))
    for name, path, kd in plan:
        torch.manual_seed(7000 + seed)
        if path == "scratch":
            first = ResidualMLP(TARGET_DEPTH, TARGET_WIDTH)
            mode = "one_shot"
        elif path == "saliency":
            first = resize(teacher, TARGET_DEPTH, TARGET_WIDTH, saliency_idx)
            mode = "one_shot"
        elif path == "depth_first":
            first = resize(teacher, TARGET_DEPTH, TEACHER_WIDTH)
            mode = path
        elif path == "width_first":
            first = resize(teacher, TEACHER_DEPTH, TARGET_WIDTH)
            mode = path
        elif path == "joint":
            first = resize(teacher, 6, 48)
            mode = path
        else:
            first = resize(teacher, TARGET_DEPTH, TARGET_WIDTH)
            mode = "one_shot"
        step1 = max(1, student_steps // 10)
        if budget == "flops":
            # One-stage controls and all paths share an approximate total training-MAC budget.
            available = student_steps * final_macs
            step2 = max(0, (available - step1 * macs(first)) // final_macs)
            if step2 == 0:
                raise ValueError("compute budget too small for this intermediate")
        else:
            step2 = student_steps - step1
        total = step1 + step2
        # For FLOP matching, the first samples and subsequent samples use a common
        # pre-registered stream; extend deterministically if rounding requires it.
        if total > len(batch_stream):
            raise AssertionError("FLOP mode unexpectedly exceeded the step stream")
        first = train_segment(first, train_x, train_y, train_soft, batch_stream, 0, step1, kd, 0.008)
        init_macs = macs(first)
        student = resize(first, TARGET_DEPTH, TARGET_WIDTH) if mode != "one_shot" else first
        student = train_segment(student, train_x, train_y, train_soft, batch_stream, step1, total, kd, 0.008)
        metrics = {
            "intermediate": [first.depth, first.width],
            "final": [student.depth, student.width],
            "parameters": sum(p.numel() for p in student.parameters()),
            "recovery_steps": total,
            "recovery_examples": total * batch,
            "train_mac_proxy": 3 * batch * (step1 * init_macs + step2 * macs(student)),
            "validation": evaluate(student, teacher, validation),
            "shifted": evaluate(student, teacher, shifted),
        }
        result["arms"][name] = metrics
        finished_models[name] = student
        if name == "one_shot_kd":
            base_for_growth = student
    result["interpolation_barriers_vs_one_shot_kd"] = {
        name: interpolation_barrier(finished_models["one_shot_kd"], model, validation)
        for name, model in finished_models.items() if name != "one_shot_kd"
    }
    if growth:
        result["growth"] = {}
        with torch.no_grad():
            source_logits = base_for_growth(validation[0])
        for variant in ("copy", "copy_scaled", "random_scaled", "identity"):
            grown = grow(base_for_growth, variant, 9000 + seed)
            with torch.no_grad():
                drift = (grown(validation[0]) - source_logits).square().mean().item()
            grow_stream = torch.randint(0, len(train_x), (max(2, student_steps // 2), batch), generator=torch.Generator().manual_seed(9999 + seed))
            train_segment(grown, train_x, train_y, train_soft, grow_stream, 0, len(grow_stream), True, 0.008)
            result["growth"][variant] = {
                "initial_function_mse": round(drift, 9),
                "training_steps": len(grow_stream),
                "parameters": sum(p.numel() for p in grown.parameters()),
                "validation": evaluate(grown, teacher, validation),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0", help="comma-separated seeds")
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--student-steps", type=int, default=60)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--budget", choices=("tokens", "flops"), default="tokens")
    parser.add_argument("--growth", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = [run_seed(int(s), args.teacher_steps, args.student_steps, args.batch, args.budget, args.growth) for s in args.seeds.split(",")]
    obj = {"experiment": "CT-02", "results": results}
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    if args.out:
        path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
