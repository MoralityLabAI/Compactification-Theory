"""Adapter capacity in bits on Qwen3-0.6B: the counting and Lipschitz bounds applied to trained adapters.

Everything runs in RAM on the CPU. The frozen model runs once to cache, for every item, the residual
stream after the last layer's attention at the answer position. The model is then freed. Each adapter
trains on that cache through the last layer's MLP, the final norm and the two answer rows of the
unembedding, which is exactly the original model's computation at that position.

Task: M random keys ("482910 ->"), each with a random label A or B. The labels are
M independent fair bits, the language-model counterpart of the random-sign bump packing in the
research note (note/bits-depth-lipschitz.html). Measured per run:

- stored information I = M (1 - H2(err)); Fano gives I(labels; adapter) >= I up to sampling noise,
  so a b-bit adapter must satisfy I_b <= W b + 32 (number of tensors). This bound is checked;
- information per adapter parameter, I / W, against the counting ceiling b;
- the quantization knee b*: fewest bits per weight that keep the error within 1 point;
- two predictions of the knee from the adapter's gradients. The worst case uses the l-infinity
  Lipschitz constant (sum of |dz/dtheta| times the rounding step, as in the note). The typical case
  treats rounding errors as independent noise;
- the top Hessian eigenvalue of the training loss (power iteration), for the optimizer-cap question.

Adapters: "probe" (a linear readout on k random directions of the final hidden state, Cover's
classical capacity is 2k points) and "lora" (low-rank factors on chosen last-layer MLP matrices).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

import config
from qwen_suffix import load_model, run_layers

ANSWERS = (" A", " B")
TEMPLATE = "{key} ->"
BITS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
# Default sweep: (adapter, size, items). LoRA size is the rank per listed matrix.
PLAN = (
    [("probe", k, m) for k in (64, 128, 256, 512) for m in (k, 2 * k, 4 * k)]
    + [("lora:gate", r, m) for r in (1, 2, 4, 8) for m in (256, 512, 1024, 2048, 4096)]
    + [("lora:down", r, m) for r in (1, 4) for m in (512, 2048)]
)


def free_commit_gb() -> float | None:
    """Commit headroom (bytes the system can still commit) in GB, on Windows; None elsewhere."""
    if os.name != "nt":
        return None
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong), ("total_commit", ctypes.c_ulonglong),
                    ("avail_commit", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong), ("avail_extended", ctypes.c_ulonglong)]

    status = Status()
    status.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.avail_commit / 2 ** 30


def wait_for_commit(min_gb: float, poll_seconds: int = 300) -> None:
    """Block until the PC has `min_gb` of commit headroom. Other sessions' jobs share this machine, and
    it has rebooted uncleanly when headroom ran out, so a model-loading job waits instead of adding to it."""
    while min_gb > 0 and (free := free_commit_gb()) is not None and free < min_gb:
        print(json.dumps({"event": "waiting_for_memory", "free_commit_gb": round(free, 1), "need_gb": min_gb}), flush=True)
        time.sleep(poll_seconds)


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def h2(p: float) -> float:
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def info_bits(items: int, err: float) -> float:
    """Fano lower bound on the bits the adapter holds about the labels, from its error rate."""
    return items * (1 - h2(err)) if err < 0.5 else 0.0


def make_items(tok, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    keys = (torch.randperm(900_000, generator=gen)[:count] + 100_000).tolist()
    labels = torch.randint(0, 2, (count,), generator=gen).float() * 2 - 1  # +1 = A, -1 = B
    ids = tok([TEMPLATE.format(key=key) for key in keys], add_special_tokens=False)["input_ids"]
    lengths = {len(row) for row in ids}
    if len(lengths) != 1:
        raise SystemExit(f"prompts tokenize to different lengths {sorted(lengths)}; change TEMPLATE")
    return torch.tensor(ids), labels


@torch.no_grad()
def cache_residual(model, ids: torch.Tensor, batch: int) -> torch.Tensor:
    """[M, d] residual after the last layer's attention, at the final position."""
    last = model.model.layers[-1]
    out = []
    start = time.time()
    for index, chunk in enumerate(ids.split(batch)):
        hidden = run_layers(model, model.model.embed_tokens(chunk), 0, len(model.model.layers) - 1)
        length = hidden.shape[1]
        positions = torch.arange(length).unsqueeze(0).expand(hidden.shape[0], -1)
        rotary = model.model.rotary_emb(hidden, positions)
        causal = torch.full((length, length), float("-inf")).triu(1)[None, None]
        attn, _ = last.self_attn(hidden_states=last.input_layernorm(hidden), position_embeddings=rotary,
                                 attention_mask=causal)
        out.append((hidden + attn)[:, -1].clone())
        if index % 20 == 0:
            print(json.dumps({"event": "cache", "done": (index + 1) * batch, "of": len(ids),
                              "seconds": round(time.time() - start, 1)}), flush=True)
    return torch.cat(out)


class Frozen:
    """The pieces of the model the adapters still need, detached from the rest."""

    def __init__(self, model, tok):
        last = model.model.layers[-1]
        self.eps = model.config.rms_norm_eps
        self.post_norm = last.post_attention_layernorm.weight.detach().clone()
        self.final_norm = model.model.norm.weight.detach().clone()
        self.weights = {name: getattr(last.mlp, f"{name}_proj").weight.detach().clone() for name in ("gate", "up", "down")}
        rows = [tok(answer, add_special_tokens=False)["input_ids"] for answer in ANSWERS]
        if any(len(row) != 1 for row in rows):
            raise SystemExit(f"answers {ANSWERS} are not single tokens: {rows}")
        unembed = model.lm_head.weight.detach()
        self.readout = (unembed[rows[0][0]] - unembed[rows[1][0]]).clone()  # logit(A) - logit(B)

    def final_hidden(self, h: torch.Tensor) -> torch.Tensor:
        x = rms(h, self.post_norm, self.eps)
        mid = F.silu(x @ self.weights["gate"].T) * (x @ self.weights["up"].T)
        return rms(h + mid @ self.weights["down"].T, self.final_norm, self.eps)


class LoraHead(nn.Module):
    """Last-layer MLP at the answer position with low-rank factors U V added to the listed matrices."""

    def __init__(self, frozen: Frozen, matrices: list[str], rank: int, seed: int):
        super().__init__()
        self.frozen, self.matrices = frozen, matrices
        gen = torch.Generator().manual_seed(seed)
        self.factors = nn.ParameterDict()
        for name in matrices:
            d_out, d_in = frozen.weights[name].shape
            self.factors[f"{name}_U"] = nn.Parameter(torch.zeros(d_out, rank))
            self.factors[f"{name}_V"] = nn.Parameter(torch.randn(rank, d_in, generator=gen) / math.sqrt(d_in))

    def project(self, name: str, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.frozen.weights[name].T
        if name not in self.matrices:
            return base
        return base + (x @ self.factors[f"{name}_V"].T) @ self.factors[f"{name}_U"].T

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        fz = self.frozen
        x = rms(h, fz.post_norm, fz.eps)
        if "gate" in self.matrices or "up" in self.matrices:
            mid = F.silu(self.project("gate", x)) * self.project("up", x)
        else:
            with torch.no_grad():
                mid = F.silu(x @ fz.weights["gate"].T) * (x @ fz.weights["up"].T)
        out = h + self.project("down", mid)
        return rms(out, fz.final_norm, fz.eps) @ fz.readout


class Probe(nn.Module):
    """margin = base margin + w . P (f - mean), with f the frozen final hidden state.

    P maps onto the top-k principal directions of f over the item pool, scaled to unit variance. The
    item-to-item variation of f is tiny next to its mean (three massive coordinates set its norm) and
    has an effective rank near 5, so raw random directions leave plain Adam far from Cover's capacity of
    2k points. Whitening changes conditioning only, not which labelings are linearly separable.
    """

    def __init__(self, frozen: Frozen, k: int, pool_h: torch.Tensor):
        super().__init__()
        self.frozen = frozen
        with torch.no_grad():
            f = torch.cat([frozen.final_hidden(chunk) for chunk in pool_h.split(4096)])
            mean = f.mean(0)
            _, s, vh = torch.linalg.svd(f - mean, full_matrices=False)
            scale = math.sqrt(len(f)) / s[:k].clamp_min(1e-12)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("proj", (vh[:k] * scale[:, None]).contiguous(), persistent=False)
        self.w = nn.Parameter(torch.zeros(k))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.frozen.final_hidden(h)
            base = f @ self.frozen.readout
            feats = (f - self.mean) @ self.proj.T
        return base + feats @ self.w


def margins(module: nn.Module, h: torch.Tensor, sigma: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([module(chunk) for chunk in h.split(batch)]) * sigma


def quantize_(module: nn.Module, bits: int) -> None:
    """Round every trainable tensor in place onto 2**bits evenly spaced levels over [-max|t|, max|t|]."""
    with torch.no_grad():
        for p in module.parameters():
            if not p.requires_grad:
                continue
            scale = p.abs().max()
            if scale > 0:
                step = 2 * scale / (2 ** bits - 1)
                p.copy_(torch.round((p + scale) / step) * step - scale)


def train(module: nn.Module, h: torch.Tensor, sigma: torch.Tensor, steps: int, lr: float, batch: int,
          seed: int, log, eval_every: int = 100, optimizer: str = "adam", sharpness_every: int = 0,
          sharpness_items: int = 1024) -> dict:
    """Minimise the logistic loss of the signed margin.

    optimizer "adam": Adam, cosine-decayed over `steps` minibatch steps.
    optimizer "gd": plain gradient descent (no momentum) at the constant step `lr`, full batch when
    `batch` >= items. This is the setting of the edge-of-stability prediction (sharpness <= 2/lr).
    With sharpness_every = k > 0, every k-th evaluation also logs the top Hessian eigenvalue of the loss
    on `sharpness_items` items.
    """
    params = [p for p in module.parameters() if p.requires_grad]
    if optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)))
    elif optimizer == "gd":
        opt, sched = torch.optim.SGD(params, lr=lr, momentum=0.0), None
    else:
        raise ValueError(f"unknown optimizer {optimizer}")
    gen = torch.Generator().manual_seed(seed)
    batch = min(batch, len(h))
    start = time.time()
    order: list[torch.Tensor] = []
    step = 0
    evaluations = 0
    peak_sharpness = 0.0
    for step in range(1, steps + 1):
        if not order:
            order = list(torch.randperm(len(h), generator=gen).split(batch))
        index = order.pop()
        loss = F.softplus(-module(h[index]) * sigma[index]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if step % eval_every == 0 or step == steps:
            evaluations += 1
            z = margins(module, h, sigma)
            err = (z <= 0).float().mean().item()
            record = {"event": "step", "step": step, "loss": round(F.softplus(-z).mean().item(), 5),
                      "err": round(err, 5), "seconds": round(time.time() - start, 1)}
            if not math.isfinite(record["loss"]):
                log({**record, "diverged": True})
                break
            if sharpness_every and evaluations % sharpness_every == 0:
                sharp = top_hessian_eigenvalue(module, h[:sharpness_items], sigma[:sharpness_items], 10, seed)
                peak_sharpness = max(peak_sharpness, sharp)
                record["hessian_top"] = round(sharp, 5)
                record["lr_times_hessian"] = round(lr * sharp, 4)
            log(record)
            if err == 0.0 and z.min().item() > 0.5:
                break
    return {"steps_run": step, "epochs_run": round(step * batch / len(h), 1), "train_seconds": round(time.time() - start, 1),
            "optimizer": optimizer, "peak_tracked_hessian": round(peak_sharpness, 5)}


def per_item_gradients(module: nn.Module, h: torch.Tensor, sigma: torch.Tensor, sample: torch.Tensor) -> dict:
    """For sampled items: signed margin z_j and, per trainable tensor, |grad z_j|_1 and |grad z_j|_2^2."""
    params = [p for p in module.parameters() if p.requires_grad]
    z, l1, l2sq, rows = [], [], [], []
    for j in sample.tolist():
        module.zero_grad(set_to_none=True)
        value = module(h[j:j + 1])[0] * sigma[j]
        value.backward()
        z.append(value.item())
        l1.append([p.grad.abs().sum().item() for p in params])
        l2sq.append([p.grad.pow(2).sum().item() for p in params])
        rows.append(torch.cat([p.grad.reshape(-1) for p in params]))
    module.zero_grad(set_to_none=True)
    return {"z": torch.tensor(z), "l1": torch.tensor(l1), "l2sq": torch.tensor(l2sq), "jacobian": torch.stack(rows)}


def tangent_rank(jacobian: torch.Tensor) -> dict:
    """Spectrum of the items-by-parameters gradient matrix: a linearized adapter is a linear classifier on
    these rows, so Cover's theorem caps it near twice their numerical rank."""
    s = torch.linalg.svdvals(jacobian - jacobian.mean(0))
    if s.numel() == 0 or s[0] == 0:
        return {}
    return {"participation_ratio": round((s.pow(2).sum() ** 2 / s.pow(4).sum()).item(), 2),
            **{f"rank_rel_{t:g}": int((s > t * s[0]).sum().item()) for t in (1e-1, 1e-2, 1e-3)},
            "rows": int(jacobian.shape[0]), "top": round(s[0].item(), 6),
            "spectrum_rel": [float(f"{v:.3g}") for v in (s / s[0]).tolist()]}


def knee_predictions(module: nn.Module, grads: dict) -> dict:
    """Error after b-bit rounding, predicted two ways from first-order sensitivity."""
    scales = torch.tensor([p.detach().abs().max().item() for p in module.parameters() if p.requires_grad])
    z, l1, l2sq = grads["z"], grads["l1"], grads["l2sq"]
    normal = torch.distributions.Normal(0.0, 1.0)
    typical, worst = {}, {}
    for bits in BITS:
        step = 2 * scales / (2 ** bits - 1)
        std = (l2sq * step.pow(2) / 12).sum(1).sqrt().clamp_min(1e-30)
        typical[bits] = round(normal.cdf(-z / std).mean().item(), 5)
        # Guaranteed flip-free when the worst-case shift sum_t |g_t|_1 step_t / 2 stays below the margin.
        shift = (l1 * step / 2).sum(1)
        worst[bits] = round(((z <= 0) | (shift >= z)).float().mean().item(), 5)
    correct = z > 0
    need = torch.log2(1 + (l1 * scales).sum(1)[correct] / z[correct])
    return {
        "err_pred_typical": typical,
        "err_bound_worst": worst,
        "worst_case_bits_quantiles": {q: round(need.quantile(q).item(), 2) for q in (0.5, 0.9, 0.99)} if correct.any() else {},
        "K_inf_max": round((l1).sum(1).max().item(), 4),
        "K_2_max": round(l2sq.sum(1).sqrt().max().item(), 4),
    }


def top_hessian_eigenvalue(module: nn.Module, h: torch.Tensor, sigma: torch.Tensor, iters: int, seed: int) -> float:
    """Largest eigenvalue of the mean training loss's Hessian in the trainable parameters (power iteration)."""
    params = [p for p in module.parameters() if p.requires_grad]
    gen = torch.Generator().manual_seed(seed)
    vec = [torch.randn(p.shape, generator=gen) for p in params]
    norm = math.sqrt(sum(v.pow(2).sum().item() for v in vec))
    vec = [v / norm for v in vec]
    value = 0.0
    for _ in range(iters):
        loss = F.softplus(-module(h) * sigma).mean()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        hv = torch.autograd.grad(sum((g * v).sum() for g, v in zip(grads, vec)), params)
        value = sum((a * b).sum().item() for a, b in zip(hv, vec))
        norm = math.sqrt(sum(x.pow(2).sum().item() for x in hv))
        if norm == 0:
            return 0.0
        vec = [x / norm for x in hv]
    return value


def run_one(kind: str, size: int, items: int, lr: float, frozen: Frozen, h: torch.Tensor, sigma: torch.Tensor,
            args, log) -> dict:
    hs, ss = h[:items], sigma[:items]
    if kind == "probe":
        module = Probe(frozen, size, h[:max(items, args.probe_pca_items)])
    else:
        module = LoraHead(frozen, kind.split(":", 1)[1].split("+"), size, args.seed)
    base_err = ((frozen.final_hidden(hs) @ frozen.readout) * ss <= 0).float().mean().item()
    return measure(module, {"adapter": kind, "size": size, "items": items, "lr": lr, "optimizer": args.optimizer},
                   hs, ss, base_err, args, log)


def measure(module: nn.Module, header: dict, hs: torch.Tensor, ss: torch.Tensor, base_err: float, args, log) -> dict:
    """Train `module` on (hs, ss), then measure information, quantization knee, sensitivities and sharpness."""
    items, lr = len(hs), header["lr"]
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    tensors = sum(1 for p in module.parameters() if p.requires_grad)
    header = {**header, "params": params}
    run_log = lambda record: log({**header, **record})  # noqa: E731
    sample = torch.randperm(items, generator=torch.Generator().manual_seed(args.seed + 1))[:args.grad_items]
    tangent_init = tangent_rank(per_item_gradients(module, hs, ss, sample)["jacobian"])
    fit = train(module, hs, ss, args.steps, lr, args.batch, args.seed, run_log,
                eval_every=getattr(args, "eval_every", 100), optimizer=getattr(args, "optimizer", "adam"),
                sharpness_every=getattr(args, "sharpness_every", 0), sharpness_items=getattr(args, "hessian_items", 1024))
    z = margins(module, hs, ss)
    err = (z <= 0).float().mean().item()
    grads = per_item_gradients(module, hs, ss, sample)
    knee = knee_predictions(module, grads)
    tangent_trained = tangent_rank(grads.pop("jacobian"))
    hess_idx = torch.randperm(items, generator=torch.Generator().manual_seed(args.seed + 2))[:args.hessian_items]
    sharp = top_hessian_eigenvalue(module, hs[hess_idx], ss[hess_idx], args.hessian_iters, args.seed)
    trained = {name: p.detach().clone() for name, p in module.named_parameters() if p.requires_grad}
    quant = {}
    for bits in BITS:
        quantize_(module, bits)
        err_b = (margins(module, hs, ss) <= 0).float().mean().item()
        err_b_sample = (margins(module, hs[sample], ss[sample]) <= 0).float().mean().item()
        info_b = info_bits(items, err_b)
        budget = params * bits + 32 * tensors
        quant[bits] = {"err": round(err_b, 5), "err_sample": round(err_b_sample, 5), "info_bits": round(info_b, 1),
                       "bits_per_param": round(info_b / params, 4), "budget_bits": budget,
                       "counting_bound_ok": info_b <= budget}
        with torch.no_grad():
            for name, p in module.named_parameters():
                if name in trained:
                    p.copy_(trained[name])
    knee_bits = next((bits for bits in BITS if quant[bits]["err"] <= err + 0.01), None)
    info = info_bits(items, err)
    result = {**header, "event": "result", **fit, "base_err": round(base_err, 5), "err": round(err, 5),
              "err_sample": round((grads["z"] <= 0).float().mean().item(), 5),
              "info_bits": round(info, 1), "bits_per_param": round(info / params, 4),
              "label_bits_per_param": round(items / params, 4),
              "margin_quantiles": {q: round(z.quantile(q).item(), 4) for q in (0.01, 0.1, 0.5)},
              "knee_bits": knee_bits, "quantized": quant, **knee, "tangent_init": tangent_init,
              "tangent_trained": tangent_trained, "hessian_top": round(sharp, 5),
              "adam_lr_times_hessian": round(lr * sharp, 5)}
    log(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=int, default=32768, help="items cached; each run uses a prefix")
    parser.add_argument("--plan", default="default", help="'default', or entries like probe:256:512,lora:gate:1:2048")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "gd"],
                        help="gd = plain full-batch gradient descent at a constant step (set --batch >= items)")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--sharpness-every", type=int, default=0, help="log the top Hessian eigenvalue every k evaluations")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--probe-lr", default="0.05", help="comma list; each plan entry runs once per value")
    parser.add_argument("--probe-pca-items", type=int, default=4096, help="items used to fit the probe's whitening")
    parser.add_argument("--lora-lr", default="0.003", help="comma list; each plan entry runs once per value")
    parser.add_argument("--cache-file", type=Path, default=None,
                        help="optional: keep the residual cache on disk between invocations (default: RAM only)")
    parser.add_argument("--grad-items", type=int, default=512)
    parser.add_argument("--hessian-items", type=int, default=2048)
    parser.add_argument("--hessian-iters", type=int, default=20)
    parser.add_argument("--cache-batch", type=int, default=64)
    parser.add_argument("--wait-commit-gb", type=float, default=12.0,
                        help="wait for this much free commit before loading the model (0 = don't wait)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=config.OUT)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    config.lower_priority()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    if args.plan == "default":
        plan = [entry for entry in PLAN if entry[2] <= args.pool]
    else:
        plan = []
        for entry in args.plan.split(","):
            parts = entry.split(":")
            kind = parts[0] if parts[0] == "probe" else f"lora:{parts[1]}"
            plan.append((kind, int(parts[-2]), int(parts[-1])))
    if max(items for _, _, items in plan) > args.pool:
        raise SystemExit("a plan entry uses more items than --pool")

    name = args.name or time.strftime("run-%Y%m%d-%H%M%S")
    out_dir = args.output / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "results.jsonl"

    def log(record: dict) -> None:
        line = json.dumps(record)
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log({"event": "config", **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "plan": plan, "template": TEMPLATE, "answers": ANSWERS})
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL, revision=config.BASE_REVISION, cache_dir=config.HF_CACHE)
    ids, sigma = make_items(tok, args.pool, args.seed)
    wait_for_commit(args.wait_commit_gb)
    model = load_model("cpu")
    frozen = Frozen(model, tok)

    # Check the split computation against the full model on a few items before trusting the cache.
    with torch.no_grad():
        probe_ids = ids[:4]
        full = model(probe_ids).logits[:, -1]
        rows = [tok(a, add_special_tokens=False)["input_ids"][0] for a in ANSWERS]
        full_margin = full[:, rows[0]] - full[:, rows[1]]
    cache_file = args.cache_file
    if cache_file and cache_file.exists():
        saved = torch.load(cache_file)
        if not torch.equal(saved["ids"], ids):
            raise SystemExit(f"{cache_file} was built from other items; delete it or change --seed/--pool")
        h = saved["h"]
    else:
        h = cache_residual(model, ids, args.cache_batch)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"ids": ids, "h": h}, cache_file)
    split_margin = frozen.final_hidden(h[:4]) @ frozen.readout
    gap = (full_margin - split_margin).abs().max().item()
    log({"event": "split_check", "max_abs_margin_gap": gap, "tokens_per_prompt": ids.shape[1]})
    if gap > 1e-3:
        raise SystemExit(f"split computation disagrees with the full model by {gap}")
    del model
    import gc
    gc.collect()

    for kind, size, items in plan:
        for lr in (args.probe_lr if kind == "probe" else args.lora_lr).split(","):
            run_one(kind, size, items, float(lr), frozen, h, sigma, args, log)
    log({"event": "done"})


if __name__ == "__main__":
    main()
