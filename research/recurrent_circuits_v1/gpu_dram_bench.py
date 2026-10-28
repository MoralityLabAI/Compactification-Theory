"""Bounded inference/transport pilot for recurrent circuits and a shared transformer.

No model weights are downloaded. CUDA requires --confirm-gpu. The spill arm
measures explicit roundtrip cost; it does NOT lower single-session peak VRAM.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--confirm-gpu", action="store_true")
    p.add_argument("--kernel", choices=("nand", "transformer"), default="nand")
    p.add_argument("--mode", choices=("resident", "spill"), default="resident")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--loops", type=int, default=32)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--max-vram-gib", type=float, default=8)
    p.add_argument("--max-host-gib", type=float, default=1)
    p.add_argument("--max-seconds", type=float, default=120)
    p.add_argument("--pageable", action="store_true", help="Use pageable host buffer instead of pinned")
    p.add_argument("--out", type=Path)
    return p


def validate(args) -> None:
    for name in ("loops", "chunk", "batch", "tokens", "width", "heads", "repeats"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.warmup < 0 or args.loops > 4096 or args.repeats > 100:
        raise ValueError("warmup must be nonnegative; loops<=4096; repeats<=100")
    if args.width % args.heads:
        raise ValueError("width must be divisible by heads")
    for name in ("max_vram_gib", "max_host_gib", "max_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"invalid {name}")
    if args.device == "cuda" and not args.confirm_gpu:
        raise ValueError("CUDA execution requires --confirm-gpu")
    if args.device == "cpu" and args.mode == "spill":
        raise ValueError("spill requires CUDA; a CPU copy would not measure host/device transfers")
    if args.device == "cpu" and args.dtype != "float32":
        raise ValueError("CPU smoke mode is restricted to float32")


def execute(args) -> dict:
    validate(args)
    import torch
    from torch import nn
    from reference import roundtrip_bytes

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no fallback benchmark was run")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    state_bytes = args.batch * args.tokens * args.width * elem_bytes
    # Conservative planning estimate. Actual framework/context overhead is measured too.
    parameter_estimate = (12 * args.width**2 + 32 * args.width) * elem_bytes if args.kernel == "transformer" else 0
    attention_estimate = 4 * args.batch * args.heads * args.tokens**2 * elem_bytes if args.kernel == "transformer" else 0
    plan_device = 8 * state_bytes + 4 * parameter_estimate + attention_estimate
    plan_host = 8 * state_bytes + 4 * parameter_estimate
    if plan_host > args.max_host_gib * 2**30:
        raise ValueError("conservative tensor plan exceeds host budget")
    props = None
    if args.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        if plan_device > min(args.max_vram_gib * 2**30, free * 0.6):
            raise ValueError("conservative device plan exceeds budget or free-memory reserve")
        props = torch.cuda.get_device_properties(0)

    model = None
    if args.kernel == "transformer":
        model = nn.TransformerEncoderLayer(args.width, args.heads, dim_feedforward=4*args.width,
                    dropout=0.0, activation="relu", batch_first=True, norm_first=True).eval()
        source = torch.randn(args.batch, args.tokens, args.width).to(dtype)
    else:
        source = torch.randint(0, 2, (args.batch, args.tokens, args.width)).to(dtype)
    parameter_count = sum(p.numel() for p in model.parameters()) if model is not None else 0
    parameter_digest = hashlib.sha256()
    if model is not None:
        for name, parameter in model.state_dict().items():
            parameter_digest.update(name.encode())
            parameter_digest.update(parameter.detach().contiguous().numpy().tobytes())
        model = model.to(device=args.device, dtype=dtype)
    state = source.to(args.device)
    pinned = args.device == "cuda" and not args.pageable and args.mode == "spill"
    host = torch.empty(source.shape, dtype=dtype, device="cpu", pin_memory=pinned) if args.mode == "spill" else None

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    deadline = time.perf_counter() + args.max_seconds

    def step(h):
        if model is not None:
            return model(h)
        y = torch.roll(h, 1, dims=-1)
        v = torch.relu(h+y-1)
        return 1-2*torch.relu(v-0.25)+2*torch.relu(v-0.75)

    def run(spill):
        h = state.clone()
        sync()  # initialization excluded from hot-loop timing
        start = time.perf_counter()
        for i in range(args.loops):
            if time.perf_counter() >= deadline:
                raise TimeoutError("soft run budget reached between steps")
            h = step(h)
            if spill and ((i+1) % args.chunk == 0 or i+1 == args.loops):
                # One stream orders D2H before H2D and protects buffer lifetime.
                host.copy_(h, non_blocking=pinned)
                h.copy_(host, non_blocking=pinned)
        sync()
        elapsed = time.perf_counter()-start
        # CPU access occurs only after device synchronization.
        return h.cpu(), elapsed

    samples = []
    with torch.inference_mode():
        # A separate resident execution is the transport reference.
        expected, reference_seconds = run(False)
        if args.kernel == "nand":
            logical = source.bool()
            for _ in range(args.loops):
                logical = ~(logical & torch.roll(logical, 1, dims=-1))
            if not torch.equal(expected, logical.to(dtype)):
                raise RuntimeError("compiled NAND disagrees with independent Boolean replay")
        for _ in range(args.warmup):
            run(args.mode == "spill")
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            result, elapsed = run(args.mode == "spill")
            if not torch.isfinite(result).all():
                raise RuntimeError("non-finite recurrent output")
            if not torch.equal(result, expected):
                raise RuntimeError("transport path differs bitwise from resident path")
            samples.append(elapsed)
    allocated = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
    reserved = torch.cuda.max_memory_reserved() if args.device == "cuda" else None
    if reserved is not None and reserved > args.max_vram_gib * 2**30:
        raise RuntimeError("measured allocator peak exceeded declared budget")
    rss = None
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except ImportError:
        pass
    transfers = roundtrip_bytes(state_bytes, args.loops, args.chunk) if args.mode == "spill" else 0
    script_paths = (Path(__file__), Path(__file__).with_name("reference.py"))
    return {
        "schema_version": 1, "evidence": "systems_microbenchmark_untrained" if model else "compiled_tensor_circuit_microbenchmark",
        "model_download": False, "learning_claim": False, "vram_capacity_gain_tested": False,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k != "out"},
        "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device_name": props.name if props is not None else platform.processor() or "CPU",
        "state_bytes": state_bytes, "trainable_parameters": parameter_count,
        "parameter_storage_bytes": parameter_count*elem_bytes,
        "parameter_initialization_sha256": parameter_digest.hexdigest() if model else None,
        "nand_constants": "fixed scalar operations in hashed source" if model is None else None,
        "planned_device_tensor_bytes": plan_device, "planned_host_tensor_bytes": plan_host,
        "host_rss_after_run_bytes": rss, "host_rss_is_peak": False,
        "pinned_host_bytes": state_bytes if pinned else 0,
        "gpu_peak_allocated_bytes": allocated, "gpu_peak_reserved_bytes": reserved,
        "h2d_bytes_per_measured_repeat": transfers//2,
        "d2h_bytes_per_measured_repeat": transfers//2,
        "initial_input_transfer_bytes_excluded": state_bytes if args.device == "cuda" else 0,
        "terminal_output_transfer_bytes_per_repeat_excluded": state_bytes if args.device == "cuda" else 0,
        "timing_scope": "hot loops and explicit spills; excludes load, initial clone, and terminal copy/digest",
        "max_seconds_scope": "soft host deadline after setup, checked between steps; synchronization may overrun",
        "reference_seconds": reference_seconds, "wall_seconds": samples,
        "median_wall_seconds": statistics.median(samples),
        "loop_steps_per_second": args.loops/statistics.median(samples),
        "bitwise_equal_to_resident": True,
        "boolean_oracle_checked": model is None,
        "output_sha256": hashlib.sha256(result.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in script_paths},
    }


def main():
    args = parser().parse_args()
    try:
        result = execute(args)
    except (ValueError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(f"benchmark stopped: {exc}") from exc
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
