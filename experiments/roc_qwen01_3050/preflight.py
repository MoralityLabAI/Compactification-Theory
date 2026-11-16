"""Fail-closed device audit. Run before importing/loading any model."""
import csv
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CEILING = 3_400_000_000  # decimal GB, stricter than 3.4 GiB
ALLOCATOR_CAP = 3_200_000_000  # additional 200 MB safety margin


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def audit():
    RESULTS.mkdir(exist_ok=True)
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits"], text=True)
    devices = [dict(zip(("index", "uuid", "name", "memory_mib", "used_mib", "driver"),
                        [v.strip() for v in row])) for row in csv.reader(query.splitlines())]
    allowed = [x for x in devices if "RTX 3050 Laptop" in x["name"] and int(x["memory_mib"]) == 4096]
    if len(allowed) != 1:
        raise RuntimeError(f"Exactly one RTX 3050 Laptop 4GB required: {devices}")
    device = allowed[0]
    # UUID identifies the audited physical GPU even if enumeration changes.
    os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import psutil
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA on the audited UUID is unavailable")
    prop = torch.cuda.get_device_properties(0)
    if "RTX 3050 Laptop" not in prop.name or prop.total_memory > 4_300_000_000:
        raise RuntimeError(f"CUDA device disagrees with nvidia-smi: {prop}")
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP / prop.total_memory, 0)
    torch.set_num_threads(2)
    packages = {}
    for p in ("torch", "transformers", "accelerate", "bitsandbytes", "numpy", "scipy",
              "huggingface-hub", "safetensors", "tokenizers", "psutil", "triton"):
        try:
            packages[p] = md.version(p)
        except md.PackageNotFoundError:
            packages[p] = None
    profile = dict(devices=devices, selected_device=device, cuda_device_name=prop.name,
                   cuda_compute_capability=list(torch.cuda.get_device_capability(0)),
                   pytorch_cuda=torch.version.cuda, packages=packages,
                   python=platform.python_version(), platform=platform.platform(),
                   host_ram_total=psutil.virtual_memory().total,
                   host_ram_available=psutil.virtual_memory().available,
                   allocated_ceiling_bytes=CEILING, allocator_cap_bytes=ALLOCATOR_CAP,
                   source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         cwd=ROOT, text=True).strip())
    save(RESULTS / "hardware_profile.json", profile)
    print(json.dumps(profile), flush=True)
    return torch, profile


if __name__ == "__main__":
    audit()
