"""Model pin, paths and process niceness for the adapter-bits experiments.

Override the paths with environment variables:
- ADAPTER_BITS_HF_CACHE: Hugging Face cache directory holding the model (default: the hub's own default).
- ADAPTER_BITS_OUT: where run directories go (default: runs/ in this repository, which git ignores).
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

HF_CACHE = os.environ.get("ADAPTER_BITS_HF_CACHE") or None
OUT = Path(os.environ.get("ADAPTER_BITS_OUT", Path(__file__).resolve().parents[1] / "runs"))


def lower_priority() -> None:
    """Run below normal priority, so a long CPU job does not starve other work on the machine."""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p  # a pseudo-handle; the default int return truncates it
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
    else:
        os.nice(10)
