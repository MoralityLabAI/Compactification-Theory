"""Tinker adapter (downloaded by tinker_sft.py into <run>/adapter) to a PEFT adapter in <run>/adapter_peft, which
tinker_lora_to_gguf.py and squid_overlay.py read. Uses the header-only stub from stub_shards.py as the base.

  python to_peft.py --run /mnt/c/.../inky-tinker/full [--stub /root/inky-tinker/inkling-2-mini-stub] [--out DIR]
"""
import argparse
import collections
import json
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinker_env import load_key  # noqa: E402

load_key()
from tinker_cookbook import weights  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="tinker_sft.py --out directory")
    ap.add_argument("--stub", default="/root/inky-tinker/inkling-2-mini-stub")
    ap.add_argument("--out", default="", help="PEFT output directory (default <run>/adapter_peft; must not exist)")
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "adapter_peft"
    weights.build_lora_adapter(base_model=a.stub, adapter_path=str(run / "adapter"), output_path=str(out))
    with open(out / "adapter_model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    print("tensors", len(h))
    pat = collections.Counter(re.sub(r"\d+", "N", k) for k in h)
    for k, c in pat.most_common(20):
        ex = next(x for x in h if re.sub(r"\d+", "N", x) == k)
        print(f"  {c:5d} {k}  {h[ex]['shape']}")
    cfg = json.loads((out / "adapter_config.json").read_text())
    print({k: cfg.get(k) for k in ("r", "lora_alpha", "target_modules", "target_parameters")})


if __name__ == "__main__":
    main()
