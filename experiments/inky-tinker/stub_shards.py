"""Header-only sparse stubs of the Inkling-2-Mini safetensors shards: tensor names and shapes, no tensor data.

tinker_cookbook.weights.build_lora_adapter reads the base model's tensor names and shapes from safetensors headers.
Pointed at the hub id it snapshots the whole 19.5 GB checkpoint (that filled D: and broke WSL on 2026-11-25), so
give it this directory instead: config/tokenizer files plus each shard's header, truncated to the shard's full
size (sparse on ext4, so no data blocks are allocated; `du -sh` shows a few MB).

  python stub_shards.py --out /root/inky-tinker/inkling-2-mini-stub
"""
import argparse
import json
import os
import struct

import requests
from huggingface_hub import HfApi, snapshot_download

REPO = "thinkingmachines/Inkling-2-Mini"
REV = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/inky-tinker/inkling-2-mini-stub")
    ap.add_argument("--revision", default=REV)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    snapshot_download(REPO, revision=a.revision, local_dir=a.out,
                      allow_patterns=["*.json", "*.jinja", "tiktoken/*", "*.tiktoken", "*.model", "*.txt"])
    idx = json.load(open(os.path.join(a.out, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    sizes = {p.path: p.size for p in HfApi().get_paths_info(REPO, shards, revision=a.revision)}
    for sh in shards:
        url = f"https://huggingface.co/{REPO}/resolve/{a.revision}/{sh}"
        n = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60).content)[0]
        head = requests.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=120).content
        assert len(head) == n, (sh, len(head), n)
        p = os.path.join(a.out, sh)
        with open(p, "wb") as fh:
            fh.write(struct.pack("<Q", n) + head)
            fh.truncate(sizes[sh])
        print(sh, sizes[sh], "header", n, flush=True)
    print("stub dir", a.out, "disk use:", os.popen(f"du -sh {a.out}").read().strip())


if __name__ == "__main__":
    main()
