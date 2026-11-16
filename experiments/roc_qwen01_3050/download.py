"""Range-resumable public HF download; verify immutable LFS hashes before loading."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import requests
from huggingface_hub import HfApi
from preflight import ROOT, save, sha


def checkpoint(model, revision):
    info = HfApi().model_info(model, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError("Immutable model revision mismatch")
    target = ROOT / "cache" / "checkpoint" / revision
    target.mkdir(parents=True, exist_ok=True)
    for entry in info.siblings:
        name, size = entry.rfilename, entry.size
        if not name.endswith((".json", ".safetensors", ".txt", ".model")):
            continue
        dest = target / name
        expected = entry.lfs.sha256 if entry.lfs else None
        if dest.exists() and dest.stat().st_size == size and (expected is None or sha(dest) == expected):
            continue
        partial = dest.with_suffix(dest.suffix + ".partial")
        progress = dest.with_suffix(dest.suffix + ".ranges.json")
        chunk = 8 << 20
        completed = set(json.loads(progress.read_text())) if progress.exists() else set()
        if not partial.exists():
            with open(partial, "wb") as f:
                f.truncate(size)
        def fetch(i):
            a, b = i*chunk, min(size, (i+1)*chunk)-1
            url = f"https://huggingface.co/{model}/resolve/{revision}/{name}?download=true&range_start={a}"
            for attempt in range(5):
                try:
                    r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=(20, 45))
                    r.raise_for_status()
                    if len(r.content) != b-a+1:
                        raise RuntimeError(f"Unexpected range response {r.status_code}, {len(r.content)}")
                    if r.status_code == 206 and r.headers.get("Content-Range", "").split("/")[0] != f"bytes {a}-{b}":
                        raise RuntimeError("Server returned a different range")
                    with open(partial, "r+b") as f:
                        f.seek(a)
                        f.write(r.content)
                    return i
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(2**attempt, 8))
        jobs = [i for i in range((size+chunk-1)//chunk) if i not in completed]
        with ThreadPoolExecutor(max_workers=12) as pool:
            for done in as_completed([pool.submit(fetch, i) for i in jobs]):
                completed.add(done.result())
                save(progress, sorted(completed))
                if len(completed) % 16 == 0 or len(completed)*chunk >= size:
                    print(f"download {name}: {min(size,len(completed)*chunk)}/{size}", flush=True)
        if expected is not None and sha(partial) != expected:
            raise RuntimeError(f"LFS SHA256 mismatch for {name}")
        partial.replace(dest)
    return target
