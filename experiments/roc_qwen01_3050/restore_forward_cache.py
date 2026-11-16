"""Restore published, hash-verified CPU forward receipts for matching replay."""
import gzip
import json
import importlib.metadata as md
import shutil
from preflight import RESULTS, sha


def main():
    source=RESULTS/"forward_receipts.sqlite.gz"
    manifest=json.loads((RESULTS/"forward_receipts_manifest.json").read_text())
    for name,version in manifest["packages"].items():
        if version is not None and md.version(name)!=version:
            raise RuntimeError(f"Locked environment mismatch for {name}; do not reuse these receipts")
    if sha(source)!=manifest["compressed_sha256"]:
        raise RuntimeError("Compressed forward receipt hash mismatch")
    dest=RESULTS/"forward_cache.sqlite"
    if dest.exists():
        if sha(dest)!=manifest["sqlite_sha256"]:
            raise RuntimeError("Existing cache differs; preserve it and choose another checkout")
        print("Matching forward cache already present")
        return
    temp=RESULTS/"forward_cache.sqlite.tmp"
    with gzip.open(source,"rb") as inp,open(temp,"wb") as out:
        shutil.copyfileobj(inp,out)
    if sha(temp)!=manifest["sqlite_sha256"]:
        raise RuntimeError("Restored forward receipt hash mismatch")
    temp.replace(dest)
    print("Restored source-keyed forward receipts; no model was loaded")


if __name__=="__main__":
    main()
