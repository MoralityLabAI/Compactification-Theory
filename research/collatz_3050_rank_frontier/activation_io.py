"""Read the pinned BPD-COL-02 activation receipt format without loading a model."""
import hashlib
import json
import re
import struct
from pathlib import Path
import numpy as np

SOURCE = "f1d489a000406e2d8732f4ea7a39e5473392efe2"
MODEL = "451ed48c3273ecef7ea8faaa43c31ce529763bb1"
ROLES = ("recipient", "effect_donor", "sham_donor")
ARMS = ("identity", "effect", "sham", "random", "full_donor_prefix")
WIDTH = 3072
MAX_BYTES = 8 * 1024 * 1024


def require(ok, why):
    if not ok:
        raise ValueError(why)


def manifest_digest(m):
    return hashlib.sha256((json.dumps({k:v for k,v in m.items() if k!="manifest_sha256"},
        sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()).hexdigest()


def read_activation(path, expected_hash, tokens):
    """Bound file and header lengths, check exact names, contiguous BF16 layout and hash."""
    require(type(tokens) is int and 2 <= tokens <= 128, "bad token count")
    path = Path(path)
    require(8 <= path.stat().st_size <= MAX_BYTES, "activation size outside profile")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_hash, "activation hash mismatch")
    n = struct.unpack_from("<Q", data)[0]
    require(2 <= n <= min(65536,len(data)-8), "invalid safetensors header")
    header = json.loads(data[8:8+n])
    header.pop("__metadata__",None)
    expected = {r+".pass1_boundary": [1,tokens,WIDTH] for r in ROLES}
    expected.update({r+".final_query": [WIDTH] for r in ROLES})
    expected.update({"arm."+r+".final_query": [WIDTH] for r in ARMS})
    require(set(header) == set(expected), "unexpected tensor keys")
    payload = memoryview(data)[8+n:]
    arrays, intervals = {}, []
    for key, shape in expected.items():
        info = header[key]
        require(info["dtype"] == "BF16" and info["shape"] == shape, "tensor profile mismatch: "+key)
        start,end = info["data_offsets"]
        require(type(start) is int and type(end) is int and 0<=start<=end<=len(payload)
                and end-start==2*int(np.prod(shape)), "bad tensor range")
        intervals.append((start,end))
        values = (np.frombuffer(payload[start:end],dtype="<u2").astype(np.uint32)<<16).view(np.float32).reshape(shape)
        require(np.isfinite(values).all(), "nonfinite tensor")
        arrays[key]=values
    end=0
    for a,b in sorted(intervals):
        require(a==end,"overlap or gap in tensor payload")
        end=b
    require(end==len(payload),"trailing tensor payload")
    return arrays


def read_run(root):
    root=Path(root).resolve()
    manifest=json.loads((root/"cases_manifest.json").read_text(encoding="utf-8"))
    contract=json.loads((root/"run_contract.json").read_text(encoding="utf-8"))
    require(manifest["schema"]=="BPD-COL-02/public-development-cases/v1" and
            manifest["specification"]["split"]=="development_only" and
            manifest["specification"]["confirmation_authorized"] is False,"manifest scope")
    require(manifest_digest(manifest)==manifest["manifest_sha256"]==contract["cases_manifest_sha256"],"manifest hash")
    require(contract["schema"]=="BPD-COL-02/native-streamed-run/v1" and
            contract["model_id"]=="Nanbeige/Nanbeige4.2-3B" and contract["revision"]==MODEL and
            contract["loop_count"]==2 and contract["dtype"]=="bfloat16" and
            contract["source_sha256"]["streaming.py"]=="273d6193288be6a2e3f1b895875cca2379e6acc18631abede1831c3e8f3b639d" and
            contract["source_sha256"]["run_gym.py"]=="d7cc5d617a25c17c06c4e16e9b2faca7a579f77a566147255b47472e416fee0f", "source profile")
    cases={c["case_id"]:c for c in manifest["cases"]}
    require(len(cases)==manifest["n_cases"]==len(manifest["cases"]) and 1<=len(cases)<=64,"manifest case count")
    for cid,c in cases.items():
        require(re.fullmatch(r"COL3050-d[0-9]{6}-h[12]-(trace|arithmetic)",cid) is not None and type(c["root_id"]) is int and c["root_id"]==c["d"],"invalid case identity")
    records={}
    records_file=root/"records.jsonl"
    for line in records_file.read_text(encoding="utf-8").splitlines() if records_file.exists() else []:
        rec=json.loads(line)
        cid=rec["case_id"]
        require(cid in cases and cid not in records and rec["root_id"]==cases[cid]["root_id"],"foreign or duplicate record")
        require(rec["status"] in ("completed","error","skipped"),"unknown status")
        records[cid]=rec
    loaded=[]
    for cid,c in cases.items():
        rec=records.get(cid)
        if rec is None or rec["status"]!="completed":
            continue
        expected=root/"activations"/(cid+".safetensors")
        declared=(root/rec["activations"]["path"]).resolve()
        require(declared==expected and declared.is_relative_to(root),"unexpected activation path")
        loaded.append((c,rec,read_activation(declared,rec["activations"]["sha256"],rec["input_tokens"])))
    return manifest,loaded
