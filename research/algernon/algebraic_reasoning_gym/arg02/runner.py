"""ARG-02 model runner: Qwen3.5 block-loop recurrence on public ARG-01 records.

The runner reads public JSONL rows exported by `analyze export` and never
builds the EvalMath roster, so hidden graphs and future coefficients cannot
reach a prompt. It writes per-case answer distributions, hidden-state
captures and timings under <run-dir>/parts/; `analyze` grades them with the
oracle afterwards. The external controller is the one arm that calls a
verifier during the run: it checks candidates against the public demos only.

Stages (see PROTOCOL.md):
    dev --stage shots     depth-1 answers with 0 and 4 shots on development rows
    dev --stage blocks    loop trajectories for each candidate block
    test                  every Qwen3.5-0.8B arm on the 192 test rows, once
    test --model ff       the Qwen3.5-2B feedforward control
    smoke                 the whole pipeline on a tiny random Qwen3.5, CPU

Exit 75 when the GPU is power-capped; `vramorama run --pause-exit 75`
restarts it and finished parts are reused.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config" / "arg02_config.json"
LOCK = HERE / "config" / "checkpoint_lock.json"
PAUSE_EXIT = 75
CODE_FILES = ("prompts.py", "recurrence.py", "runner.py", "scoring.py")


class Paused(Exception):
    pass


def file_sha256(path):
    h = sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def code_hashes():
    return {name: file_sha256(HERE / name) for name in CODE_FILES}


def read_public(path):
    from .prompts import PUBLIC_FIELDS
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != PUBLIC_FIELDS:
            raise ValueError(f"{path}:{n}: public row has fields {sorted(row)}")
        rows.append(row)
    return rows


def read_shots(path, k):
    from .prompts import SHOT_FIELDS
    shots = load_json(path)
    if k == 0:
        return []
    if any(set(s) != SHOT_FIELDS for s in shots):
        raise ValueError("shot rows must be public fields plus program")
    if k > len(shots):
        raise ValueError("not enough shots")
    return shots[:k]


def nvidia_smi():
    q = "name,driver_version,memory.total,memory.used,enforced.power.limit,pstate"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}
    return dict(zip(q.split(","), [x.strip() for x in out.split(",")]))


def enforced_power_limit():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class PowerWatch:
    def __init__(self, min_w, *, active, every=10.0, read=enforced_power_limit):
        self.min_w, self.active, self.every, self.read, self.last = min_w, active, every, read, None

    def check(self, where):
        if not self.active:
            return
        now = time.monotonic()
        if self.last is not None and now - self.last < self.every:
            return
        self.last = now
        watts = self.read()
        if watts is not None and watts < self.min_w:
            raise Paused(f"{where}: GPU power-capped at {watts:.0f} W")


class Parts:
    """Per-case results under <run-dir>/parts/<label>/, reused only under the same key."""

    def __init__(self, root, key):
        self.root, self.key = Path(root), key

    def _path(self, label, case_id, suffix):
        return self.root / label / (re.sub(r"[^A-Za-z0-9_.-]", "_", case_id) + suffix)

    def load(self, label, case_id):
        p = self._path(label, case_id, ".json")
        if not p.exists():
            return None
        rec = load_json(p)
        if rec.get("key") != self.key or rec.get("id") != case_id:
            return None
        if rec.get("arrays_sha256"):
            npz = self._path(label, case_id, ".npz")
            if not npz.exists() or file_sha256(npz) != rec["arrays_sha256"]:
                return None
        return rec["value"]

    def arrays(self, label, case_id):
        import numpy as np
        with np.load(self._path(label, case_id, ".npz")) as z:
            return {k: z[k] for k in z.files}

    def save(self, label, case_id, value, arrays=None):
        import numpy as np
        (self.root / label).mkdir(parents=True, exist_ok=True)
        rec = {"id": case_id, "key": self.key, "value": value}
        if arrays is not None:
            npz = self._path(label, case_id, ".npz")
            tmp = npz.with_name(npz.name + ".tmp")
            with tmp.open("wb") as fh:
                np.savez(fh, **arrays)
            os.replace(tmp, npz)
            rec["arrays_sha256"] = file_sha256(npz)
        write_atomic(self._path(label, case_id, ".json"), json.dumps(rec, sort_keys=True) + "\n")


def snapshot_dir(entry):
    env = os.environ.get(entry.get("snapshot_env", ""), "")
    if env:
        return Path(env)
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub" / ("models--" + entry["repo"].replace("/", "--")) / "snapshots" / entry["revision"]


def verify_snapshot(entry):
    root = snapshot_dir(entry)
    for name, digest in entry["files_sha256"].items():
        p = root / name
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint file missing: {p}")
        if file_sha256(p) != digest:
            raise AssertionError(f"checkpoint file differs from lock: {name}")
    return root


class CharTokenizer:
    """Smoke-test tokenizer: one id per character, operator tokens reserved."""

    def __init__(self, vocab=128):
        self.vocab = vocab
        self.ops = {" I": vocab - 4, " N": vocab - 3, " S": vocab - 2, " R": vocab - 1}

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            pair = text[i:i + 2]
            if pair in self.ops:
                out.append(self.ops[pair])
                i += 2
            else:
                out.append(ord(text[i]) % (self.vocab - 4))
                i += 1
        return out


def tiny_model(seed=0, vocab=128):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                            linear_value_head_dim=16, full_attention_interval=4, max_position_embeddings=4096,
                            tie_word_embeddings=True)
    model = Qwen3_5ForCausalLM(cfg).eval()
    return model, CharTokenizer(vocab)


def tiny_nanbeige():
    """Random 2-layer Nanbeige from the local snapshot's own remote code (needs transformers < 5)."""
    import torch
    from . import nanbeige
    from .settings import load_lock
    real = load_lock()["nanbeige"]
    root = snapshot_dir(real)
    torch.manual_seed(0)
    over = {"hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "kv_channels": 16}
    model = nanbeige.build(root, "cpu", torch.float32, "sdpa", over, random_init=True)
    return model, nanbeige.load_tokenizer(root, real["bos_token_id"])


def load_model(entry, device):
    import torch
    if entry.get("loader") == "nanbeige":
        from . import nanbeige
        root = verify_snapshot(entry)
        model = nanbeige.build(root, device, getattr(torch, entry["dtype"]), entry.get("attn_implementation", "sdpa"))
        return model, nanbeige.load_tokenizer(root, entry["bos_token_id"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = verify_snapshot(entry)
    dtype = getattr(torch, entry["dtype"])
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    # Load straight onto the GPU so the weights never also sit in host RAM.
    cuda = torch.device(device).type == "cuda"
    model = AutoModelForCausalLM.from_pretrained(root, dtype=dtype, local_files_only=True,
                                                 attn_implementation=entry.get("attn_implementation", "sdpa"),
                                                 device_map=str(device) if cuda else None)
    return (model if cuda else model.to(device)).eval(), tok


def op_token_ids(tok):
    from .prompts import OP_TEXT
    ids = {}
    for op, text in OP_TEXT.items():
        enc = tok.encode(text, add_special_tokens=False)
        if len(enc) != 1:
            raise AssertionError(f"operator {op!r} is not one token")
        ids[op] = enc[0]
    if len(set(ids.values())) != 4:
        raise AssertionError("operator tokens collide")
    return ids


class Engine:
    def __init__(self, model, tok, block, shots, device):
        from .prompts import TAIL
        self.model, self.tok, self.shots, self.device = model, tok, shots, device
        self.op_ids = op_token_ids(tok)
        if getattr(model.config, "model_type", None) == "nanbeige":
            from .nanbeige import NanbeigeLoop
            self.loop = NanbeigeLoop(model, block)
        else:
            from .recurrence import BlockLoop
            self.loop = BlockLoop(model, block)
        # Tail tokens as they appear after other text (sentencepiece tokenizers add a
        # dummy prefix at the start of a standalone string).
        head = self.tok.encode("x", add_special_tokens=False)
        full = self.tok.encode("x" + TAIL, add_special_tokens=False)
        if full[:len(head)] != head:
            raise AssertionError("tokenizer does not split the tail cleanly")
        self.tail = full[len(head):]

    def prompt_ids(self, row, variant="canonical", feedback=()):
        from .prompts import render
        text = render(row, shots=self.shots, variant=variant, feedback=feedback)
        encode = getattr(self.tok, "encode_prompt", None)
        ids = encode(text) if encode else self.tok.encode(text, add_special_tokens=False)
        if ids[-len(self.tail):] != self.tail:
            raise AssertionError("prompt tail tokenises differently")
        return ids

    def rows(self, row, variant="canonical", feedback=()):
        from .recurrence import candidate_rows
        ids = self.prompt_ids(row, variant, feedback)
        return candidate_rows(ids, self.op_ids), len(ids)


def _peak_reset(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    return None


def _usage(out, rows, length, layer_evals, extra=None):
    t = out["timing"]
    gpu_ms = t["prelude_ms"] + sum(t["block_ms"]) + sum(t["readout_ms"].values())
    rec = {"rows": rows, "length": length, "tokens": rows * length, "layer_evals": layer_evals,
           "gpu_ms": gpu_ms, "timing": {"prelude_ms": t["prelude_ms"], "block_ms": t["block_ms"],
                                       "readout_ms": {str(k): v for k, v in t["readout_ms"].items()}}}
    if extra:
        rec.update(extra)
    return rec


def _layer_evals(engine, rows, length, depth, readouts, mode="loop"):
    a, b = engine.loop.block
    n = engine.loop.n
    block_runs = depth if mode == "loop" else 1
    return rows * length * (a + block_runs * (b - a) + len(readouts) * (n - b))


def arm_loop(engine, row, steps, *, keep_states=False):
    ids, plen = engine.rows(row)
    depth = max(steps)
    out = engine.loop.trajectory(ids, engine.op_ids, depth, readout_steps=set(steps), keep_states=keep_states)
    value = {"records": {str(t): r for t, r in out["records"].items()}, "residual": out["residual"],
             "residual_site": out["residual_site"], "prompt_tokens": plen,
             "usage": _usage(out, out["rows"], out["length"],
                             _layer_evals(engine, out["rows"], out["length"], depth, steps))}
    import numpy as np
    states = np.stack([s.numpy() for s in out["states"]]).astype("float32")  # (depth+1, capture, H)
    return value, {"states": states}, out


def arm_open(engine, row, steps):
    ids, plen = engine.rows(row)
    out = engine.loop.trajectory(ids, engine.op_ids, max(steps), mode="open", readout_steps=set(steps))
    return {"records": {str(t): r for t, r in out["records"].items()}, "residual": out["residual"],
            "prompt_tokens": plen,
            "usage": _usage(out, out["rows"], out["length"],
                            _layer_evals(engine, out["rows"], out["length"], max(steps), steps, mode="open"))}


def donor_run(engine, row, depth, delta_steps, site_steps):
    ids, _ = engine.rows(row)
    return engine.loop.trajectory(ids, engine.op_ids, depth, keep_delta=set(delta_steps), keep_site=set(site_steps))


def arm_shuffle(engine, row, donor_out, donor_id, steps):
    ids, plen = engine.rows(row)
    out = engine.loop.trajectory(ids, engine.op_ids, max(steps), mode="shuffle", readout_steps=set(steps),
                                 donor_delta=donor_out["deltas"])
    rows, length = out["rows"], out["length"]
    a, b = engine.loop.block
    n = engine.loop.n
    donor_layers = donor_out["rows"] * donor_out["length"] * (a + max(steps) * (b - a))
    own_layers = rows * length * (a + (b - a) + len(steps) * (n - b))
    donor_ms = donor_out["timing"]["prelude_ms"] + sum(donor_out["timing"]["block_ms"])
    return {"records": {str(t): r for t, r in out["records"].items()}, "donor": donor_id, "prompt_tokens": plen,
            "usage": _usage(out, rows, length, own_layers + donor_layers,
                            {"donor_gpu_ms": donor_ms, "donor_tokens": donor_out["rows"] * donor_out["length"],
                             "donor_timing": {"prelude_ms": donor_out["timing"]["prelude_ms"],
                                              "block_ms": donor_out["timing"]["block_ms"]}})}


def arm_patch(engine, row, donor_out, donor_id, depth, steps):
    ids, plen = engine.rows(row)
    results = {}
    total_ms = 0.0
    layers = 0
    for t in steps:
        out = engine.loop.trajectory(ids, engine.op_ids, depth, readout_steps={depth},
                                     patch={"step": t, "site": donor_out["sites"][t]})
        u = _usage(out, out["rows"], out["length"],
                   _layer_evals(engine, out["rows"], out["length"], depth, [depth]))
        total_ms += u["gpu_ms"]
        layers += u["layer_evals"]
        results[str(t)] = {"record": out["records"][depth], "residual": out["residual"]}
    return {"depth": depth, "donor": donor_id, "patches": results, "prompt_tokens": plen,
            "usage": {"gpu_ms": total_ms, "layer_evals": layers}}


def arm_perturb(engine, row, clean, spec, seed):
    ids, plen = engine.rows(row)
    out = engine.loop.trajectory(ids, engine.op_ids, spec["depth"], readout_steps=set(spec["readout_steps"]),
                                 perturb={"step": spec["step"], "sigma": spec["sigma"], "seed": seed},
                                 clean_states=clean["full_states"])
    return {"records": {str(t): r for t, r in out["records"].items()}, "distance": out["distance"],
            "kick_norm": out.get("kick_norm"), "seed": seed, "prompt_tokens": plen,
            "usage": _usage(out, out["rows"], out["length"],
                            _layer_evals(engine, out["rows"], out["length"], spec["depth"], spec["readout_steps"]))}


def arm_ensemble(engine, row, variants):
    recs = {}
    total_ms = 0.0
    layers = 0
    for v in variants:
        ids, plen = engine.rows(row, variant=v)
        out = engine.loop.trajectory(ids, engine.op_ids, 1, readout_steps={1})
        recs[v] = out["records"][1]
        u = _usage(out, out["rows"], out["length"], out["rows"] * out["length"] * engine.loop.n)
        total_ms += u["gpu_ms"]
        layers += u["layer_evals"]
        recs[v + "_usage"] = u
    return {"records": {v: recs[v] for v in variants}, "usage_by_variant": {v: recs[v + "_usage"] for v in variants},
            "usage": {"gpu_ms": total_ms, "layer_evals": layers}}


def arm_controller(engine, row, rounds):
    """External iterative controller: verifier feedback on public demos, depth-1 model."""
    from .scoring import joint_scores, ranking, public_check
    tested, feedback, log = [], [], []
    total_ms = 0.0
    layers = 0
    accepted = False
    for r in range(1, rounds + 1):
        ids, plen = engine.rows(row, feedback=feedback)
        out = engine.loop.trajectory(ids, engine.op_ids, 1, readout_steps={1})
        rec = out["records"][1]
        u = _usage(out, out["rows"], out["length"], out["rows"] * out["length"] * engine.loop.n)
        total_ms += u["gpu_ms"]
        layers += u["layer_evals"]
        order = ranking(joint_scores(rec))
        choice = next(p for p in order if p not in tested)
        t0 = time.perf_counter_ns()
        ok, faults = public_check(row, choice)
        check_ns = time.perf_counter_ns() - t0
        tested.append(choice)
        log.append({"round": r, "program": list(choice), "accepted": ok, "faults": faults,
                    "prompt_tokens": plen, "gpu_ms": u["gpu_ms"], "check_ns": check_ns})
        if ok:
            accepted = True
            break
        feedback.append((choice, faults))
    return {"tested": [list(p) for p in tested], "accepted": accepted, "rounds": log,
            "usage": {"gpu_ms": total_ms, "layer_evals": layers, "forward_passes": len(log)}}


def environment(device):
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "device": str(device)}
    try:
        import numpy
        import torch
        import transformers
        env.update({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                    "transformers": transformers.__version__, "numpy": numpy.__version__})
    except ImportError as e:
        env["import_error"] = str(e)
    if str(device).startswith("cuda"):
        env["nvidia_smi"] = nvidia_smi()
    root = HERE.parents[3]
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()
    env["code"] = {"repo": "MoralityLabAI/Compactification-Theory", "commit": git("rev-parse", "HEAD"),
                   "arg02_dirty": bool(git("status", "--porcelain", "--", str(HERE.relative_to(root))))}
    return env


def part_key(cfg, entry, block, shots_k, stage, extra=None):
    key = {"stage": stage, "model": entry["repo"], "revision": entry["revision"], "dtype": entry.get("dtype"),
           "block": list(block) if block else None, "shots": shots_k, "code": code_hashes(),
           "config_sha256": sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}
    if extra:
        key.update(extra)
    return key


def run_dev(args, cfg, engine_factory, rows, watch, log, device):
    run = Path(args.run_dir)
    if args.stage == "shots":
        for k in cfg["shot_options"]:
            engine, entry = engine_factory(tuple(cfg["block_candidates"][0]), k)
            parts = Parts(run / "parts", part_key(cfg, entry, None, k, "dev_shots"))
            for row in rows:
                label = f"dev_shots_k{k}"
                if parts.load(label, row["case_id"]) is not None:
                    continue
                watch.check(row["case_id"])
                _peak_reset(device)
                value, _, _ = arm_loop(engine, row, [1])
                value["peak_bytes"] = _peak(device)
                parts.save(label, row["case_id"], value)
            log(f"dev shots k={k}: {len(rows)} rows")
        return
    sel = load_json(run / "selection_shots.json")
    k = sel["shots"]
    for block in cfg["block_candidates"]:
        engine, entry = engine_factory(tuple(block), k)
        parts = Parts(run / "parts", part_key(cfg, entry, block, k, "dev_blocks"))
        label = f"dev_block_{block[0]}_{block[1]}"
        for row in rows:
            if parts.load(label, row["case_id"]) is not None:
                continue
            watch.check(row["case_id"])
            _peak_reset(device)
            value, arrays, _ = arm_loop(engine, row, cfg["depths"])
            value["peak_bytes"] = _peak(device)
            parts.save(label, row["case_id"], value, arrays)
        log(f"dev block {block}: {len(rows)} rows")


def run_test(args, cfg, engine_factory, rows, watch, log, device):
    run = Path(args.run_dir)
    sel = load_json(run / "selection.json")
    block, k, dstar = tuple(sel["block"]), sel["shots"], sel["depth"]
    extra = {"selection_sha256": file_sha256(run / "selection.json")}
    if args.model == "ff":
        engine, entry = engine_factory(None, k, role="ff")
        parts = Parts(run / "parts", part_key(cfg, entry, None, k, "test_ff", extra))
        for row in rows:
            if parts.load("test_ff", row["case_id"]) is not None:
                continue
            watch.check(row["case_id"])
            _peak_reset(device)
            value, _, _ = arm_loop(engine, row, [1])
            value["peak_bytes"] = _peak(device)
            parts.save("test_ff", row["case_id"], value)
        log(f"test ff: {len(rows)} rows")
        return
    engine, entry = engine_factory(block, k)
    parts = Parts(run / "parts", part_key(cfg, entry, block, k, "test", extra))
    steps = list(range(1, max(cfg["depths"]) + 1))
    pd = cfg["perturb"]
    pdepth = max(2, dstar)
    psteps = sorted({1, math.ceil(pdepth / 2), pdepth})
    shuffle_steps = [d for d in cfg["depths"] if d > 1]
    variants = cfg["ensemble_variants"]
    ids = [r["case_id"] for r in rows]
    for i, row in enumerate(rows):
        cid = row["case_id"]
        donor = rows[i - 1]
        need = [lbl for lbl in (args.arms or TEST_LABELS) if parts.load(lbl, cid) is None]
        if not need:
            continue
        watch.check(cid)
        clean = None
        if "test_loop" in need or "test_perturb" in need:
            _peak_reset(device)
            value, arrays, clean = arm_loop(engine, row, steps, keep_states=True)
            value["peak_bytes"] = _peak(device)
            if "test_loop" in need:
                parts.save("test_loop", cid, value, arrays)
        if "test_perturb" in need:
            _peak_reset(device)
            value = arm_perturb(engine, row, clean, pd, pd["seed_base"] + i)
            value["peak_bytes"] = _peak(device)
            parts.save("test_perturb", cid, value)
        clean = None
        if "test_open" in need:
            _peak_reset(device)
            value = arm_open(engine, row, shuffle_steps)
            value["peak_bytes"] = _peak(device)
            parts.save("test_open", cid, value)
        if {"test_shuffle", "test_shuffle_a1", "test_patch"} & set(need):
            _peak_reset(device)
            dout = donor_run(engine, donor, max(shuffle_steps), shuffle_steps, psteps)
            for lbl in ("test_shuffle", "test_shuffle_a1"):
                if lbl in need:
                    _peak_reset(device)
                    value = arm_shuffle(engine, row, dout, donor["case_id"], shuffle_steps)
                    value["peak_bytes"] = _peak(device)
                    parts.save(lbl, cid, value)
            if "test_patch" in need:
                _peak_reset(device)
                value = arm_patch(engine, row, dout, donor["case_id"], pdepth, psteps)
                value["peak_bytes"] = _peak(device)
                parts.save("test_patch", cid, value)
            dout = None
        if "test_ensemble" in need:
            _peak_reset(device)
            value = arm_ensemble(engine, row, variants)
            value["peak_bytes"] = _peak(device)
            parts.save("test_ensemble", cid, value)
        if "test_controller" in need:
            _peak_reset(device)
            value = arm_controller(engine, row, cfg["controller"]["rounds"])
            value["peak_bytes"] = _peak(device)
            parts.save("test_controller", cid, value)
        log(f"test {i + 1}/{len(rows)} {cid}")
    write_atomic(run / "test_order.json", json.dumps(ids) + "\n")


TEST_LABELS = ("test_loop", "test_perturb", "test_open", "test_shuffle", "test_patch", "test_ensemble",
               "test_controller")
# Amendment 1: the donor shuffle rerun with the corrected position alignment.
AMENDED_LABELS = ("test_shuffle_a1",)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["dev", "test"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stage", choices=["shots", "blocks"])
    p.add_argument("--model", choices=["main", "ff"], default="main")
    p.add_argument("--profile", choices=["qwen", "nanbeige"], default="qwen")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true", help="tiny random Qwen3.5 on CPU, no checkpoint")
    p.add_argument("--limit", type=int)
    p.add_argument("--arms", nargs="+", choices=TEST_LABELS + AMENDED_LABELS,
                   help="test stage: compute only these part labels")
    args = p.parse_args(argv)
    from .settings import load_config, model_entry, ff_entry
    cfg = load_config(args.smoke, args.profile)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    exported = run / "export_manifest.json"
    if exported.exists() and load_json(exported).get("profile", "qwen") != args.profile:
        raise SystemExit("run directory was exported for a different profile")
    if args.model == "ff" and ff_entry(cfg) is None:
        raise SystemExit(f"profile {args.profile} has no feedforward control")
    logf = run / "job.log"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with logf.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    import torch
    device = torch.device("cpu" if args.smoke else args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg["cuda_memory_fraction"], 0)
    watch = PowerWatch(cfg["min_power_w"], active=device.type == "cuda")
    loaded = {}

    def engine_factory(block, k, role="main"):
        entry = model_entry(cfg) if role == "main" else ff_entry(cfg)
        if role not in loaded:
            for other in list(loaded):
                del loaded[other]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.smoke:
                loaded[role] = tiny_nanbeige() if args.profile == "nanbeige" else tiny_model()
            else:
                loaded[role] = load_model(entry, device)
        model, tok = loaded[role]
        shots = read_shots(run / "public" / "shots.json", k)
        eng_block = block if block is not None else (0, len(model.model.layers))
        engine = Engine(model, tok, eng_block, shots, device)
        name = "model" if role == "main" else "ff_control"
        write_atomic(run / f"op_ids_{name}.json", json.dumps(engine.op_ids, sort_keys=True))
        return engine, entry

    split = "development" if args.command == "dev" else "test"
    rows = read_public(run / "public" / f"{split}.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    manifest_name = f"manifest_{args.command}_{args.stage or args.model}" +         ("_" + "_".join(args.arms) if args.arms else "") + ".json"
    manifest = {"command": args.command, "profile": args.profile, "arms": args.arms, "stage": args.stage, "model": args.model, "smoke": args.smoke,
                "environment": environment(device), "config_sha256": file_sha256(CONFIG),
                "lock_sha256": file_sha256(LOCK), "rows": len(rows), "started": time.time()}
    write_atomic(run / manifest_name, json.dumps(manifest, indent=2) + "\n")
    try:
        with torch.inference_mode():
            if args.command == "dev":
                if args.stage is None:
                    raise SystemExit("dev needs --stage")
                run_dev(args, cfg, engine_factory, rows, watch, log, device)
            else:
                run_test(args, cfg, engine_factory, rows, watch, log, device)
    except Paused as e:
        log(f"PAUSE: {e}; exit {PAUSE_EXIT}")
        sys.exit(PAUSE_EXIT)
    manifest["finished"] = time.time()
    write_atomic(run / manifest_name, json.dumps(manifest, indent=2) + "\n")
    log("done")


if __name__ == "__main__":
    main()
