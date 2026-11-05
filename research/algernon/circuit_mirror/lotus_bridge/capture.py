"""LOTUS post-loop hidden-state extraction, without changing upstream weights.

This calls the *published* Lotus.forward, instruments its exact efficient_forward
loop boundary, and collects each pass's last latent-token hidden vector.
Final LM logits and answer generation are deliberately omitted in capture mode.
The public model and upstream dependency must be installed on the executing host.

No Hugging Face download occurs unless --allow-download is supplied, and the
resolved HF revision is pinned before loading the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from ..spec import digest
from .roster import prompt_for, validate_roster

UPSTREAM_SHA = "eb77e2f7909c5006f58ff0ad7cd6629b942caa9e"
DEFAULT_MODEL = "yingfanbot/gsm-lotus-llama3b"
COMMIT_RE = re.compile(r"^[a-f0-9]{40}$")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_upstream(repo: Path, *, expected: str = UPSTREAM_SHA) -> str:
    source = repo / "scripts" / "lotus.py"
    if not source.is_file():
        raise FileNotFoundError(f"LOTUS source missing: {source}")
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                         capture_output=True, text=True).stdout.strip()
    if out != expected:
        raise ValueError(f"LOTUS upstream commit mismatch: got {out}; require {expected}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "scripts/lotus.py"],
                           cwd=repo, check=True, capture_output=True, text=True).stdout
    if dirty:
        raise ValueError("upstream scripts/lotus.py modified; refuse untracked instrumentation")
    return file_sha256(source)


def import_lotus_module(repo: Path):
    path = repo / "scripts" / "lotus.py"
    key = "algernon_lotus_pinned"
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load LOTUS source at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_hidden(hidden, *, first_latent: int, last_latent: int,
                  loop_start: int, loop_end: int, c_thought: int):
    """Return independent last-token and block-mean vectors on CPU.

    Upstream efficient_forward returns either the sliced loop region, or the
    entire prefix + loop if its prefix cache could not be constructed.
    """
    import torch
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("LOTUS capture requires batch size one, [batch,sequence,hidden]")
    if hidden.shape[1] == loop_end - loop_start:
        offset = loop_start
    elif hidden.shape[1] == loop_end:
        offset = 0
    else:
        raise ValueError(f"unexpected LOTUS loop output length: {hidden.shape[1]}")
    end = last_latent - offset + 1
    start = max(first_latent - offset, end - c_thought)
    if start >= end or end > hidden.shape[1]:
        raise ValueError("latent positions outside actual post-loop hidden states")
    last = hidden[0, end-1, :].detach().to(dtype=torch.float32, device="cpu").clone()
    mean = hidden[0, start:end, :].detach().float().mean(0).cpu().clone()
    if not (torch.isfinite(last).all() and torch.isfinite(mean).all()):
        raise ValueError("nonfinite LOTUS hidden features")
    return torch.stack((last, mean), dim=0)


def select_blocks(hidden, logits, *, first_latent: int, last_latent: int,
                  loop_start: int, loop_end: int, c_thought: int):
    """Every latent block of one pass, plus the static start token and readout ids.

    Upstream supervises block b (latents first+b*c .. first+(b+1)*c-1) with CoT
    step b after the final loop, so step k of a task lives in block k-1, not in
    the last block. Returns CPU tensors: blocks [n_blocks, 2, H] float32 with
    pools (last latent of block, block mean) in select_hidden's order; static
    [H], the start-latent token, which attends only to the prompt and is never
    re-injected; readout [loop_end-loop_start] int32, the argmax of upstream's
    own logits over the loop region (position b*c predicts block b's first CoT
    token), or None when logits were not computed.
    """
    import torch
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("LOTUS capture requires batch size one, [batch,sequence,hidden]")
    if hidden.shape[1] == loop_end - loop_start:
        offset = loop_start
    elif hidden.shape[1] == loop_end:
        offset = 0
    else:
        raise ValueError(f"unexpected LOTUS loop output length: {hidden.shape[1]}")
    n_lat = last_latent - first_latent + 1
    if n_lat <= 0 or n_lat % c_thought:
        raise ValueError("latent span must be a positive multiple of c_thought")
    lat = hidden[0, first_latent-offset:last_latent-offset+1, :]
    if lat.shape[0] != n_lat:
        raise ValueError("latent positions outside actual post-loop hidden states")
    lat = lat.detach().reshape(n_lat // c_thought, c_thought, -1)
    blocks = torch.stack((lat[:, -1, :].float(), lat.float().mean(1)), dim=1).cpu()
    static = hidden[0, loop_start-offset, :].detach().float().cpu().clone()
    if not (torch.isfinite(blocks).all() and torch.isfinite(static).all()):
        raise ValueError("nonfinite LOTUS hidden features")
    readout = None
    if logits is not None and logits.shape[-1] > 1:
        if logits.shape[1] != hidden.shape[1]:
            raise ValueError("logits and hidden states cover different positions")
        readout = logits[0, loop_start-offset:loop_end-offset, :].argmax(-1).to(torch.int32).cpu()
    return blocks, static, readout


@contextmanager
def intercept_loop_states(lotus_module, *, first_latent: int,
                          last_latent: int, c_thought: int, suppress_logits=True,
                          record_blocks=False):
    """Instrument only this model call. Unconditionally restore original code.

    capture contains one entry from the initial prefix pass and one entry per
    refinement. It intentionally does not catch calls on other threads. With
    record_blocks each entry is a dict holding select_hidden's pools and
    select_blocks' blocks, static token and readout ids.
    """
    captured = []
    original = lotus_module.efficient_forward
    loop_start = max(0, first_latent-1)
    loop_end = last_latent+1

    def hooked(*args, **kwargs):
        out = original(*args, **kwargs)
        value = select_hidden(out.last_hidden_state, first_latent=first_latent,
                              last_latent=last_latent, loop_start=loop_start,
                              loop_end=loop_end, c_thought=c_thought)
        if record_blocks:
            blocks, static, readout = select_blocks(
                out.last_hidden_state, out.logits, first_latent=first_latent,
                last_latent=last_latent, loop_start=loop_start, loop_end=loop_end,
                c_thought=c_thought)
            value = {"pools": value, "blocks": blocks, "static": static, "readout": readout}
        captured.append(value)
        # Upstream eval mode would retain full-vocabulary intermediate logits
        # for all 400 latent slots and every iteration. The neural recurrence
        # reads hidden_states, never those logits. Drop them before returning.
        # Avoid a slice view, which would retain the full underlying allocation.
        if suppress_logits:
            out.logits = out.last_hidden_state.new_zeros((*out.last_hidden_state.shape[:2], 1))
        return out

    lotus_module.efficient_forward = hooked
    try:
        yield captured
    finally:
        lotus_module.efficient_forward = original


def capture_one(model, lotus_module, tokenizer, question: str, *, loops: int, c_thought: int,
                suppress_logits: bool = True):
    """Use the trained model's real recurrent computations and preserve every stage."""
    import numpy as np
    import torch
    if loops < 1 or not 1 <= c_thought <= 256:
        raise ValueError("loops and c_thought must be positive")
    latent_tokens = loops * c_thought
    q_ids = tokenizer.encode(question, add_special_tokens=True)
    if not q_ids:
        raise ValueError("question tokenized to empty prefix")
    ids = (q_ids + [model.start_latent_id] + [model.latent_token_id]*latent_tokens
           + [model.end_latent_id])
    device = next(model.base_causallm.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    first = len(q_ids)+1
    last = first + latent_tokens-1
    with torch.inference_mode():
        with intercept_loop_states(lotus_module, first_latent=first,
                                   last_latent=last, c_thought=c_thought,
                                   suppress_logits=suppress_logits) as captured:
            # Zero target labels: no answer tokens or gold trace are consumed.
            # We call forward rather than generate; upstream generate uses
            # CUDA timing and text decoding and would retain logits by default.
            model.forward(input_ids, torch.ones_like(input_ids),
                          torch.full_like(input_ids, -100),
                          torch.arange(len(ids), device=device)[None],
                          n_looped_iters=loops)
    if len(captured) != loops + 1:
        raise AssertionError(f"expected initial + {loops} loops; got {len(captured)} calls")
    z = torch.stack(captured, dim=0).numpy()
    if z.ndim != 3 or z.shape[1] != 2 or not np.isfinite(z).all():
        raise AssertionError("invalid captured latent features")
    return z, {"prompt_tokens": len(q_ids), "latent_tokens": latent_tokens,
               "passes_captured":len(captured), "hidden_dim":z.shape[2],
               "input_sha256":hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()}


def latent_input(model, tokenizer, question: str, *, loops: int, c_thought: int, slots: int | None = None):
    """Token ids exactly as upstream eval.py builds them, plus latent span bounds.
    `slots` (H-A2d): that many latent tokens whatever the loop count, instead of loops*c_thought."""
    if loops < 1 or not 1 <= c_thought <= 256:
        raise ValueError("loops and c_thought must be positive")
    q_ids = tokenizer.encode(question, add_special_tokens=True)
    if not q_ids:
        raise ValueError("question tokenized to empty prefix")
    n = slots if slots else loops*c_thought
    ids = (q_ids + [model.start_latent_id] + [model.latent_token_id]*n
           + [model.end_latent_id])
    first = len(q_ids)+1
    return ids, first, first + n - 1, len(q_ids)


def capture_case(model, lotus_module, tokenizer, question: str, *, loops: int, c_thought: int,
                 max_new_tokens: int = 0, suppress_logits: bool = True, slots: int | None = None):
    """Every block of every pass, the static token, readout ids and optionally the answer.

    With max_new_tokens > 0 this calls upstream Lotus.generate (CUDA only: it
    calls torch.cuda.synchronize) inside the hook, so the captured states are
    the ones the model's own answer is decoded from. Otherwise it calls forward
    with all labels -100, as capture_one does.
    """
    import numpy as np
    import torch
    ids, first, last, n_prompt = latent_input(model, tokenizer, question,
                                              loops=loops, c_thought=c_thought, slots=slots)
    n_blocks = (last - first + 1) // c_thought
    device = next(model.base_causallm.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = None
    with torch.inference_mode():
        with intercept_loop_states(lotus_module, first_latent=first, last_latent=last,
                                   c_thought=c_thought, suppress_logits=suppress_logits,
                                   record_blocks=True) as captured:
            if max_new_tokens:
                out = model.generate(input_ids, torch.ones_like(input_ids),
                                     max_new_tokens=max_new_tokens, n_looped_iters=loops)
                generated = out[0, len(ids):].tolist()
            else:
                model.forward(input_ids, torch.ones_like(input_ids),
                              torch.full_like(input_ids, -100),
                              torch.arange(len(ids), device=device)[None],
                              n_looped_iters=loops)
    if len(captured) != loops + 1:
        raise AssertionError(f"expected initial + {loops} loops; got {len(captured)} calls")
    result = {"pools": torch.stack([c["pools"] for c in captured]).numpy(),
              "blocks": torch.stack([c["blocks"] for c in captured]).numpy(),
              "static": torch.stack([c["static"] for c in captured]).numpy(),
              "readout": torch.stack([c["readout"] for c in captured]).numpy()
              if all(c["readout"] is not None for c in captured) else None,
              "generated": generated}
    if result["blocks"].shape[:3] != (loops+1, n_blocks, 2) or not np.isfinite(result["blocks"]).all():
        raise AssertionError("invalid captured block features")
    return result, {"prompt_tokens": n_prompt, "latent_tokens": last - first + 1,
                    "passes_captured": len(captured), "hidden_dim": result["blocks"].shape[-1],
                    "sequence_tokens": len(ids),
                    "input_sha256": hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()}


def resolve_revision(model_id: str, revision: str | None, *, local_only: bool) -> str:
    if revision is not None:
        if not COMMIT_RE.fullmatch(revision):
            raise ValueError("--hf-revision must be a full 40-character commit SHA")
        return revision
    if local_only:
        raise ValueError("offline execution requires --hf-revision <40-char-SHA>")
    from huggingface_hub import HfApi
    sha = HfApi().model_info(model_id).sha
    if not sha or not COMMIT_RE.fullmatch(sha):
        raise RuntimeError("could not pin Hugging Face model revision")
    return sha


def load_model(repo: Path, *, model_id: str, revision: str,
               device: str, dtype: str, local_only: bool, c_thought: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    lm = import_lotus_module(repo)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                       local_files_only=local_only)
    # Upstream eval.py adds the three tokens, including on published checkpoint.
    for token in ("<|start-latent|>", "<|end-latent|>", "<|latent|>"):
        tok.add_tokens(token)
    start, end, lat = (tok.convert_tokens_to_ids(t) for t in
                        ("<|start-latent|>", "<|end-latent|>", "<|latent|>"))
    precision = torch.bfloat16 if dtype == "bf16" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                 torch_dtype=precision, local_files_only=local_only)
    if base.get_input_embeddings().weight.shape[0] < len(tok):
        raise ValueError("checkpoint lacks trained latent token rows; check model and tokenizer revision")
    model = lm.Lotus(base, latent_token_id=lat, start_latent_id=start,
                     end_latent_id=end, eos_token_id=tok.eos_token_id,
                     pad_token_id=tok.eos_token_id, c_thought=c_thought)
    model.to(device)
    model.eval()
    return model, lm, tok


def extract_roster(args):
    import numpy as np
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    validate_roster(roster)
    if args.loops != roster["parameters"]["steps"]:
        raise ValueError("--loops must equal frozen roster steps for stage/target alignment")
    src_hash = verify_upstream(args.upstream)
    revision = resolve_revision(args.model_id, args.hf_revision,
                                local_only=not args.allow_download)
    model, module, tokenizer = load_model(args.upstream, model_id=args.model_id,
                   revision=revision, device=args.device,
                   dtype=args.precision, local_only=not args.allow_download,
                   c_thought=args.c_thought)
    # Verify once that discarding unused intermediate logits does not change
    # the post-loop representation. Use a very short trace so full-vocab logits
    # fit comfortably even on consumer GPUs.
    sample_prompt=prompt_for(roster["cases"][0])
    small_full,_=capture_one(model,module,tokenizer,sample_prompt,loops=2,c_thought=2,
                             suppress_logits=False)
    small_light,_=capture_one(model,module,tokenizer,sample_prompt,loops=2,c_thought=2,
                              suppress_logits=True)
    import numpy as np
    if not np.allclose(small_full,small_light,rtol=1e-4,atol=1e-4):
        raise RuntimeError("logit suppression changes hidden states; abort extraction")
    # An optional strict cap supports tiny smoke runs. A scientific run
    # requires full frozen roster and must record any excluded rows.
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    cases = roster["cases"][:args.limit] if args.limit else roster["cases"]
    if not cases:
        raise ValueError("empty extraction")
    features, per_case = [], []
    begin=time.perf_counter()
    for i, row in enumerate(cases):
        t0=time.perf_counter()
        z, meta = capture_one(model, module, tokenizer, prompt_for(row),
                           loops=args.loops, c_thought=args.c_thought)
        if z.shape[0] != args.loops+1:
            raise RuntimeError("incomplete latent trace")
        features.append(z)
        per_case.append({"id":row["id"], "split":row["split"],
                          "seconds":round(time.perf_counter()-t0,4), **meta})
        if (i+1)%8==0:
            print(f"captured {i+1}/{len(cases)}", flush=True)
    feature_array = np.stack(features)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=feature_array, ids=np.array([r["id"] for r in cases]))
    record={"schema_version":"algernon-lotus-hidden-v1",
            "status":"actual_lotus_forward_observations",
            "source":{"repo":"https://github.com/yingfan-bot/lotus","commit":UPSTREAM_SHA,
                      "lotus_py_sha256":src_hash, "model_id":args.model_id,
                      "model_revision":revision},
            "roster_sha256":roster["roster_sha256"], "roster_file_sha256":file_sha256(args.roster),
            "cases":len(cases),"complete_roster":len(cases)==len(roster["cases"]),
            "feature_sha256":file_sha256(args.out), "device":args.device,
            "precision":args.precision,"loops":args.loops,"c_thought":args.c_thought,
            "feature_shape":list(feature_array.shape),"feature_pools":["last_latent","mean_last_block"],
            "logit_capture":"suppressed, no answer grading",
            "instrumentation_parity":{"short_trace_loops":2,"short_trace_c_thought":2,
                "max_abs_diff":float(np.max(np.abs(small_full-small_light))),"passed":True}, "total_seconds":round(time.perf_counter()-begin,3),
            "per_case":per_case}
    if args.device.startswith("cuda"):
        import torch
        record["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()
    meta_path=args.out.with_suffix(".manifest.json")
    meta_path.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(f"{len(cases)} LOTUS post-loop traces, {feature_array.shape}, sha {record['feature_sha256']}")
    print(f"manifest {meta_path}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream",type=Path,required=True,help="Clean checkout of LOTUS at pinned SHA")
    ap.add_argument("--roster",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--model-id",default=DEFAULT_MODEL)
    ap.add_argument("--hf-revision",default=None,help="Exact HF model revision; resolved automatically when download allowed")
    ap.add_argument("--allow-download",action="store_true",help="Allow downloading the 3B model")
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--precision",choices=("bf16","fp32"),default="bf16")
    ap.add_argument("--loops",type=int,default=16)
    ap.add_argument("--c-thought",type=int,default=25)
    ap.add_argument("--limit",type=int,default=None)
    extract_roster(ap.parse_args())


if __name__ == "__main__":
    main()
