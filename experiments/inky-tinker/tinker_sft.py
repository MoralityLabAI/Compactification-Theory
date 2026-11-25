"""LoRA SFT of Inkling 2 Mini on Tinker from suite JSONL rows, with a held-out loss and a sampled eval.

  python tinker_sft.py --train train.jsonl --val val.jsonl --eval eval.jsonl --out <dir> --steps 15

Rows are the suite schema ({"id", "messages": [system?, user], "target": assistant text}). Each row becomes one
supervised example rendered by the cookbook's tml_v0 renderer (the model's own TMLv0 chat format) at a fixed
thinking effort (default 0.0, the lowest the format allows), loss on the final assistant message only.

Per step: one batch of --batch rows (deterministic shuffle by --seed), forward_backward + Adam step. Every
--save-every steps: save_state (resumable on Tinker) and log the tinker:// path. At the end: held-out loss before
and after, save_weights_for_sampler + save_weights_external, the adapter downloaded to <out>/adapter, and greedy
samples for the --eval rows from the base model and from the adapter (<out>/samples_{base,adapter}.jsonl as
{"id","reply"}), so any suite scorer can score them. Writes <out>/run.json with every path, loss and setting.
The API key is read by tinker_env.load_key() and never printed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinker_env import load_key  # noqa: E402

load_key()
import tinker  # noqa: E402
from tinker import types  # noqa: E402
from tinker_cookbook import renderers, tokenizer_utils  # noqa: E402
from tinker_cookbook.renderers.base import TrainOnWhat  # noqa: E402
from tinker_cookbook.supervised.common import datum_from_model_input_weights  # noqa: E402

FREE_MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
TOKENIZER_MODEL = "thinkingmachines/Inkling-2-Mini"


def read(p: str) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def to_messages(row: dict, with_target: bool) -> list[dict]:
    msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    if with_target:
        msgs.append({"role": "assistant", "content": row["target"]})
    return msgs


def weighted_nll(fwd_result, data) -> tuple[float, float]:
    """Summed NLL and token count over weight>0 targets of a forward(_backward) result."""
    nll, n = 0.0, 0.0
    for out, d in zip(fwd_result.loss_fn_outputs, data):
        lp = out["logprobs"].to_torch() if hasattr(out["logprobs"], "to_torch") else out["logprobs"]
        w = d.loss_fn_inputs["weights"].to_torch() if hasattr(d.loss_fn_inputs["weights"], "to_torch") else d.loss_fn_inputs["weights"]
        nll += float(-(lp * w).sum())
        n += float(w.sum())
    return nll, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--eval", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=FREE_MODEL)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--effort", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--base-max-tokens", type=int, default=0, help="token budget for the base arm (default --max-tokens)")
    ap.add_argument("--val-every", type=int, default=0, help="held-out loss every N steps (default: at each save)")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="inky-bdb")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = (out / "log.jsonl").open("a", encoding="utf-8")

    def log(rec):
        rec["t"] = time.time()
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps(rec), flush=True)

    tok = tokenizer_utils.get_tokenizer(TOKENIZER_MODEL)
    renderer = renderers.get_renderer("tml_v0", tok, model_name=TOKENIZER_MODEL)

    def datum(row):
        mi, w = renderer.build_supervised_example(to_messages(row, True), train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                                                  effort=a.effort)
        return datum_from_model_input_weights(mi, w, max_length=a.max_length)

    train, val = read(a.train), read(a.val)
    train_d = [datum(r) for r in train]
    val_d = [datum(r) for r in val]
    lens = [d.model_input.length for d in train_d]
    log({"event": "data", "train": len(train_d), "val": len(val_d), "max_len": max(lens), "mean_len": sum(lens) / len(lens)})

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=a.model, rank=a.rank, seed=a.seed, train_unembed=False)
    tc = tc.result() if hasattr(tc, "result") else tc
    log({"event": "client", "model": a.model, "rank": a.rank, "console": tc.get_console_url() if hasattr(tc, "get_console_url") else None})

    def val_loss():
        nll, n = 0.0, 0.0
        for i in range(0, len(val_d), 8):
            chunk = val_d[i:i + 8]
            res = tc.forward(chunk, loss_fn="cross_entropy").result()
            a_, b_ = weighted_nll(res, chunk)
            nll, n = nll + a_, n + b_
        return nll / n

    run = {"settings": vars(a), "saves": [], "val_loss": {}}
    run["val_loss"]["0"] = val_loss()
    log({"event": "val", "step": 0, "loss": run["val_loss"]["0"]})

    order, rng = [], random.Random(a.seed)
    for step in range(1, a.steps + 1):
        if len(order) < a.batch:
            idx = list(range(len(train_d)))
            rng.shuffle(idx)
            order += idx
        b, order = order[:a.batch], order[a.batch:]
        batch = [train_d[i] for i in b]
        t0 = time.time()
        fb = tc.forward_backward(batch, loss_fn="cross_entropy")
        op = tc.optim_step(types.AdamParams(learning_rate=a.lr))
        res = fb.result()
        op.result()
        nll, n = weighted_nll(res, batch)
        log({"event": "step", "step": step, "loss": nll / n, "target_tokens": n, "step_s": time.time() - t0})
        if step < a.steps and step % (a.val_every or a.save_every) == 0:
            run["val_loss"][str(step)] = val_loss()
            log({"event": "val", "step": step, "loss": run["val_loss"][str(step)]})
        if step % a.save_every == 0 or step == a.steps:
            p = tc.save_state(name=f"{a.name}-step{step}").result().path
            run["saves"].append({"step": step, "state": p})
            log({"event": "save_state", "step": step, "path": p})

    run["val_loss"][str(a.steps)] = val_loss()
    log({"event": "val", "step": a.steps, "loss": run["val_loss"][str(a.steps)]})
    sampler_path = tc.save_weights_for_sampler(name=f"{a.name}-final").result().path
    ext = tc.save_weights_external(name=f"{a.name}-export").result()
    run["sampler_weights"], run["external_weights"] = sampler_path, ext.path
    log({"event": "saved", "sampler": sampler_path, "external": ext.path})
    try:
        from tinker_cookbook import weights
        run["adapter_dir"] = weights.download(tinker_path=ext.path, output_dir=str(out / "adapter"))
        log({"event": "downloaded", "dir": run["adapter_dir"]})
    except Exception as e:  # keep the run's other outputs even if the download path differs
        run["download_error"] = f"{type(e).__name__}: {e}"
        log({"event": "download_error", "error": run["download_error"]})

    if a.eval:
        rows = read(a.eval)
        stop = renderer.get_stop_sequences()
        for arm, client, mt in (("adapter", sc.create_sampling_client(model_path=sampler_path), a.max_tokens),
                                ("base", sc.create_sampling_client(base_model=a.model), a.base_max_tokens or a.max_tokens)):
            params = types.SamplingParams(max_tokens=mt, temperature=0.0, stop=stop)
            futs = [(r, client.sample(prompt=renderer.build_generation_prompt(to_messages(r, False), effort=a.effort),
                                      num_samples=1, sampling_params=params)) for r in rows]
            with (out / f"samples_{arm}.jsonl").open("w", encoding="utf-8") as f:
                for r, fu in futs:
                    seq = fu.result().sequences[0]
                    msg, _term = renderer.parse_response(seq.tokens)
                    content = msg["content"] if isinstance(msg["content"], str) else renderers.format_content_as_string(msg["content"]) if hasattr(renderers, "format_content_as_string") else str(msg["content"])
                    f.write(json.dumps({"id": r["id"], "reply": content, "tokens": len(seq.tokens),
                                        "raw": tok.decode(seq.tokens)}) + "\n")
            log({"event": "sampled", "arm": arm, "rows": len(rows), "max_tokens": mt})
    (out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    log({"event": "done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
