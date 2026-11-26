# Handoff: training Inkling 2 Mini ("Inky") LoRAs on the Tinker free tier

For agents picking up adapter training on Inky. Tinker (Thinking Machines) trains a LoRA on
`thinkingmachines/Inkling-2-Mini` on their GPUs at no cost on the free tier, so a run takes no local GPU at all.
Evaluating locally does use the GPU, through llama.cpp or squid5090 (see the last section). One finished example: the BDB adapter
(450 rows, 168 steps, 42 min wall clock). It scores 0.646 on the 36-row BDB eval split against 0.000–0.201 for the base model, and
the same 0.646 when folded into squid5090 on the RTX 5080. Results and method: [README.md](README.md).

## What you need

- A Tinker API key in a one-line file. `tinker_env.load_key()` reads `TINKER_KEY_FILE` (default
  `/mnt/c/Users/patri/OneDrive/Desktop/tinker.txt.txt`) into `TINKER_API_KEY` without printing it. Never print, log,
  copy or commit the key. If a run needs a different account, set `TINKER_KEY_FILE` to that account's file.
- WSL `Ubuntu-24.04`, run as root (`wsl.exe -d Ubuntu-24.04 -u root --exec ...`). `tml-renderers` (the Inkling chat
  format) has Linux/macOS wheels only, so everything that touches Tinker runs in WSL.
- The venv at `/root/inky-tinker/venv` already exists. To rebuild it elsewhere:
  `python3.12 -m venv venv && venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-wsl.txt`.
  [wpy.sh](wpy.sh) runs a script in it (`INKY_VENV` overrides the path).
- **Disk:** D: is full (about 1 GB free) and holds the WSL disk image. Never pull the 19.5 GB Inky checkpoint into WSL
  or a Hugging Face cache on D:, because that filled D: on 2026-11-25 and WSL stopped booting. Check `df -h /mnt/d /` before any
  download. Work directories go on C: (`C:\Users\patri\Documents\Codex\inky-tinker\<run>`).

## Data format

JSONL, one row per example, the bonsai-gym suite schema:

```json
{"id": "bonding/fast/5000", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "target": "the assistant reply to learn"}
```

Extra fields are ignored. Each row is rendered with the cookbook's `tml_v0` renderer at thinking effort 0 (the lowest
the format allows; there is no off switch), with loss on the final assistant message only. You need `train`, `val` (held-out
loss) and optionally `eval` (greedy samples from base and adapter written out for any scorer). Existing sets live under
`C:\Users\patri\Documents\Codex\onebit-suite-25\data\<family>\{train,val,eval}.jsonl` (bdb, srb, math, logic). Never
train on an `eval.jsonl`.

## Train

```bash
I=/mnt/c/Users/patri/OneDrive/Documents/GitHub/adapter-bits/experiments/inky-tinker
D=/mnt/c/Users/patri/Documents/Codex/onebit-suite-25/data/bdb
wsl.exe -d Ubuntu-24.04 -u root --exec bash $I/wpy.sh -u $I/tinker_sft.py \
  --train $D/train.jsonl --val $D/val.jsonl --eval $D/eval.jsonl \
  --out /mnt/c/Users/patri/Documents/Codex/inky-tinker/<run> --name inky-<task>-<run> \
  --rank 16 --lr 2e-4 --batch 8 --steps 168 --save-every 28 --val-every 14 \
  --max-tokens 1536 --base-max-tokens 6144
```

From Git Bash, prefix the command with `MSYS_NO_PATHCONV=1` or it rewrites the `/mnt/...` paths. For runs longer than a
session, start it detached (Win32_Process Create or `start /b`) with stdout to `<run>/stdout.log`.

- These are the BDB full-run settings. Steps = epochs × rows / batch (3 epochs × 450 / 8 ≈ 168). Each step took about 13 s on the free
  tier; sampling the eval rows at the end takes a few minutes.
- `--model` defaults to the free tier string `thinkingmachines/Inkling-2-Mini:peft:65536:free` (max sequence 65,536).
  `--max-length 8192` drops longer rows; the log's `data` event reports `max_len`.
- `--base-max-tokens 6144`: the base model thinks at length even at effort 0, so at 1,536 tokens it scores 0 by
  truncation. Give the base arm a bigger budget, or the comparison measures truncation rather than skill.
- `--save-every` writes a resumable Tinker state and logs its `tinker://` path in `log.jsonl` and `run.json`.

Outputs in `--out`:
- `log.jsonl`: train loss per step, val loss, saves, and the Tinker console URL.
- `run.json`: settings, val-loss curve, every `tinker://` path.
- `adapter/`: the downloaded Tinker adapter.
- `samples_{base,adapter}.jsonl`: `{"id", "reply"}` per eval row.

## Score

BDB: `node --experimental-strip-types scripts/score_sft_replies.ts` in `bitcoin-defi-bench` on the samples files.
The full run's `scored_*.jsonl` show the format. SRB, math and logic: the bonsai-gym suite scorers (`onebit_suite/score_all.py`).
Report the base arm next to the adapter arm.

## Run the adapter locally (optional, needs the GPU through vramorama)

1. PEFT form (local, no download of the base model): build the header-only stub once, then convert.
   - `wpy.sh stub_shards.py --out /root/inky-tinker/inkling-2-mini-stub`. The stub already exists, a few MB on disk.
   - `wpy.sh to_peft.py --run <run dir>` writes `<run>/adapter_peft`. Rerunning it on the full run reproduced the committed PEFT
     adapter byte for byte (sha256 `26e684df…`).
2. Pick a runtime:
   - **llama.cpp** (upstream b12611 or later; the PrismML fork has no inkling2):
     - Convert: `python tinker_lora_to_gguf.py --peft <run>/adapter_peft --out <run>/lora-f16.gguf` (Windows Python with
       llama.cpp's `gguf-py`).
     - Serve `C:\Users\patri\models\inkling-2-mini\Inkling-2-Mini-UD-Q4_K_M.gguf` with `--lora`.
   - **squid5090** (the 5080 Inkling engine, about 105 tok/s; it has no runtime LoRA, so the adapter is folded into the weights):
     - Fold: `python squid_overlay.py --peft <run>/adapter_peft --ckpt <Inkling-2-Mini safetensors dir> --out-dir <overlay dir>`
       writes `attn.safetensors` and `experts.safetensors` (11.4 GB on C:).
     - Serve with `SQUID_OVERLAY=<dir>/attn.safetensors,<dir>/experts.safetensors`, using the binary
       `/root/squid5080-overlay/bin/squid5090` (local commit 149a746).
     - Requests go through [inky_proxy.py](inky_proxy.py), so chat requests are rendered with the same `tml_v0` prompt the
       adapter was trained on.
   - **adaptergym** ([../adaptergym](../adaptergym/README.md)) runs the squid bench as a plan step (`kind: bench`,
     `backend: inky`, `overlays: [...]`) inside a vramorama lease. One GPU job per agent at a time, queued through
     vramorama.

## Known limits

- **Netting:** 0 for every adapter so far (Inky and Bonsai alike). The model learns the `settle` format but fills
  transfers with placeholder amounts that do not balance. Training on more rows of the same targets has not moved it; it needs
  targets or a loss that reach the arithmetic.
- **Free-tier quota:** one 168-step run used under an hour. The free tier's limits are set on Tinker's side and have not
  been measured here. If a call fails with a quota or rate error, report it and stop; don't retry in a loop.
- **HTTPS:** Norton re-signs HTTPS on this machine. `tinker_env` injects `truststore` so Python trusts the OS store; keep
  `truststore` in the venv.
