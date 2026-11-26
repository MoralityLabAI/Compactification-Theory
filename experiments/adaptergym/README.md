# adaptergym

One plan file drives adapter training and evals across model backends on the shared RTX 5080, one vramorama
lease per GPU step. It replaces the per-experiment shell drivers (`bonsai-gym/onebit_suite/run_suite.sh`,
`bdb_compare.sh`, `evals_only.sh`, `inky-tinker/inky_bdb_arm.sh`), which stay in place and write the same files.

```
cd experiments/adaptergym
python -m adaptergym preflight plans/bdb_compare.json
python -m adaptergym status    plans/bdb_compare.json --tags eval-bonsai
python -m adaptergym launch    plans/bdb_compare.json --tags eval-bonsai      # detached; survives the session
```

Use a stdlib-only interpreter that exists outside the Claude desktop app (the plan's `paths.gym_python`, the
system Python 3.12). `launch` runs `preflight` first and then starts `run` through `Win32_Process Create` with the
current environment, so vramorama attributes the leases to this session.

## Plan

- `backends`: named model servers.
  - `llama`: llama-server on Windows (Bonsai GGUFs), the suite's flags (q8 KV, reasoning off, greedy), optional
    MTP draft flags with a fallback to plain decoding if the server refuses them.
  - `squid`: squid5090 in WSL with `SQUID_OVERLAY` files (a LoRA folded by `inky-tinker/squid_overlay.py`),
    behind `inky-tinker/inky_proxy.py`, which renders each conversation with the Tinker `tml_v0` renderer
    the adapter was trained on.
- `steps`: run in order; each is skipped when its outputs exist, and `after` blocks a step until the steps it
  names are done.
  - `generate`: eval splits (`<work>/gen/<arm>/<family>.jsonl`) or sample pools through `onebit_suite/generate.py`.
    That script resumes from its output file and pauses while the GPU is power-capped.
  - `score`: `onebit_suite/score_all.py`. BDB uses bitcoin-defi-bench `score_sft_replies.ts`; SRB, math and logic
    use the suite scorers.
  - `bench`: bitcoin-defi-bench agent episodes (bonding, dlc, netting; development seeds).
  - `train_bonsai`: bonsai-gym `canary.py`, which resumes `checkpoint.pt`.
  - `tinker_sft`: `inky-tinker/tinker_sft.py` in WSL.
- `vram` on a step puts it in its own vramorama lease (`--guard-power 1800 --restart 20`). Steps without it run
  inline.
- `tags`: select subsets, e.g. `eval-bonsai` (inference on Bonsai only), `train-bonsai`,
  `eval-bonsai-adapter`, `eval-inky`.

## Preflight

`preflight` checks every interpreter, binary, model, adapter, overlay (inside WSL) and data file a plan names.
For venvs it reads `pyvenv.cfg`. It flags a base interpreter inside the Claude app's virtualized
`AppData\Local\Packages\Claude_*\LocalCache` and reports it as unusable outside the app. That is what stopped
the 2026-11-25 overnight run: the suite venv sat on uv's Python installed from inside the app.

Events: `<work>/logs/gym-events.jsonl`. Detached run log: `<work>/logs/gym-run-<time>.log` (one per launch). Server logs:
`<work>/logs/gym-<step>.log`.
