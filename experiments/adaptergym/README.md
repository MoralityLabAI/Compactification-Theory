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
  - `hillclimb`: one evaluation move per arm in a DataGoblin hill-climb lane, driven by the served model itself
    through the `data-goblin` GDK app. Arms (`label`, optional `adapter`, `alias` for the served model id, `args`
    after `/datagoblin-hillclimb`) run in order inside the step's lease, one server each. If no app run
    (`conversations`, default 2) reaches a decision and the step sets `"fallback": "script"`,
    `datagoblin.hillclimb.drive` records the move with the same service calls on the lane's eval contract. An arm
    is done when `<work>/results/<step>-<label>.json` exists; it is written only after the lane's ledger
    (`paths.datagoblin_state`) has a decision with a measured value made while that arm was served.
- `vram` on a step puts it in its own vramorama lease (`--guard-power 1800 --restart 20`). Steps without it run
  inline.
- `tags`: select subsets, e.g. `eval-bonsai` (inference on Bonsai only), `train-bonsai`,
  `eval-bonsai-adapter`, `eval-inky`.
- `alias` on a step (or a `hillclimb` arm) replaces the backend's `name` as the served model id, so an adapter
  arm's requests and scores carry their own model id.

## Preflight

`preflight` checks every interpreter, binary, model, adapter, overlay (inside WSL) and data file a plan names.
For venvs it reads `pyvenv.cfg`. It flags a base interpreter inside the Claude app's virtualized
`AppData\Local\Packages\Claude_*\LocalCache` and reports it as unusable outside the app. That is what stopped
the 2026-11-25 overnight run: the suite venv sat on uv's Python installed from inside the app.

Events: `<work>/logs/gym-events.jsonl`. Detached run log: `<work>/logs/gym-run-<time>.log` (one per launch). Server logs:
`<work>/logs/gym-<step>.log`.

## bdb_compare results (2026-11-27)

Same BDB data for every adapter (bonsai-gym `data/bdb`, 450 train rows). Greedy, thinking off (Inkling: `tml_v0`
effort 0). Eval split: 36 rows scored by `score_sft_replies.ts`. Bench: normalized gain over development seeds,
results in bitcoin-defi-bench (82799a2, fdcd86b).

| arm | split all | bonding | dlc | netting | bench bonding | bench dlc | bench netting |
|---|---|---|---|---|---|---|---|
| Bonsai 2.5 27B base | 0.229 | 0.688 | 0.000 | 0.000 | 0.652 | 0.475 | -0.628 |
| Bonsai 2.5 + 1-bit LoRA r8 (60 steps) | 0.667 | 1.000 | 1.000 | 0.000 | 0.959 | 0.996 | -0.628 |
| Bonsai 2.5 + fp LoRA r8 (60 steps) | 0.667 | 1.000 | 1.000 | 0.000 | 0.960 | 0.996 | -0.628 |
| Inkling 2 Mini base (squid) | 0.000 | 0.000 | 0.000 | 0.000 | -0.746 | -0.213 | -0.628 |
| Inkling 2 Mini + Tinker LoRA r16 folded (squid) | 0.646 | 1.000 | 0.938 | 0.000 | 0.879 | 0.989 | -0.628 |

- The 1-bit and fp rank-8 adapters on Bonsai score the same on both measures.
- Netting is -0.628 for every arm, including both bases: no plan was accepted.

## driver_adapter (Data Goblin self_adapter lane)

`plans/driver_adapter.json` closes the self_adapter loop of the DataGoblin `data-goblin` app: an adapter on the model
that drives the app, trained on the app's own hill-climb decisions and judged by the app's gate.

- Data (`<work>` = `Documents/Codex/datagoblin-driver`): DataGoblin `bench make --split train --seed 1 --count 2000`
  -> `bench export-sft` -> `curate` gives 2000 exact-oracle rows (`curated-train-s1/admitted.jsonl`: gate_verdict 814,
  proposal_legality 721, next_lane 465; 347 tokens per row on average, 631 at most). Val: train seed 2, 96 rows,
  minus 7 whose bench view also occurs in seed 1 (`val/val.jsonl`, 89 rows). The dev bench (split dev, seed 0,
  120 tasks) shares no view with either.
- `train-driver`: `b25_driver_fp_r8`, rank 8 on every projection of Ternary Bonsai 2.5 27B PTQ1_0, lr 2e-4, one
  epoch (250 steps x 8 rows), 12 GB lease with `cuda_memory_fraction` 0.72.
- `eval-driver`: backend `b25plain` (the PTQ1_0 file without the MTP head, plain decoding, so both arms decode the
  same way), ctx 32768, 9 GB lease. Arm `base`, then arm `fp-r8` with the adapter applied and served as
  `bonsai-2.5-27b-driver-fp-r8`. The ledger is `<work>/ledger`; app receipts are `<work>/ledger/receipts.jsonl`.
