# ARG-02 status

**2026-11-10: implemented, registered and checked on CPU. The GPU stages have not run**: the owner put the GPU on hold for this work on 2026-11-10, and other jobs held 7.5-14 GB of the shared card that day. There are no Qwen scores on the development or test rows yet, so no hypothesis in [PROTOCOL.md](PROTOCOL.md) has an outcome.

## Done and verified

- Splits: 192 test rows identical to ARG-01's `external.public_records()`, 36 calibration, 96 dev, 4 shots; zero signature overlap between any non-test roster and test (`results/arg02_splits_audit.json`).
- Checkpoints: Qwen3.5-0.8B-Base and Qwen3.5-2B-Base snapshot files match `config/checkpoint_lock.json`.
- Real-weight CPU check (`results/arg02_checkpoint_check.json`): the 0.8B loads as `Qwen3_5ForCausalLM` with 752,393,024 text parameters; the block-loop runner's depth-1 readout equals the HF forward exactly (max |log-prob difference| 0.0) on two development prompts; one extra pass of block `[12,16)` moves the block output by 58% of its norm. Prompt lengths: 321-399 tokens with 0 shots, 1084-1194 with 4 shots.
- Tests: 26 torch-free tests (`test_arg02.py`) and 10 torch tests on a tiny random Qwen3.5 (`test_arg02_torch.py`) pass locally: HF parity at depth 1 for three block placements, a readout inside a trajectory equals a standalone run at that depth, open(1) equals loop(1), a self-donor shuffle equals the loop, a self-patch and a zero kick change nothing, every runner stage runs with `oracle.roster` patched to raise, a restart reuses every part, and the full export -> select -> test -> report -> replay chain completes.
- ARG-01: `gym --check` still verifies its receipt in this branch; `gym.py`, `external.py`, `test_gym.py` and the ARG-01 receipts are unchanged.

## Next command

From the repository root on the RTX 5080 machine, once the GPU is released:

```powershell
powershell -ExecutionPolicy Bypass -File research\algernon\algebraic_reasoning_gym\arg02\run_arg02.ps1 -RunDir C:\Users\patri\Documents\Codex\algernon\runs\arg02
```

The run directory already holds the exported public rows (`export_manifest.json` digests match `results/arg02_export_manifest.json`). The chain writes `selection.json` from development data before the test stage, runs the test stage once, and writes `results/arg02_receipt.json`, `arg02_profile.json`, `arg02_traces.jsonl.gz` and `arg02_answer_tape.jsonl.gz`. Peak VRAM has not been measured on the GPU; the runner caps itself at 45% of the card (`cuda_memory_fraction`) inside an 8 GB lease.
