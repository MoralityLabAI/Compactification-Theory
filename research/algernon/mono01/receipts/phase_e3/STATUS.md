# MONO-01 E3 status (checkpoint 2026-11-10)

No E3 outcome exists. Pre-registration: `PREREG_E3.md` (committed before any run). Pipeline: `phase_e3.py`.

What ran (2026-11-04/05, shared 5080, chain of vramorama leases):
- 0.8B bf16, batch 64: CUDA out of memory in the first generation batch with another job holding 7.7 GB. Rerun at batch 32 finished the base calibration generations (`gen_calib.jsonl` for 96 of 144 sequences) and then hit CUDA out of memory again in the next batch; the process had committed about 26 GB at that point while holding 32 MB of RAM, so the cause is host commit or shared-GPU spill pressure, not proven.
- 9B NF4 smoke (`techwithsergiu/Qwen3.5-text-9B-bnb-4bit` at 220c703421073067939e7ad5cd733b5a85e07118, local copy `Documents\Codex\qwen35_9b_nf4`): generation worked, then `collect` raised `AttributeError: 'float' object has no attribute 'cpu'`. Cause found: the 96-token smoke generations contain no looping tokens, so the loop accumulators stayed Python floats.

Fixed in this checkpoint: `collect` now skips a layer with fewer than 32 loop or non-loop tokens and says so; if no layer is kept the run stops with a clear message; smoke mode adds a synthetic loop mask (plumbing check only, its numbers mean nothing). The batch-splitting retry on CUDA OOM was added earlier (`feaa7ed`).

Not fixed or not verified: the 0.8B out-of-memory cause; the 9B path after `collect` has never run. No lease or job of mine is running. Before relaunching: check free commit (`vramorama sessions`), keep one job in the queue, and run the 9B smoke before the 9B full run.
