# ARG-02 status

**2026-11-13: Extension N1 (Nanbeige4.2-3B, natively looped) executed.** See [REPORT_NANBEIGE.md](REPORT_NANBEIGE.md). Development chose 0 shots and d* = 8 loops; on the test rows loop 8 recovered 71/168 vs 44 for one loop. H1, H2b and H3a are met; H2a (open loop) is not, so the mechanism claim is not met; H3b is not applicable. Results in `results_nanbeige/`; run directory `C:\Users\patri\Documents\Codex\algernon\runs\arg02-nanbeige`; commit 65039a5 for every stage.

## Qwen3.5 run

**2026-11-11: executed. Negative result.** See [REPORT.md](REPORT.md).

- Development selection (once): 4 shots, block `[8,12)`, d* = 16.
- Test stage (once, 192 rows): loop(16) 67/168 future-correct at 12 checks vs 68 for the single pass; mechanism claim not met (H1, H2b, H3a, H3b, H3c, H4 not met; H2a met only because the open-loop ablation is destructive). Qwen3.5-2B single pass 79; complete certified class search 159.
- Amendment 1 (PROTOCOL.md): the donor-shuffle alignment defect was found by the row-spread diagnostic after the run; the corrected arm was rerun alone (66/168 at d = 16). Decisions are unchanged.
- Receipts: `results/arg02_receipt.json` (deterministic, regraded by `analyze replay` from `results/arg02_answer_tape.jsonl.gz`), `results/arg02_profile.json` (timings, compute, peak VRAM, run manifests), `results/arg02_traces.jsonl.gz` (per-case grades for 48 arms), `results/arg02_tables.md`.
- Run directory with every part, captured states and logs (outside Git): moved on 2026-11-13 from `C:\Users\patri\Documents\Codex\algernon\runs\arg02` to the private Hugging Face dataset `AlephFunk/algernon-runs`.

## Run log

| Stage | Commit | Wall time |
|---|---|---|
| CPU checkpoint check | 453c15d | 2026-11-10 |
| dev shots (2 x 132 rows) | deea6c6 | 13:42-13:47 |
| dev blocks (4 x 132 rows) | deea6c6 | 13:47-14:56 |
| test, 0.8B arms (192 rows) | deea6c6 | 14:56-19:15 |
| test, Qwen3.5-2B (192 rows) | 547f190 | 19:17-19:26 |
| Amendment 1 shuffle rerun (192 rows) | 5b7d2bc | 19:5x-20:22 |

The card was shared with another session's Lean job for part of the 0.8B test stage, so wall times and latencies include contention; MAC counts do not.
