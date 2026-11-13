# ARG-02: recurrent-depth evaluation of Qwen3.5 on the Algernon algebraic gym

ARG-02 runs a pretrained Qwen checkpoint on the ARG-01 held-out roster with explicit recurrent depth, exact EvalMath verification of every proposal, and the controls needed to tell recurrence apart from extra compute. The registered design, arms and decision rules are in [PROTOCOL.md](PROTOCOL.md); results are in [REPORT.md](REPORT.md) (Qwen3.5, negative: an untrained loop did not increase verified program recovery) and [REPORT_NANBEIGE.md](REPORT_NANBEIGE.md) (Nanbeige4.2-3B, natively looped: a real depth gain, mechanism claim not met) and the run log in [STATUS.md](STATUS.md).

## Layout

| File | Role |
|---|---|
| `config/arg02_config.json` | depths, candidate blocks, shot options, budgets, perturbation, controller, decision rules |
| `config/checkpoint_lock.json` | model revisions, file SHA-256, text configs, checkpoint inventory, software |
| `splits.py` | fixed test/calibration rosters from EvalMath, seeded dev and shot rosters, contamination audit |
| `prompts.py` | prompt text from public records only |
| `recurrence.py` | block-loop runner over Qwen3.5 decoder layers: loop, open loop, donor shuffle, site patching, perturbation, per-step capture |
| `runner.py` | GPU stages (dev shots, dev blocks, test, test ff), resumable parts, power watch |
| `scoring.py` | 64-program ranking, greedy decode, EvalMath search/grade, controller regrade |
| `analyze.py` | export, development selection, grading, statistics, probes, receipts, tape replay |
| `stats.py`, `flops.py`, `probe.py`, `symbolic.py` | paired statistics, analytic MACs, numpy probes, ARG-01 baselines |
| `checkpoint_check.py` | CPU pre-run check on the real weights |
| `smoke.py` | the whole pipeline on a tiny random Qwen3.5 on CPU |
| `run_arg02.ps1` | the GPU chain under vramorama |
| `tables.py` | Markdown tables from the receipt and profile |
| `nanbeige.py` | Nanbeige4.2-3B loader and native full-stack loop (profile `nanbeige`, transformers < 5) |

## Run

From the Compactification-Theory root, with EvalMath checked out at the commit in `../evalmath_pin.json`:

```sh
export PYTHONPATH=.:../EvalMath/src
python -m unittest research.algernon.algebraic_reasoning_gym.arg02.test_arg02          # numpy only
python -m unittest research.algernon.algebraic_reasoning_gym.arg02.test_arg02_torch    # torch + transformers>=5.17
python -m research.algernon.algebraic_reasoning_gym.arg02.smoke --run-dir /tmp/arg02-smoke
```

GPU run (Windows, RTX 5080, vramorama 0.15):

```powershell
powershell -ExecutionPolicy Bypass -File research\algernon\algebraic_reasoning_gym\arg02\run_arg02.ps1 -RunDir C:\Users\patri\Documents\Codex\algernon\runs\arg02
```

The chain is export -> `runner dev --stage shots` -> `analyze select --stage shots` -> `runner dev --stage blocks` -> `analyze select --stage blocks` (writes `selection.json`) -> `runner test` -> `runner test --model ff` -> `analyze report --out arg02/results` -> `analyze replay`. Analytic compute from `flops.py` at the measured prompt lengths: about 3 PFLOP for the shots stage, 14 PFLOP (0 shots) or 47 PFLOP (4 shots) for the block stage, 40 or 133 PFLOP for the 0.8B test arms and 3 or 9 PFLOP for the 2B control.

Replay the committed results on CPU, with no model:

```sh
python -m research.algernon.algebraic_reasoning_gym.arg02.analyze replay \
  --tape research/algernon/algebraic_reasoning_gym/arg02/results/arg02_answer_tape.jsonl.gz \
  --check research/algernon/algebraic_reasoning_gym/arg02/results/arg02_receipt.json
```

## Boundaries

The EvalMath oracle and its tests are imported from the pinned checkout and not modified. ARG-01's `gym.py`, `external.py`, `test_gym.py` and receipts are unchanged; `symbolic.py` calls `gym.score_case` as is, and the report re-scores ARG-02's top-1 answers through ARG-01's `external.py` as a cross-check.
