# DCF-24: frozen message-passing correction in the high-width regime

Tests whether DCF-23's learned correction to BP, used frozen, pays off where exact inference is expensive. The instances are fresh, wide graphs (dense, large grids, large sparse; width 5–19), each verified by two independent exact paths.

- [`hw24.py`](hw24.py): `run` (generate and verify records, compute BP and graph features), `report` (score frozen DCF-23 `mp_bp` seeds, evaluate gates and capped policies). [`PROTOCOL.json`](PROTOCOL.json), `PRE_RUN_SHA256`.
- `records24.jsonl.gz`, `features24.json.gz`, `results24.json`.

Result: all three gates fail. In these wide graphs BP is already nearly exact (MSE .0036; EU .596 vs .594 for the exact solver-model posterior), so the correction has nothing to fix. The model is cheaper than exact on only 9 of 88 instances, and the learned fallback costs about 90× the BP fallback for a non-significant gain. See [RESULTS.md](RESULTS.md).

## Reproduce

Python 3.12, NumPy 2.3.5, PyTorch 2.5.1. Needs `../dcf23_message_passing/models/*.pt`, which are gitignored. Rebuild them from `../dcf23_message_passing/models23_weights.json`, or rerun DCF-23's `train`; digests are in that directory's `PRE_PREDICT_SHA256`.

    sha256sum -c PRE_RUN_SHA256
    python hw24.py run       # about 25 s with 6 workers
    python hw24.py report
