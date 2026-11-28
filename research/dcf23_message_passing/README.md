# DCF-23: learned message passing on the factor graph

DCF-22's token models learned almost nothing transferable about posteriors. DCF-23 runs a learned message-passing network on each query component's bipartite factor graph. Variable nodes carry BP beliefs; factor nodes carry the full likelihood table. There are two variants:
- `mp_bp`: outputs BP's parity logit plus a learned correction;
- `mp_pure`: uses no BP input.

- [`mp23.py`](mp23.py): stages `features` (BP and graph features, cached), `train`, `export`, `predict`, `report`. [`PROTOCOL.json`](PROTOCOL.json), `PRE_RUN_SHA256`, `PRE_PREDICT_SHA256`.
- `models23_weights.json`: frozen weights; the `.pt` files are gitignored. Also `train23.json`, `predictions23.json`, `results23.json`.
- [`tests/test_mp23.py`](tests/test_mp23.py): BP beliefs exact on trees, batching offsets, and the pure variant ignoring BP inputs.

Result: all three gates pass on 3/3 seeds. `mp_bp` lowers BP's posterior error and raises held-out decision utility from .370 to .410, about half of the gap to exact (.446), and the gain transfers to four of six unseen families. It costs about 100× BP's counted operations in model multiply-accumulates, and exact inference is cheaper still on these records, so this is an accuracy result, not a compute result. See [RESULTS.md](RESULTS.md).

## Reproduce

Python 3.12, NumPy 2.3.5, PyTorch 2.5.1 (CPU). Needs `../dcf16_decision_curriculum` and `../dcf22_curriculum_model/data`.

    python -m pytest tests
    python mp23.py features     # about 2 min
    python mp23.py train        # about 21 min on 4 threads
    python mp23.py export
    sha256sum -c PRE_PREDICT_SHA256
    python mp23.py predict
    python mp23.py report
