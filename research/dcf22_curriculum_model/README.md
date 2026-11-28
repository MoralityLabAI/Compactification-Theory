# DCF-22: compact recurrent model on the DCF-16 curriculum

A compact model is trained on the DCF-16 curriculum's targets: verified posterior (calibration), operator choice under a budget, and log2 exact cost. It is scored once on the held-out split against the symbolic baseline and loopy BP. Two architectures are compared: a GRU over elimination-ordered factor tokens (3,175 params) and a pooled MLP over the same tokens (2,707). Three seeds each, CPU only.

- [`model22.py`](model22.py): stages `data` (fresh training-family records), `train`, `predict`, `report`. [`PROTOCOL.json`](PROTOCOL.json), `PRE_RUN_SHA256`, `PRE_PREDICT_SHA256`.
- Data: `data/train_large.jsonl.gz` (3,000 records, with `MANIFEST.json`). Models: `models22_weights.json`; the `.pt` files are gitignored, and their digests are in `PRE_PREDICT_SHA256`.
- Outputs: `train22.json`, `predictions22.json`, `preds/*.jsonl` (EvalMath-format prediction files), `results22.json`, `diagnostic22.json`.
- [`tests/test_model22.py`](tests/test_model22.py): encoding shapes, and no operator posterior in the inputs.

Result: all three gates fail. See [RESULTS.md](RESULTS.md).
- The learned selectors use a quarter of the operator ops but are not non-inferior in utility.
- The learned posteriors barely beat a constant in-distribution and lose to it on held-out families.
- Recurrence does not beat pooling.

The prediction files can be scored independently with EvalMath:

    python scripts/score_dcf16_decision.py <Compactification-Theory>/research/dcf16_decision_curriculum/data/heldout.jsonl.gz preds/gru_s1_selector.jsonl

## Reproduce

Python 3.12, NumPy 2.3.5, PyTorch 2.5.1 (CPU), scikit-learn 1.7.2.

    python -m pytest tests
    python model22.py data      # about 1 min
    python model22.py train     # about 4.5 min on 4 threads
    python model22.py predict
    python model22.py report
