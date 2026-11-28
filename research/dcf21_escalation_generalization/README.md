# DCF-21: does escalate-on-abstention generalise?

DCF-20B found that escalating BP's abstentions to exact inference beat both fixed operators in a hard mesh world at high compute prices. DCF-21 tests the same parameter-free policy on fresh instances of the ten DCF-14 graph families against DCF-14's policies. As a secondary, descriptive check it also scores the policy on the DCF-16 held-out records.

- [`escalate21.py`](escalate21.py): stages `run` (generate and verify instances with DCF-14's `run_instance`) and `report`. [`PROTOCOL.json`](PROTOCOL.json), `PRE_RUN_SHA256`.
- `receipt21.json.gz`: every instance's operators, posteriors, counted ops and verification. `results21.json`.
- [`tests/test_escalate21.py`](tests/test_escalate21.py).

Result: all three gates fail. The abstention flag has recall .727 and precision .166, and escalation loses to exact-else-mini-bucket in 365 of 438 conditions. Exact elimination is usually cheaper than BP in counted operations on these graphs, so probing with BP first doubles the work. See [RESULTS.md](RESULTS.md).

## Reproduce

Python 3.12, NumPy 2.3.5, scikit-learn 1.7.2 (imported by width14). Needs `../dcf14_width_routing` and `../dcf16_decision_curriculum`.

    python -m pytest tests
    sha256sum -c PRE_RUN_SHA256
    python escalate21.py run       # about 75 s with 6 workers
    python escalate21.py report
