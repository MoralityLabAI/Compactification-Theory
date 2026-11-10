# RRC-19: family-balanced certified graph ranking

- [Exploratory experiment and results](RRC19_REPORT.md)
- [Finite mathematics and objectives](RRC19_THEORY.md)
- [Executed split, controls and replay protocol](RRC19_PROTOCOL.md)
- Standard-library benchmark: `python research/rrc_compactification/bench19.py --check`
- Unit tests: `python -m unittest discover -s research/rrc_compactification -p 'test_family19.py' -v`
- Optional deterministic PyTorch CPU retraining: `python research/rrc_compactification/train19.py --out-dir /tmp/rrc19-retrain`

**Measured:** Balanced GRU is 134/140 on in-training-family product-series tests versus 119/140 for the same-size skewed GRU, but trails the skewed GRU 215/280 versus 226/280 on two *new* source-generator families. Exactly verified class search provides the program certificates. All neural proposals use a maximum of twelve complete two-demo checks and no candidate-execution previews.
