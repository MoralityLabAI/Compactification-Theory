# RRC-18: exact semantic certificate plus compact class proposer

**Status: CPU experiment executed and source-bound.** The new
[`certify18.py`](certify18.py) normalizes all 256 graph/topology programs
as polynomials in the first twelve coefficients of an arbitrary formal
unit series. It confirms RRC-17's random-probe classes *exactly* through
degree 12: fork 25, cascade 24, additive 33, Hadamard 24.

A 6,240-parameter GRU reads two 13-coefficient input/output demonstrations
and proposes verified quotient representatives. On 240 fresh signed-series
tests it recovers 233 programs using up to twelve exact checks, compared
with 205 for a matched feedforward model and 155 for the descending-class-
size symbolic search. On 160 product-series source-shift tests, the GRU
recovers 92 and the class-size prior recovers 110. The stronger cross-source
learned-transfer criterion fails.

- [Proof and scope](RRC18_THEORY.md)
- [Frozen protocol](RRC18_PROTOCOL.md)
- [Report, metrics and work ledger](RRC18_REPORT.md)
- [Exact source-bound receipt](results/rrc18_pilot.json)
- [Unit tests](test_certify18.py)

From repository root:

```bash
python -m unittest discover -s research/rrc_compactification -p test_certify18.py -v
python research/rrc_compactification/bench18.py --check
```

Training-only PyTorch is optional; frozen replay uses Python's standard
library. The exact certificate is finite-degree, and all candidate
operations, formal-series transforms and test-time verification are
supplied externally to the model.
