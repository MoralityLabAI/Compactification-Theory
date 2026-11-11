# OIC-29 seven-phase search audit and constructive fixed-schedule codes

Finite, exact one-bit-substitution monitoring reference for unknown modular shift and initial phase on Z7. The adaptive query interval remains **15–20**, with resource-limited searches at intermediate horizons. The new fixed schedules are **fully verified achievable upper bounds**, with 22 queries using only three marker positions and 27 queries using only two.

[Report](OIC29_REPORT.md) · [Theory](OIC29_THEORY.md) · [OIC-30 protocol](OIC30_PROTOCOL.md).

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic29_fixed.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic29_fixed.py' -v
python research/observational_inwardness_v1/oic29_fixed.py --check
~~~

Search code used to discover candidate words is auxiliary; all published positive claims are independently verified by standard-library integer arithmetic.
