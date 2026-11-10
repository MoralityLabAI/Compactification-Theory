# RRC-24: learning base-operator ordering under long q-adic collisions

- [Measured results and interpretation](RRC24_REPORT.md)
- [Exact valuation and prefix-collision mathematics](RRC24_THEORY.md)
- [Executed protocol and resource accounting](RRC24_PROTOCOL.md)
- [Source-bound benchmark](bench24.py), [algebra/interpreter](delay24.py), [portable model](model24.py), [trainer](train24.py), [tests](test_delay24.py)
- CPU replay: `python research/rrc_compactification/bench24.py --check`
- Frozen weights: `results/rrc24_{recurrent,feedforward,shuffled}.json`
- Deterministic receipt: `results/rrc24_pilot.json`; separately measured host timing: `results/rrc24_reference_profile.json`

**Main result:** A two-query 609-parameter GRU ranks the correct executable
base in 195/240 primary cases, above the static 123/240. An exact
short-prefix algebraic ranking reaches 217/240. All methods receive a
supplied exact test-time suffix solver that identifies the delayed factor.
Under four full base queries, symbolic recovery is 240/240.
