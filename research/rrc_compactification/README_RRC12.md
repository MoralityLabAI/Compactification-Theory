# RRC-12: direct neural graph proposal from demonstrations

A recurrent and parameter-matched feedforward model observe two raw integer
formal-series input/output examples and propose one complete executable
`post(left(F)*right(F))` operator graph. The interpreter verifies the proposal
only **after** model inference. Eight unseen S+R graph-composition classes
are excluded from training, and a new periodic-part source family tests transfer.

In the first fixed-seed CPU pilot, the one-scan GRU verifies **17/68** familiar
cases and **0/160** unseen combinations; feedforward verifies **12/68** and
**0/160**. Exhaustive exact search verifies **68/68** and **160/160**;
the untrained public-execution preview solver verifies **67/68** and **159/160**.

See [protocol](RRC12_PROTOCOL.md), [mathematics](RRC12_THEORY.md),
[measured report](RRC12_REPORT.md), [portable model](model12.py),
[graph data](graph12.py), [train script](train12.py),
[exact replay](bench12.py), [tests](test_graph12.py),
[frozen receipt](results/rrc12_pilot.json), and
[lightweight CI](../../.github/workflows/rrc-12.yml).

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_graph12.py' -v
python research/rrc_compactification/bench12.py --check
```
