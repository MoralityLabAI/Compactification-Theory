# RRC-21: OOD-aware selective class ranking

- [Measured source-shift and deferral results](RRC21_REPORT.md)
- [Exact finite and conformal mathematics](RRC21_THEORY.md)
- [Data split, query budgets, replay and limitations](RRC21_PROTOCOL.md)
- [Known-only source generator](select21.py), [conformal selector](risk21.py)
- [Reproducible no-update calibration](train21.py), [full benchmark](bench21.py)
- [Frozen calibration](results/rrc21_calibration.json), [complete exact receipt](results/rrc21_pilot.json)

```
python -m unittest discover -s research/rrc_compactification -p 'test_select21.py' -v
python research/rrc_compactification/train21.py --out /tmp/rrc21_calibration_replay.json
python research/rrc_compactification/bench21.py --check
```

**Result:** 457/480 future-correct predictions from a policy permitted up to
33 exact checks on deferred cases, compared with 374/480 at a strict neural
12 checks and 480/480 from the class-complete symbolic solver. A policy
that preserves the 12-check budget falls to 324/480. The OOD detector's
empirical AUROC is 0.961 on selected source-family shifts; model weights
and the symbolic compiler are frozen.
