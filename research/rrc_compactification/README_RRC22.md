# RRC-22: physical cost of selective symbolic deferral

[Measured CPU results and failed break-even](RRC22_REPORT.md) | [Math and bit-width accounting](RRC22_THEORY.md) | [Reproduction protocol](RRC22_PROTOCOL.md)

This experiment reuses the RRC-21 test roster and frozen neural models. On an AMD EPYC Python 3.13 CPU, complete class search reaches 480/480 positive future predictions in 0.574 seconds over three 508-case passes, while selective full deferral reaches 457/480 in 5.583 seconds over the same passes. The selective policy saves 607 full-degree positive-case graph queries; its extra routing/model costs exceed the saved verifier time. The source-hashed semantic receipt is deterministic; wall clock, RSS and Python traced memory are host-specific.

Reproduce from repo root:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_meter22.py' -v
python research/rrc_compactification/bench22.py --check
python research/rrc_compactification/bench22.py --smoke
python research/rrc_compactification/bench22.py --profile --rounds 3 --out /tmp/rrc22_profile.json
```
