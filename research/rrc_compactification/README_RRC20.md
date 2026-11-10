# RRC-20: calibrated mixtures of certified graph-class rankings

[Report](RRC20_REPORT.md), [algebra and calibration theory](RRC20_THEORY.md),
[executed experimental protocol](RRC20_PROTOCOL.md), and
[source-locked receipt](results/rrc20_pilot.json).

Run from the repository root:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_family20.py' -v
python research/rrc_compactification/bench20.py --check
# Optional CPU PyTorch gate retraining:
python research/rrc_compactification/train20.py --out-dir /tmp/rrc20-retrain
```

**Measured result:** a 60-parameter source-conditioned gate combines
three frozen RRC-19 neural models with the certified class-size prior.
It predicts 301/384 unseen composed-source futures under twelve exact
checks; a uniform ensemble of the same experts predicts 318/384.
Calibration therefore fails the main OOD gate. Exact finite verifier
calls, model MACs, training-label privilege, and prior compiler cost
remain separately accounted for.
