# RRC-17: graph semantic quotients with charged offline compilation

[Executed CPU report](RRC17_REPORT.md) · [Mathematical derivation and conditional collision bound](RRC17_THEORY.md) · [Source-bound protocol](RRC17_PROTOCOL.md)

A deterministic four-probe classifier groups 64 candidate three-node graphs into 24–33 behavior classes depending on topology, using the *public exact algebra interpreter*. The twelve most common classes provide an untrained 12-query search baseline for frozen RRC-16 recurrent and feedforward policies. All candidate acceptances receive independent exact coefficient certificates, and all source/checkpoint and offline compilation costs are recorded. The experiment includes three additional independent-seed Hadamard evaluations.

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_quotient17.py' -v
python research/rrc_compactification/bench17.py --check
```

RRC-17 trains no weights and makes no claim of universal equivalence based on randomized probes. The observed held-out uplift is an interpreter-assisted **search baseline** for future learned algebraic graph discovery.
