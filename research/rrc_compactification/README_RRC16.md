# RRC-16: multi-topology transfer to a new join

[Report](RRC16_REPORT.md) · [Protocol](RRC16_PROTOCOL.md) · [Exact theory](RRC16_THEORY.md) · [Source-bound receipt](results/rrc16_pilot.json)

**2026-11-10:** The recurrent graph-edit policy was fine-tuned on fork, cascade and additive join wirings, then tested under the **same 12-call interpreter budget** on a previously withheld coefficientwise Hadamard join. Its executable-program recovery was **52/120** on the held-out join. Its state-reset ablation reached **60/120**, matched feedforward **61/120**, fixed twelve-candidate search **62/120**, and the higher-budget exact 64-graph solver **120/120**. All accepted new-join programs predict the withheld degree-32 series exactly. The recurrent-superiority gate failed; results, negative controls and resource accounting are in the report.

Run CPU replay:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_topology16.py' -v
python research/rrc_compactification/bench16.py --check
```
