# RRC-15: frozen topology-transfer evaluation

[Report](RRC15_REPORT.md) · [Protocol](RRC15_PROTOCOL.md) · [Algebra and proof boundaries](RRC15_THEORY.md) · [Executable test suite](test_topology15.py) · [Frozen source-bound receipt](results/rrc15_pilot.json)

The 7,353-parameter **already-trained** RRC-14 recurrent graph-edit policy is evaluated without retraining on two different public graph wirings. There are 144 instances whose observed coefficients cannot be explained by any original fork-join graph. With at most 12 distinct two-demo interpreter queries, the frozen recurrent model recovers 71/144. Its history-reset and matched feedforward controls recover 46/144 and 49/144, while the untrained residual-frontier controller recovers 74/144 at lower nominal compute. The recurrent result is **positive cross-wiring transfer** and a **failed learned-search superiority gate**.

The test uses the same fixed three-operator syntax, public algebra primitives, and interpreter query budget. The new topology is known to the interpreter; the weights receive no topology-ID input channel. Full 64-candidate search recovers 144/144 and is clearly labeled as above budget. Twenty out-of-grammar negatives and twelve paired late collisions audit finite verification and information limits. No GPU, training run or additional checkpoint is required.

Run from the repository root:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_topology15.py' -v
python research/rrc_compactification/bench15.py --check
```
