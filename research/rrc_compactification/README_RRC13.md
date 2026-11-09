# RRC-13 index: causal feedback for program construction

[Theory](RRC13_THEORY.md) · [Protocol](RRC13_PROTOCOL.md) ·
[Measured CPU result](RRC13_REPORT.md) · [Source-bound receipt](results/rrc13_pilot.json)

A 6,148-parameter recurrent controller constructs a three-operation
formal-series graph by choosing left, right and post nodes. It receives
two public exact partial-execution residual vectors before final
verification. The equal-sized passive control receives the same
observed demonstration data and supervised graph labels with its
feedback bus fixed to zero. A second ablation disables feedback in
the trained active controller itself.

**Result:** familiar-class accepted programs 25/64 active, 12/64
passive, 8/64 active with feedback off. On the 168 unseen S/R
compositions and product-source transfers every trained direct
controller gets 0/168. An untrained greedy controller with 12
partial queries recovers 33/168, the 64-preview heuristic recovers
165/168, and the complete exact solver recovers 168/168.

Reproduce with Python standard library:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_graph13.py' -v
python research/rrc_compactification/bench13.py --check
```

Training requires PyTorch CPU and was separately reproduced
byte-for-byte using `train13.py`. The decoder's interpreter is public;
no infinite graph-identity or general neural operator discovery claim is
made from finite certificates.
