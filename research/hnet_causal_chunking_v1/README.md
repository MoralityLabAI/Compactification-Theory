# HC-01: learned byte chunking as intervention-preserving compactification

Additive research package dated 2026-10-29. Based on `MoralityLabAI/Compactification-Theory@8a79b9668d2fc4f2e2028b8914379aded33285f7`. Integrates Albert Gu and Cartesia's H-Net dynamic-chunking research with the existing intervention-preserving quotient work, finite-state resource bounds and prospective Storyworld transfer experiments.

**Question:** when a model jointly learns boundaries over raw bytes, which information can its resulting states discard while remaining sufficient for future observations and declared interventions?

- [THEORY.md](THEORY.md): full prefix-causal machine state; Myhill–Nerode / future-observation quotient; exact minimum code bits; factorization versus observational sufficiency; boundary-margin condition; error and resource ledgers; falsifiable hypotheses.
- [EXPERIMENTS.md](EXPERIMENTS.md): complete preflight requirements, source-byte-normalized scaling comparisons, BPE/byte/fixed-stride/frozen/shuffled/learned boundary controls, unseen symbolic grammar, Storyworld rule mutations and intervention adversaries.
- [reference.py](reference.py): deterministic finite-state oracle returning minimum behavioral classes and shortest distinguishing suffix for a defective encoding.
- [test_reference.py](test_reference.py): ten standard-library tests, including an independent fixed-point equivalence oracle over 100 seeded small finite machines.

## Run the exact reference

Python 3.10+ and standard library only:

```bash
python -m unittest discover -s research/hnet_causal_chunking_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/hnet_causal_chunking_v1 -p 'test_*.py' -v
python research/hnet_causal_chunking_v1/reference.py
```

In the included two-bit example, observing only the first bit initially needs one bit of abstract state when the operation alphabet is `{hold}`. Admitting `shift` exposes the previously hidden bit, raising the minimal exact state to two bits. The proposed one-bit encoding is falsified by the suffix `shift`. The reference reports the exact counterexample.

## Evidence status

Ten CPU tests ran locally in ordinary and optimized Python before publication. They establish the finite-state reference's behavior under its small test domain. The neural hypotheses, 2026 H-Net compute headline, real learned-router intervention invariants, RTX 5080 throughput and Storyworld transfer have not been reproduced by this package. The theory is handwritten and not Lean-checked. No source assets or seeds are represented as preregistered model experiments.

This directory adds a fresh hypothesis and baseline. It leaves E10's frozen selector protocol, the existing [IC-01](../../experiments/interventional-compactification/README.md) study, [QPC-T01](../qwenputer_v1/README.md), [RC-01](../recurrent_circuits_v1/README.md), prior adapter experiments and their results unchanged.

## Primary sources

- Hwang, Wang, Gu (2025): [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955) and [original H-Net code](https://github.com/goombalab/hnet).
- Gu and Wang (2025): [Cartesia hierarchical modeling](https://www.cartesia.ai/blog/hierarchical-modeling).
- Gu (2026-10-29): [new H-Net scaling announcement](https://x.com/_albertgu/status/2115502110578290900). Figures are author-reported pending independent replication.
