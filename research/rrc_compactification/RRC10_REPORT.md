# RRC-10: few-shot hidden-operator identification, CPU result

**Executed:** 2026-11-09. Exact rational two-coefficient fitting on 128 training episodes; 24 development and 128 held-out episodes; source-bound standard-library inference and replay.

The two-weight recurrent program, after rational least-squares training and the registered integer snap, is `(1,-1)`. The two-weight raw-local control fits fractions `4866995/316947332` and `-1798271/158473666`, which snap to `(0,0)` and fail demonstration verification. The same exact causal law used in the independent triangular division baseline recovers every tested hidden multiplier whose support is within the demonstration horizon.

| Method | Held-out finite-demo certificates /128 | Correct futures /128 | Collision false accepts | Conflict abstentions |
|---|---:|---:|---:|---:|
| Learned 2-weight causal recurrence | 112 | 104 | 8 | 16/16 |
| Exact symbolic inversion | 112 | 104 | 8 | 16/16 |
| Two-weight raw-local feedforward | 0 | 0 | 0 | 16/16 |
| No-demonstration control | 0 | 0 | 0 | 16/16 |

All **96 new-family** examples succeed: 48 held-out multiplier examples plus 48 examples with known wrapper composition. The remaining held-out split has 16 late-collision examples; both members receive the same verifiable prefix rule, and exactly one per pair generalizes. Sixteen inconsistent second demonstrations are rejected. The 24 development examples are excluded from held-out numbers.

The constrained result explains how correct operator structure, supplied as a convolution state, makes few-shot learning feasible. A direct rational symbolic solve produces the same predictions, so there is **no measured computational benefit from the learned weights**. The result also does not test whether a recurrent net can discover an unknown primitive family, since multiplication by a hidden polynomial is built into the public architecture.

## Reproduction

From repository root, Python 3.10+ standard library:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_primitive10.py' -v
python research/rrc_compactification/train10.py --out-dir /tmp/rrc10_retrain
python research/rrc_compactification/bench10.py --check
```

Training uses deterministic integer matrices and `fractions.Fraction`; saved checkpoint equality may be checked without numerical-tolerance claims. The source-bound result is `results/rrc10_pilot.json` and exact fitted coefficients are in `results/rrc10_{recurrent,local}.json`. This run has no GPU, PyTorch, neural-network training, external inference, or novel mathematical identity.

**Next gate (RRC-11):** give a learner an *unknown operator graph* selected from nonlinear arithmetic classes, only a bounded set of I/O examples, and a strictly separated symbol interpreter. Charge program synthesis and oracle access symmetrically. Preserve identical-prefix and observational-equivalence tests.
