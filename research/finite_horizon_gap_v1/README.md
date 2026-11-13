# FHC-01: finite-horizon compactification gap

Research started 2026-11-13. Sources: OpenAI Math [#094](https://github.com/openai/math/blob/main/lean/docs/094.md) and [#098](https://github.com/openai/math/blob/main/lean/docs/098.md).

## Main theorem

Conditional on the two published source results, a compact doubling Hilbert subset has N-point dense prefixes whose distortion-D embedding dimension must diverge with N, while l4 admits the upper bound exp(C_D sqrt(log N)).

[THEORY.md](THEORY.md) supplies the Gaussian transfer, finite obstruction compactness, robust epsilon-net obstruction, additive coarse embedding and exact linear shift-register observability rank min(T,m).

## Repeatable finite controls

Python 3.10+ and standard library:

    python -m unittest discover -s research/finite_horizon_gap_v1 -p 'test_*.py' -v
    python research/finite_horizon_gap_v1/reference.py --check --out /tmp/fhc01_control.json

The finite controls check Gaussian fourth moments, equilateral packing, distortion-transfer tolerance epsilon=delta/6, doubling cover arithmetic, rational finite l2->l4 maps and shift observability ranks.

The source papers' compact doubling witness is not reproduced by these finite checks, and no trained neural checkpoint is tested.

[PROTOCOL.md](PROTOCOL.md) defines the future-observation metric experiment, owned separately from RC-01, IC-01, IRC-02 and SUD-01.

The independently kernel-checked discrete bridge is in [MathCan PR #31](https://github.com/patrickdugan/MathCan/pull/31). It covers the integer one-sixth clearance and full-prefix recovery lemmas only.
