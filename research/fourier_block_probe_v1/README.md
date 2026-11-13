# FPB-01: Fourier-blind coordinated modes

Started November 13, 2026. The mathematics pairs
[OpenAI #132](https://github.com/openai/math/blob/main/lean/docs/132.md)
(superquadratic block sensitivity) and
[OpenAI #192](https://github.com/openai/math/blob/main/lean/docs/192.md)
(unbounded singleton Fourier sum relative to square-root degree).

[Theorem and proofs](THEORY.md) |
[exact finite reference](reference.py) |
[heldout Fourier benchmark](benchmark.py) |
[future neural protocol](PROTOCOL.md)

## New conditional theorem

For every C,K>0, conditional on source results #132/#192,
a selector-mode Boolean function F exists with

    sum_i Fourier_1(F,i) > C sqrt(degree(F)),

yet an embedded hidden mode G has zero first-order Fourier coefficients
and block_sensitivity(G,0)/sensitivity(G)^2 > K.

The key parity lift G(x,y)=(-1)^h(x xor y) has exact Fourier support
only on matched x/y coordinate sets, sensitivity 2s(h), and preserves
every disjoint zero-sensitive block of h. Surface Fourier salience
comes from a separate mode selected using result #192.

This theorem is MODE-CONDITIONAL. It does not assert a
superquadratic global ratio for the full selector F.

## Reproduction

Python 3.10+ standard library, from repository root:

    python -m unittest discover -s research/fourier_block_probe_v1 -p 'test_*.py' -v
    python research/fourier_block_probe_v1/reference.py --check --out /tmp/fpb01_exact.json
    python research/fourier_block_probe_v1/benchmark.py --out /tmp/fpb01_pilot.json

The finite source uses a 16-bit Rubinstein-style OR over four
four-bit blocks, augmented with one direct sentinel bit. It
provides nine pairwise disjoint sensitive blocks at zero and
exact sensitivity five, with no Fourier singleton signal after
a 34-bit parity lift.

The sentinel's matched-pair Fourier correlation is exactly
2401/4096, while every raw input singleton correlation is zero.
The benchmark fits first-order Fourier regression coefficients
on 4,096 training input pairs, then evaluates 4,096 independent
test pairs. It compares bias-only, raw 34 singletons, one paired
sentinel, all paired coordinates and paired fourth-order features.
Every result retains source/data SHA-256 and records
population-proof versus heldout-measurement status.

The finite Rubinstein-style example is a **diagnostic control**;
it is not the asymptotic superquadratic construction from #132,
and our finite five-bit surface majority does not reproduce
#192's unbounded-ratio witness.

## Applications

LoopedTransformers / SEM-ECHO, sparse autoencoder interpretability
and compact recurrent reasoning probes. A global high-salience
singleton Fourier direction may report another computation mode
while missing coordinated interventions in a hidden mode.
The proposed checkpoint-level experiment requires actual frozen
weights, a registered feature basis and heldout interventions.
No trained language-model or sentience claims arise from this
synthetic benchmark.
