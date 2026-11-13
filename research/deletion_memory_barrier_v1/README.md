# DMB-01: deletion–memory barrier

Research campaign launched November 13, 2026.

Source results: [OpenAI Math #122](https://github.com/openai/math/blob/main/lean/docs/122.md), worst-case deletion trace reconstruction lower bounds, and [OpenAI Math #140](https://github.com/openai/math/blob/main/lean/docs/140.md), streaming Gaussian memory–precision lower bounds.

[Mathematical proofs](THEORY.md) | [Exact reference](reference.py) | [Tests](test_reference.py) | [Experimental protocol](PROTOCOL.md)

## New deductions

DMB-1: a B-bit terminal monitor without leaked time index cannot exactly recover a uniform n-bit source with average success exceeding 2^(B-n), irrespective of trace count.

DMB-2/DMB-3: matching short-deletion decks bound one-trace total variation and hence bound exact recovery from m iid traces, even with unrestricted memory. Witness: 0110 versus 1001, identical subsequence decks through length 2, one-trace TV=17/625 under deletion q=4/5, which forces at least 15 traces for success >=2/3 on both words.

DMB-4: a two-resource information ceiling I(X;state,stop index|independent randomness) <= min(B+log2(m+1), m*n*(1-q)). Finite-alphabet Fano supplies the approximate-recovery corollary.

DMB-5/DMB-6: an elementary spherical-cap precision floor and a conditional extension of source #140 to independently erased Gaussian observation opportunities. The sample bound requires at least 5/6 success before truncation and charges additional counters. Its numerical constant is inherited symbolically from #140.

DMB-7: a two-word structured prior can be recovered by a one-bit monitor, demonstrating why prior entropy must be declared.

## Reproduce

From repository root, with Python 3.10+ and only the standard library:

    python -m unittest discover -s research/deletion_memory_barrier_v1 -p 'test_*.py' -v
    python research/deletion_memory_barrier_v1/reference.py --check --out /tmp/dmb01.json

All tests are exact finite controls and elementary analytic checks. No trained neural checkpoint is run; no asymptotic hard pairs or original OpenAI source proofs are reproduced.

The research owner is Compactification Theory, with applications to RC-01, LoopedTransformers, RefutationBench and StegoBench. The source #122/#140 manuscripts are research claims under community review.
