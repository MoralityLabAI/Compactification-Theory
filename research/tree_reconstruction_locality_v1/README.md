# KSR-01: branching reconstruction and compact monitor locality

Research started November 14, 2026.

Source pair:
[OpenAI Math #229](https://github.com/openai/math/blob/main/lean/docs/229.md),
three-state reconstruction thresholds, and
[OpenAI Math #236](https://github.com/openai/math/blob/main/lean/docs/236.md),
factor-of-IID threshold for free Ising spins on regular trees.

[THEORY.md](THEORY.md) derives the exact second-moment recurrence
for a normalized q-state symmetric broadcast leaf-count monitor:

    t = b * lambda^2
    V_L = t^(-L) + ((b-1)/b) * sum_{j=0}^{L-1} t^(-j)

The theorem proves that optimal root recovery from leaf-label
histograms has accuracy at least 1/q + (q-1)/(q*V_L).
For t>1 the floor remains above 1/q at arbitrarily
large depth, although a streaming counter requires only
O(depth) persistent bits after processing b^L leaves.

Independent leaf retention r adds exactly (1-r)/(r*t^L)
to the normalized second moment. The same conditional
posterior argument bounds recovery from retained counts.

The threshold t=1 matches algebraically with the
source #236 condition for automorphism-equivariant
factor-of-IID generation on an unrooted infinite Ising
tree, while these are distinct generative/inferential tasks.
The critical FIID endpoint is included in the source
claim; finitary coding and coding radius are outside scope.

## Reproduce

Python 3.10+ and standard library only, from repository root:

    python -m unittest discover -s research/tree_reconstruction_locality_v1 -p test_reference.py -v
    python research/tree_reconstruction_locality_v1/reference.py --check --out /tmp/ksr01_exact.json

The exact reference independently checks q-state transition
eigenfunctions, LCA covariance, full Bayes posterior distributions,
count-only optimal decoding and retained-leaf moment corrections,
including the critical rational cell b=4, lambda=1/2.

[PROTOCOL.md](PROTOCOL.md) specifies prospective tests in
LoopedTransformers and recurrent semantic monitoring. The
current phase reproduces neither the original infinite-tree
proofs nor trained language-model results.
