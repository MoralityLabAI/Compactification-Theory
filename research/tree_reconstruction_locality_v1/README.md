# KSR-01: branching reconstruction and compact monitor locality

Research started November 14, 2026.

Source pair:
[OpenAI Math #229](https://github.com/openai/math/blob/main/lean/docs/229.md),
three-state reconstruction thresholds, and
[OpenAI Math #236](https://github.com/openai/math/blob/main/lean/docs/236.md),
factor-of-IID threshold for free Ising spins on regular trees.

[Exact results](RESULTS.md) | [Learned monitor results](RESULTS_B.md) | [THEORY.md](THEORY.md) derives the exact second-moment recurrence
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


## Executed extension: sparse sampling and learned count monitor

KSR-6 fixes the depth-dependent retention probability
r_L=(b*lambda^2)^(-L) whenever t=b*lambda^2>1. It
proves that expected observed leaves decrease from b^L
to lambda^(-2L), while the retained-count monitor's
reconstruction floor stays above the uniform-prior
guessing level. For b=2, lambda=4/5 the exact moment
gap is 7*B_L+16*25^L=23*32^L. The resulting
count-decoder root accuracy is at least 37/69 for
q=3 and 15/23 for q=2 at all finite depths.

[RESULTS.md](RESULTS.md) records exact rational
posterior probabilities and independently checked
dropout moments. The companion
[MathCan PR #35](https://github.com/patrickdugan/MathCan/pull/35)
kernel-checks the discrete eigenfunction and variance
gap arithmetic, without claiming an infinite-source proof.

[RESULTS_B.md](RESULTS_B.md) reports the separate
newly trained synthetic 15-parameter multinomial
count-monitor pilot. It has 4096 train and 4096
independent heldout trees per condition, paired
fit seeds and matched random-label controls.
[PROTOCOL_LEARNED.md](PROTOCOL_LEARNED.md) retains
the pre-execution model, hyperparameter and split
contract. The learned model does not improve on
simple plurality in every tested condition, which
is recorded as a substantive negative comparison.
