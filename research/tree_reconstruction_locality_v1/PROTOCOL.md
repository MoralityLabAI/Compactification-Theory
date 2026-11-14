# KSR-01 synthetic-to-neural experimental protocol

Status: prospective beyond exact finite broadcast controls.

## Primary channel

Use a b-ary rooted semantic broadcast tree, with q=2 or 3
equally likely source labels. Every edge independently
transmits according to the q-state symmetric channel with
eigenvalue lambda. Record its transition matrix and
verify conditional independence. Sweep b={2,3,4},
L={1,2,3,4,5,8,16} where feasible, with below-critical,
critical and supercritical t=b*lambda^2 conditions.
Use b=4, lambda=1/2 for an exact rational critical cell.

## Observation and resource budget

The raw observer sees b^L terminal state labels.
The counter monitor streams all labels into q-1
counters for complete observation. With independent
retention r, keep q counters and a separate
missing-value token. Test r={1,1/2,1/4} and
depth-dependent r_L=t^(-L) for t>1.

Charge all symbol accesses, persistent counter bits,
static decoder/lookup description, model parameters,
precision, and arithmetic operations separately.
The mathematical O(L) bit claim concerns mutable
counter memory, not the entire decoder.

## AI research integration

A LoopedTransformers branch must mean a declared
proof-search continuation or synthetic storyworld
fork. Labels require an independent event oracle.
Freeze model checkpoint, tokenizer and source hashes,
and feature-to-semantic-label discretization before
fitting monitors. Do not apply the Potts channel
formula to correlated neural trajectories without
testing the source Markov and symmetry assumptions.

For hierarchical-reasoning models, compare a trained
streaming counter to the exact count-MAP oracle,
and to full-pattern Bayes where enumeration permits.
Use independent whole-tree examples for the
training and heldout splits. Publish root-recovery
accuracy, counter memory, retained observation budget,
per-tree failure witnesses and clustered confidence
intervals.

## Scientific limitations

Factor-of-IID is a generation property for an
infinite unrooted Ising field, distinct from root
reconstruction given leaf observations.
The source #229 critical non-reconstruction result
and source #236 factor construction are imported
claims at the indicated formalization scopes.
Finite Monte Carlo cannot establish either source
theorem. Synthetic reconstruction says nothing
about a trained model's hidden chain of thought.
