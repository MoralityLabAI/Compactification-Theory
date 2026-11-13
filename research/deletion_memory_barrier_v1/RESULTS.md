# DMB-01 results: deletion channels and protected monitorability

Date: 2026-11-13. Distinct evidence stages:
- **Exact finite mathematical controls:** standard-library Fraction tests.
- **Trained-model output diagnostic:** original source-locked RRC-25 GRU
  produces ranked candidate IDs; the monitor is a deterministic
  finite-state rule with an optimal terminal MAP decoder for the
  selected empirical source prior.
- **Source-level open research:** OpenAI Math #122 and #140 claims are
  imported by their stated scopes, without reproducing their proofs.

## Source pair

OpenAI [#122](https://github.com/openai/math/blob/main/lean/docs/122.md)
supplies a claimed worst-case deletion-trace sample lower bound.
OpenAI [#140](https://github.com/openai/math/blob/main/lean/docs/140.md)
supplies a claimed Gaussian streaming memory–precision lower bound.

The new theory note proves elementary consequences for terminal-state
capacity, finite deletion-deck pairs, data processing, spherical
precision and erasure-transferred Gaussian sampling. Its use of the
asymptotic #122 and #140 results is explicitly conditional.

## Exact deletion-deck witness

The strings 0110 and 1001 have the same subsequence multiset for
lengths 0,1,2. Their length-3 multisets are disjoint, and their
length-4 full traces differ. At deletion q=4/5,

    TV(P_0110, P_1001) = 17/625 = 0.0272.

An estimator attaining two-thirds exact recovery on *each* source
requires at least 15 independent traces by the m-fold coupling
bound. This is a necessary condition, not a decoder achieving it.

MathCan [PR #32](https://github.com/patrickdugan/MathCan/pull/32)
has Lean 4 / Init checks for the exact short-deck equality,
length-three disjointness and terminal decoder collision.

## Source-locked trained RRC-25 diagnostic

Use the existing original recurrent RRC-25 checkpoint with SHA-256

    54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b

The source's bench25.checkpoints gate verifies original training
hashes, model shape, and 1,000 trained parameters. Its public
feature/header pipeline supplies two top-ranked suffix class IDs,
encoded as eight bits. The original suffix truth and future labels
are never used to construct the ranked outputs.

The selected finite cohort contains 36 source-generated test cases
from six families. It has 19 distinct ranker-output words, Shannon
entropy 3.647462909576375 bits, and maximum prior source probability
11/36 (30.56%). Each source word has literal length eight.
Thus the uniform eight-bit source ceiling is inapplicable to this
structured empirical distribution.

At any number of trace observations, the DMB-8 state-count
ceiling is 20/36 (55.56%) for B=2, 33/36 (91.67%)
for B=4, and 100% for B=6. These are upper bounds over
*all* monitors with the stated mutable bit capacity for the
empirical source prior, given no external source-dependent side
channel. The demonstrated monitor falls below these bounds.

### Empirical exact-rational recovery rates

The monitor sees iid deletion traces of the fixed eight-bit ranking
word and stores only its B-bit state. The ideal final MAP decoder
is evaluated by exact Fraction arithmetic over the finite prior.
Its source-dependent lookup table is a separately charged
static decoder resource.

| Deletion q | 6-bit monitor | m=0 | m=1 | m=3 |
|---|---|---:|---:|---:|
| 1/2 | Overwriting rolling hash | 30.56% | 38.29% | 30.56% |
| 1/2 | Protected first nonempty trace | 30.56% | 36.18% | 36.20% |
| 4/5 | Overwriting rolling hash | 30.56% | 31.36% | 30.56% |
| 4/5 | Protected first nonempty trace | 30.56% | 31.32% | 31.47% |

The unlimited-message decoder using one complete deletion trace
achieves 42.55% at q=1/2 and 31.39% at q=4/5, evaluated exactly.
The protected monitor's three-trace score at q=4/5 can exceed
the full one-trace decoder score because it uses three independent
attempts to collect nonempty evidence.

DMB-9 supplies the formula

    P_guarded(m) = q^(n*m) * pi_max + (1-q^(n*m)) * A_g

under a fixed source prior, with A_g the Bayes success from
a nonempty retained trace's protected hash. It is monotone
and converges to A_g, while the rolling hash is permitted to
overwrite useful earlier information. Exact Markov-chain
enumeration and an independent closed-form implementation are
compared in the finite tests.

## Run instructions and provenance

From the root of Compactification Theory:

    python -m unittest discover -s research/deletion_memory_barrier_v1 -p 'test_reference.py' -v
    python research/deletion_memory_barrier_v1/reference.py --check --out /tmp/dmb01_control.json
    python -m unittest discover -s research/deletion_memory_barrier_v1 -p 'test_rrc25_deletion.py' -v
    python research/deletion_memory_barrier_v1/bench_rrc25_deletion.py --test-per 6 --deletions 1/2 4/5 --counts 0 1 2 3 --bits 2 4 6 --out /tmp/dmb01b_ranker.json

The dedicated workflow stores a finite source-hash receipt and a
trained ranker diagnostic JSON artifact. The source generator and
checkpoint remain unchanged. No network call or CUDA resource is used.

The trained RRC-25 word is an encoded pair of *ranked candidate IDs*.
It is neither a natural-language transcript nor an observation of
internal neural hidden-state computations. No claim about general
monitorability or reconstructing arbitrary proof traces follows
without a frozen, separately defined source distribution and
native hidden-state observations.
