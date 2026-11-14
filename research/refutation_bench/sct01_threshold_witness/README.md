# SCT-01: SAT core thresholds and streaming proof memory

Started November 14, 2026, extending the existing
RefutationBench research in Compactification Theory.

Source pair:
[OpenAI Math #175](https://github.com/openai/math/blob/main/lean/docs/175.md),
Talagrand integral/fractional expectation covers, and
[OpenAI Math #235](https://github.com/openai/math/blob/main/lean/docs/235.md),
random SAT thresholds, hitting-time variance and computability.

[Mathematical derivations](THEORY.md) |
[Exact oracle](reference.py) |
[Streaming control gym](arena.py) |
[Resource and evidence protocol](PROTOCOL.md) |
[Executed results](RESULTS.md)

## New mathematical findings

Two disjoint complete eight-clause 3-SAT gadgets generate an
increasing UNSAT family on sixteen clause types.
Its integral and fractional expectation-cover costs
both equal min(1,2p^8). Under independent Bernoulli
clause inclusion, UNSAT probability is 2p^8-p^16.

For clause draws *with replacement*, the first-UNSAT
time has an exact inclusion-exclusion CDF,
an independent Stirling/surjection formula,
mean 1481069/45045 (about 32.88), variance
80211997256/676350675 (about 118.60), and median 31.

Any deterministic monitor deciding SAT correctly on
all possible streams, with arbitrary repeated clauses,
needs exactly (2^k-1)^r+1 states for r disjoint
complete k-type gadgets. The corresponding
proof-producing monitor needs (2^k-1)^r+r states.
For r=2,k=8 these are 65,026 and 65,027 states,
both requiring 16 fixed-length mutable bits.
A completed contradiction itself is identified
with one public gadget-index bit.

The exact full proper random 3-SAT model on
n=3 variables is also provided as a coupon-collector
calibration with mean 761/35 and variance
838034/11025. The six-variable main experiment
uses a *restricted two-gadget clause universe*,
so its finite results are distinct from source
#235's full random-six-variable distribution.

## Reproduce

Python 3.10+ standard library:

    python -m unittest discover -s research/refutation_bench/sct01_threshold_witness -p 'test_*.py' -v
    python research/refutation_bench/sct01_threshold_witness/reference.py --out /tmp/sct01_exact.json
    python research/refutation_bench/sct01_threshold_witness/arena.py --trials 1024 --out /tmp/sct01_stream.json

Every one of the 65,536 possible formula subsets is
checked against an independent full 64-assignment
SAT oracle. Exact rational occupancy CDFs are
cross-checked against Stirling numbers and
a separate subset-state Markov moment oracle.
The deterministic monitor transition is tested
over all incomplete clause states and all sixteen
possible next clauses.

The small streaming gym reports binary correctness,
false-positive UNSAT claims, missed contradictions,
witness validity and live-state bit budgets.
A monitor counting raw clause arrivals can invent
an UNSAT claim after eight repeats of one clause.
The exact monitor prevents this by tracking
distinct clause presence.

The [MathCan Lean companion PR #39](https://github.com/patrickdugan/MathCan/pull/39)
kernel-checks the local proper-clause semantics
and the finite monitor-state arithmetic. The
general expectation-threshold and asymptotic
random-SAT source claims are external, with their
full proofs outside these finite checks.

No trained Lean reasoner or language-model
checkpoint was evaluated in this campaign.
