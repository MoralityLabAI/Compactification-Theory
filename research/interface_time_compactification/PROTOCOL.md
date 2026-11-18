# IRTC-REV-01: interface-timed predictive compactification

**Registered source-only addendum, 18 November 2026.** Mathematical source [Beyond Bayes BB-CQ-05](https://github.com/patrickdugan/beyond-bayes/pull/24), merged SHA `abc89f73d223f00305eb0416a34fe5aee602494d`. Connects to [IRC-02 interface-relative compactification](../../experiments/interface-relative-compactification/README.md) and [CGC-01 protected causal compactification](../causal_gauge_compactification/THEORY.md).

## New resource axis: interface disclosure time

An encoder's available information at compression time is part of the interface contract. Let `S` be a future correction subset of `j` named reports among `n` old reports. The old match-bit vector is `x∈{0,1}^n`. Under the exact three-regime complementary ECA-04 likelihood, the present regime posterior can be encoded using the total `m=sum(x)` when `S` is uniformly hidden. A later update reveals `S` and its accepted replacement reports. Exact future inference for every legal `S` can require the initial encoder to distinguish **all `2^n` vectors**.

A delayed-disclosure encoder with unrestricted exact continuations thus requires `n` fixed bits for `1<=j<n`. With `S` supplied before compression, the valid count `u=sum_{i∉S} x_i` suffices and can be incremented on admitted replacements. A four-bit count field accommodates all accepted-report counts for `n=12`, whereas the initial known-schedule count at `j=6` has seven values (three bits). These statements charge *logical states*, and assume public stage, eligibility and known evidence-model parameters.

For delayed disclosure, an external authoritative archive may send `v=sum_{i∈S} x_i` on demand and the numeric updater recovers the accepted total from `m-v+q` for replacement-match total `q`. With `n=12,j=6`, a `v` message has seven possibilities and fits in **three bits**. The archive stores the old vector and verifies version/identity/source; its work and bytes must enter the resource vector.

## Interface-relative and physical costs

The resource vector should include
`(model_params, live_encoded_bits, external_archive_bits, correction_message_bits, schedule_bits, event_versioning, inference_operations, acquisition_calls, latency, physical_RAM, physical_VRAM, epsilon)`.
An unverified numeric message has no integrity guarantee. A current posterior in arbitrary-precision rationals may need more physical bytes than the theoretical count register. The cost comparison requires an explicit encoding/precision model.

Define equivalence of two initial old histories by equality of **all allowed future output posterior vectors**, after the same legal correction messages. The exact finite witness enumerates every correction subset on n<=7 and shows the `2^n` classes, with `n+1` current count-posterior classes. For general n the following proof applies: if every h=n−j size retained subset had equal bit-sum on two histories, exchanging one retained and one excluded coordinate forces every coordinate difference equal to a constant; any such subset sum forces that constant zero. Therefore the histories were identical. Under the strict interior ECA-04 likelihood distinct accepted match counts have different posteriors. This argument requires all correction schedules to have positive support and exact continuation fidelity.

The finite reference is **not** a lower bound on the number of real hidden coordinates in a neural model, entropy-compressed average storage, causal feature recovery, or optimal task regret. A compact neural recurrence may preserve only a task quotient, and that task quotient must be declared as a different contract.

## Cross-repo experiment registry

- [EvalMath EVAL-REV-01](https://github.com/MoralityLabAI/EvalMath): independently re-enumerate conditional posteriors and validate the Brier tower.
- [LoopedTransformers LRM-REV-01](https://github.com/MoralityLabAI/LoopedTransformers): native recurrent input/correction stream and posterior recovery gym.
- [BPD BPD-REV-01](https://github.com/MoralityLabAI/BPD): delayed-observable causal quotient witness with matched native interventions proposed for a separately preregistered phase.
- [Control-Harness ECA-REV-01](https://github.com/MoralityLabAI/Control-Harness): original synthetic host-signed report correction contract.
- [Beyond Bayes BB-CQ-05](https://github.com/patrickdugan/beyond-bayes): canonical twelve-slot exact Brier curve, memory theorem, and source-hashed receipt.

The CQ04 authorization HEADROOM_BLOCKED finding (0.006206757 cost benefit below 0.02) remains fixed. Proper-score Brier headroom 0.064462691 for six corrections is an exact property of its synthetic information law, not a newly demonstrated neural improvement. Existing NOR/SCM/CGC trained and finite experiments remain frozen.

Run `python -m unittest discover -s research/interface_time_compactification -p 'test_*.py' -v` for this source-only finite audit. No models are loaded or hidden confirmation roots sampled.
