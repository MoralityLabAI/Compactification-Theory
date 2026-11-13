# DMB-01B: frozen trained RRC-25 ranking-trace monitor

**Registration date:** 2026-11-13, before the DMB-01B model run.
**Status:** retrospective diagnostic on existing weights and constructed
deletion observations; no new model training.

## Model and source

Source code and committed JSON checkpoint from
research/rrc_compactification:
RRC-25's 1,000-parameter recurrent GRU template ranker.
Use the original bench25.checkpoints() validation of source SHA256,
training-step count, model type, shape and weight-file digest.

Generate the original deterministic RRC-25 source roster with
train_per=1, dev_per=1 and test_per=6. Restrict the study to the
six original test families (signed, sparse, product, rational,
squared, late_source). At inference scan public candidate headers
using compound25.possible_headers. Choose the first valid header
deterministically, independent of true hidden program suffix.

For each case compute model25.logits(case,header,checkpoint), sort
sixteen logits descending (tie-break by class ID), and take the two
top-ranked 4-bit class IDs. Concatenate these IDs into one 8-bit
word. The word encodes a **trained model's ranker output**, not
its neural hidden state or a claimed chain-of-thought transcript.

The study treats the resulting multiset of 36 output words as a
finite empirical prior. Repeated words carry their multiplicity.
No source ground-truth future enters the ranker input; mutation
of case.future must leave the ranking unchanged.

## Monitoring intervention

For each word, independently delete each bit with q=1/2 or q=4/5.
The monitor sees only retained word bits in their original order
with the resulting output length. Position masks remain hidden.
Run m in {0,1,2,3} independent deletion traces.

Compare fixed deterministic monitor updates with B in {2,4,6}
bits, using the exact transition modulo 2^B defined in
reference.update_state. The monitor is not trained; the ideal
terminal decoder is a Bayes MAP lookup derived from the declared
finite prior. Charge its static lookup separately from the B
mutable bits.

Primary analytic control: DMB-8 top-2^B prior mass ceiling.
Secondary oracle: unrestricted full one-trace Bayes success from
the exact deletion laws. Also record the prior's Shannon entropy,
distinct word count, weighted source label counts, and all
exact rational finite-state reconstruction probabilities.

The original checkpoint has weights learned on other cases.
The prior and decoder table are defined retrospectively by
the selected source-generated ranker outputs. This is a finite
in-sample population diagnostic with no claim about deployment
generalization.

## Reproducibility

Run the trained checkpoint gate and exact monitor calculations:

    python research/deletion_memory_barrier_v1/bench_rrc25_deletion.py --out /tmp/dmb01b_ranker.json

No CUDA, torch, network requests or optimizer required. The
portable RRC-25 inference uses the source's existing Python model
and original JSON checkpoint. The report includes all original
case identifiers, class-pair words, source hashes and monitor
success fractions.

Interpretation distinguishes source support entropy from the
eight-bit literal word length. No claim about the source #122
asymptotic pair or the #140 Gaussian observation experiment
follows from this small trained-ranker diagnostic.
