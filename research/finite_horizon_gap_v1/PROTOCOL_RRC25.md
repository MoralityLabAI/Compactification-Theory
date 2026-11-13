# FHC-01C: trained RRC-25 future-observation geometry

**Registration:** 2026-11-13, before evaluating this FHC-01C code.
**Evidence status:** exploratory retrospective frozen-model pilot.
**Original model:** RRC-25 recurrent GRU template ranker, 1,000 trained
parameters, source-locked JSON state dictionary in
research/rrc_compactification/results/rrc25_recurrent.json.
The existing bench25.checkpoints() gate validates model kind,
architecture, source hashes and training-step count.

## Construction

Build the RRC-25 case roster with the original generator, selecting
train_per=1, dev_per=12, test_per=8. Only the dev cases supply
the new geometry calibration; only the named six test families
signed/sparse/product/rational/squared/late_source enter test
metrics. The original model's training process precedes this study.

Each case supplies two public demonstrations and a known public
header. Call compound25.features(case, header) to get the same
9x8 feature sequence used by the source model. Run the source
portable GRU recurrence exactly, with zero initial hidden state.
Verify the final 16 logits against model25.logits on every case.

The represented recurrent state is the final hidden vector h9,
dimension 12, after processing all nine input rows. Freeze a
specific intervention: every later input vector is identically
zero, and no public query features are re-injected. For
T in {1,2,4,8,16}, repeatedly apply the trained GRU transition
to h9 with that zero input and concatenate the 16 logit outputs.

This future-logit metric is distinct from the native task's
suffix-class accuracy; it is a controlled extrapolation of a
learned recurrent state under an intentionally artificial
continuation. The experiment does not infer a physical or
natural-language semantic metric from these logits.

## Comparators and scoring

- Native hidden 12-dimensional code, paired with its exact
  recurrent continuation decoder. It is dynamically sufficient
  for this continuation, subject to floating numerical precision.
- Full future trace, dimension 16T, exact metric identity.
- PCA of future traces fitted only on dev cases.
- Fixed-seed Gaussian projection of future traces, independent
  of test features.
- Widths d={2,4,8,12,16}, restricted to d<=16T and
  d<=number of usable PCA components as appropriate.

Use a median positive calibration-pair ratio to fit the global
scale per representation. Record all test-pair distortion ratios,
source-near-zero exclusions, calibration D=1.5 and D=2.0
violation counts and worst replayable case-ID pair.
Also report each held-out family separately. The primary unit is
one source-generator case, not a pair of cases.

Retain model JSON SHA256, source hashes from bench25,
the exact feature roster identifier hash, independent source
file hashes, fitting seed, number of parameters, native hidden
width and number of oracle future steps consumed to construct
PCA/projected representations. Projection matrices are charged
as 16*T*d coefficients; PCA mean vectors are extra.

The native hidden state's 12 float32 coordinates do not count
as 12 **bits**. Its state description is at least 384 bits
under an fp32 storage model, separate from approximately 1,000
trained weights and the decoder program.

The absolute size of geometric distortion on one generated
case set does not establish a global Lipschitz or bi-Lipschitz
bound. A code that passes tested pair distances need not pass
the source distribution's support. The #098 theorem applies
to an infinite-dimensional compact doubling witness, which
the native twelve-dimensional GRU does not instantiate.

Results will be added after CI passes. This protocol does not
modify source checkpoints, RRC-25 scoring or its stored receipts.
