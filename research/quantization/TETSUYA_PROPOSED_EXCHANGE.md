# Follow-up to Tetsuya — draft only, not sent

Subject: Independent kernel replay, sentinel decoding, and recurrent-state training traces

Hi Tetsuya,

I’ve pushed a repository-first update to the research branches:

- [MathCan proofs and replay evidence](https://github.com/patrickdugan/MathCan/tree/research/quantized-jet-replay-20261115)
- [Approximation construction and reference traces](https://github.com/MoralityLabAI/Compactification-Theory/tree/research/quantization-d-audit-20261115)

The unchanged Hodge theorem now passes the pinned Lean 4.32/exporter/OxiLean
pipeline: 1,256 verified declarations, no unsupported or rejected declarations.
The earlier toolchain blocker is resolved.

The latest addition is 39 proved lemmas connecting base-4 sentinel decoding to
clipped spatial jet capture. Its dependency-closed replay gives 3,243
verified declarations and zero unsupported/rejected declarations; all
3,560 exported types and proof/value terms match the native
environment. The axiom inventories are explicit: propext, Quot.sound and
Classical.choice. Both the pinned 4.32 checker and the preserved 4.35 checker pass.

The decoder recovers concatenated payload frames exactly, retains the sentinel,
and has a proved 1/48 per-read margin. It needs 2(S+1) fractional slots to store
S payload bits. Code-state error grows by 4 per read, so the full-payload
robustness guarantee uses a length-dependent input budget; this is not a claim
of arbitrary-precision recurrent stability. Composing it with the spatial
address/rank proof gives the same selected clipped jet as the reference updates.
The full smooth approximation theorem still needs the particular amplitude/Taylor
transport specialization, analytic estimates, and dense compiler/resource proof.

This applied-math work is feeding our looped-layer transformer program at the
specification and reference-data level: we now have 95 generic research theorem
roots, and this continuation adds 144 positive trace fixtures, one no-sentinel
negative control, and 1,896 exact decoder steps through payload length 64.
Both circuit evaluators agree on every step. The fixtures currently label code,
read bits and suffix states. The proved correction, jet-update and capture
semantics define further candidate supervision and intervention targets,
indexed by recurrent visit and computational phase.

We have not yet shown that this guidance improves a trained transformer or
identifies its learned mechanism. The next controlled comparison is output-only
versus transition/invariant-guided training at matched compute, with independently
held-out lengths and interventions on those named states. The mathematical
reference being correct is one layer of evidence; the learned recurrence still
needs its own causal tests.

I’d value an independent attempt to break the decoder/capture composition or
its replay assumptions. The manifest, exact export hash, axiom closures and
per-declaration verdicts are in MathCan’s `research/oxilean_v015/pinned432_decoder/`.
A written repository exchange works well for me.

Patrick
