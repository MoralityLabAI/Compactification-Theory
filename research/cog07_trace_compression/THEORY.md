# Finite version-space compression

**Evidence:** mathematical derivation with generic Lean supporting lemmas; the concrete corpus certificate uses exhaustive finite enumeration.

## Statement

Let `H` be a finite ordered class of Boolean hypotheses on a feature domain `X`. For a finite labeled sample `S`, write

\[
V_H(S)=\{h\in H : h(x)=y\text{ for every }(x,y)\in S\}.
\]

Choose `C` from `S` so that every `h` outside `V_H(S)` disagrees with at least one example in `C`. Then

\[
V_H(C)=V_H(S).
\]

Every full-sample survivor satisfies the selected constraints because `C` is a subset. Coverage eliminates every other allowed hypothesis. Together these inclusions establish equality. A fixed function of the version space therefore returns the same output for either sample. This conclusion applies even when the version space is empty; the executable classifier requires a nonempty space so that it can choose a consistent hypothesis.

The three named Lean declarations are `cover_preserves_consistency`, `cover_preserves_version_space`, and `fixed_decoder_is_preserved`. The last allows an arbitrary decoder of a version-space predicate. It places no requirement on the way that decoder is implemented.

## Executable selector and bound

Each labeled example eliminates a subset of `H`. The selector starts from the hypotheses eliminated by the full sample and repeatedly takes the example covering the largest remaining subset. Ties use the first source index. Repeated feature/label pairs have identical elimination sets and are deduplicated before selection.

While a hypothesis remains uncovered, some full-sample example eliminates it by the definition of `V_H(S)`. Hence a selection step always removes at least one remaining hypothesis, giving

\[
|C|\le |H\setminus V_H(S)|.
\]

This bound depends on the declared class. It supplies neither a VC-dimension bound nor a bound on neural model size. The experiment checks equality of the full bit masks after selection and again after decoding the transmitted packet. The independent provenance verifier recomputes the source features, labels, and surviving hypothesis IDs.

## Concrete instance

Eight feature bits and three literal choices per bit yield 6,561 conjunctions, including constant true. Appending constant false gives 6,562 hypotheses. The complete training sample leaves a single hypothesis with `required = 7`, `forbidden = 0`, and `constant_false = false`.

The selected five examples exclude the same 6,561 alternatives. Their class-independent feature/label payload uses `5 × 9 = 45` bits. Byte alignment makes this six bytes; the format adds a seven-byte header. The 46 unique training constraints require 59 bytes in the same format. The entire 11,836-row sample requires 13,323 bytes.

These byte counts condition on a shared feature extractor, hypothesis order, decoder and format. A direct hypothesis ID in the shared class requires `ceil(log2(6562)) = 13` bits. The transmitted five-example packet therefore answers a constraint-preservation question. The paper-inspired connection to sample compression consists of selecting labeled observations that preserve a decoder within an explicit class.

## Verification boundary

Python enumerates the concrete feature class and checks every survivor. Lean proves the general implication from the coverage premises. The concrete packet and corpus have no Lean encoding in this change. The native receipt records the exact supporting theorem source and compiler version.

The known target formula uses the first three public features. It follows that the recovered hypothesis agrees with the target on all 256 feature patterns. The empirical zero-error rows exercise corpus extraction and decoding. They do not estimate the capacity needed for an unknown mathematical reasoning task.
