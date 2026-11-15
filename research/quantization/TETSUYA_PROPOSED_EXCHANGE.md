# Proposed written exchange (draft only; not sent)

Subject: OxiLean v0.1.5 independent controls and MathCan replay

Hi Tetsuya,

I built commit 3adb983 with an existing Rust 1.94.1 compiler and replayed the
eight projection/recursor controls. Every named verdict matched the documented
expectation: 73 verified, 0 unsupported, 32 rejected across the eight files.
I have full declaration reports and SHA-256 fingerprints.

Our unchanged MathCan.Hodge.closed_path_gauge at 1315157 passes Lean 4.35.0 and
leanchecker, with propext and Quot.sound as its reported axioms. Its pinned
4.32.0-rc1 export/replay remains blocked by that missing local toolchain; I am
not treating the native check or regression controls as an independent replay
of that theorem.

Would you recommend retaining the exact 4.32 pipeline for this example, or is
there a tested newer-exporter combination? Is there a preferred report format
for comparing the exported target type and its complete dependency verdicts?

Separately, we proved a bounded-error quantized-jet recurrence and formalized
six exact integer lemmas in an Init-only module. Factorial normalization makes
adjacent Taylor transport matrices integral. The full smooth-function network
theorem still has an explicit spatial-interface/resource obligation.

I would be glad to exchange the reproducible artifacts in writing first.

Patrick
