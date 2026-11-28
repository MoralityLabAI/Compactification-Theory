# Recorded ARC-2 pilot results

Scope: six selected **public evaluation** tasks; exploratory result. The correct result was obtained after task exposure.

- Frozen generic DSL/program search: 0/8 exact outputs; 0/6 tasks.
- Exploratory glyph extension: 1/8 exact outputs; 1/6 tasks.
- Correct task: `136b0064`, first prediction exact.
- Gate cell: 9,378 trainable parameters; three seeds each 64/64 Boolean truth-table rows.
- Exact controlled comparator: each seed 4,096/4,096 unsigned six-bit comparisons.
- Additional direct-forward audit: 126,976 gate instances across 4,096 comparisons, all correct.
- Masking gate role key reduced local accuracy to 50%.
- Glyph cell: 9,784 parameters, four inferred action rows, three successful seeds.
- Complete exploratory hybrid: 19,162 parameters.
- 256 within-grammar synthetic sequences matched final grids and state traces.
- Separate training of a one-pass counterpart and opaque recurrent baseline remain undone.
- Original packaged tests: 16 passed.

Interpretation: this is evidence for a **typed neural-semantic / exact-control interface** on small finite domains. It is not evidence of competitive ARC-AGI-2 generalization or superior recurrence.

Primary recorded score artifact: `gcpt_arc2_score.json` in the original sandbox. Its prediction SHA256 is `ad81143e05689e99f0453e0cc463d09d40b213f4ffd6db3f7602d5d083a6a2e6`.
