# NOR-41 historical source-block variation audit (NOR-40 only)

**Exploratory historical design sensitivity.** This note reads only the SHA-archived **original NOR-40 first grader**, preceding the first NOR-41 source cohort. It is written after NOR-41's prospective source-registration commit and after its first source-only fixture freeze, but before any NOR-41 grade was read. It does not adjust NOR-41's science gates, sources, selection algorithm or EVAL sample size. The new input-family OOD variance is unknown.

NOR-40 archival grade SHA-256: `42adeebcbf92626c080a6f0f5cb906eed491d6b5ed67250ba7b70bd6d0559092`, original 48 sources / eight replicates, each six old source family×degree cases. Its Global7 own-attack heldout improvement for CAL256 baseline versus CAL2048 strong attack was +0.3396pp at D512 and +0.0068pp at D1024 on equal-source averages. The **within-replicate D512-minus-D1024 gap** was +0.3328pp, with eight archived replicate block contrasts:

| Original NOR-40 replicate | D512 minus D1024 held-out attack improvement, percentage points |
| --- | ---: |
| 76 | +0.4780 |
| 77 | −0.0674 |
| 78 | +0.5595 |
| 79 | −0.0725 |
| 80 | +1.3345 |
| 81 | +0.6273 |
| 82 | −0.5989 |
| 83 | +0.4017 |

The historical contrast's standard deviation across eight blocks is **0.5806 percentage points**, with 5/8 positive block contrasts. The descriptive two-sided t95 half-width for n=8 is approximately **0.4854pp**. If the new cohort's block variance happened to match that historical sample, n=16 independent blocks would yield a t95 half-width near **0.3094pp**. This calculation uses `t_{15}=2.13145`, n=16 and the old sample SD. The new OOD family can change that variance materially.

NOR-41 Gate 1 was prospectively registered to require a positive scale contrast on **≥11/16** replicate blocks plus a mean contrast of **≥+0.1000pp**. These are exact registered practical thresholds, not a significance guarantee. A prior empirical mean +0.3328pp with an approximate half-width +/−0.3094pp shows why the new cohort may pass or fail while credible descriptions of the scale difference remain broad. Gate 3's ≥+0.1000pp combined EVAL attacker gain and ≥32/64 positive source conditions remain separate. Source-level CAL/EVAL samples or 16 greedy restarts are correlated within source and cannot be counted as additional independent replicates.

For the first NOR-41 grade, report degree- and family-specific source means and the 16 within-replicate scale contrasts. Analyze each registered gate on the frozen receipt before interpreting 16-block intervals or any optional sign-flip p-values. A scientific FAIL is a valid new scaling/OOD observation and must remain archived unchanged.

**Scope:** This is a variance-planning annotation using the already published NOR-40 grade with an old CAL2048 attacker. NOR-41's new attack uses CAL4096 and a new input family, so direct transfer of historical effect magnitudes or standard deviations is conditional and exploratory.
