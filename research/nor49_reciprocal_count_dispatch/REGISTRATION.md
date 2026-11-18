# NOR-49 preregistration: reciprocal multiplicity dispatch on timing-clean NOR42 programs

**Freeze this protocol in its first commit on November 18, 2026, before timing any NOR49 holdout program.** This is a historical-source but **timing-prospective** engineering study. First registration must be committed with no new holdout runtime measurement. Preserve the first original performance PASS or FAIL and its raw SHA byte-for-byte.

## Prior discovery and strictly excluded training panel

The original NOR48 fixed 16-source CPU receipt is **27,609 bytes SHA-256 `d5cf4b0e3860907eb58c2a37b84ff631f023caf2b69e6e0d0e5a322e2f0b2b99`**, archived before this registration. Its original first engineering speed target failed: median sourcewise NOR48 R-aware dispatch/NOR45 direct was **0.818287×**, against a registered <0.75×. Program operation counts on those **16 TRAINING timing sources** yield the new, explicitly data-derived hypothesis: seven signed programs with ≥2 active reciprocal `R` operators had ratios **0.193–0.551×** and five signed programs with exactly one `R` had ratios **0.785–1.096×**. The remaining four no-R programs were already correctly routed to NOR45 direct.

Freeze a single test-only decision rule *without any threshold tuning on holdout timings*: among `base,u,v` and `hidden_transform` **only when hidden_beta≠0**, count the number of positions whose signed authorized operation is `R`, counting repetitions in different fields. Choose NOR47's fully exact GMP trace with reentrant injection **iff count≥2**. Otherwise choose NOR45 direct full exact execution. This is a complexity heuristic selected using original NOR48 timed training sources, not a theorem about native arithmetic complexity. No other query coefficient, output, trace or timed feature affects routing. Read/signed-program parsing cost is included in each NOR49 dispatch timing.

## First completely timing-clean validation panel

Read the **original NOR42 64 public source-pair fixture and first grade already SHA frozen** before NOR45–48. Public fixture is 9,059,929 bytes SHA256 `f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033`, original grade 201,956 bytes SHA256 `bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae`; permanent original ZIP 2,799,867 bytes SHA256 `e60e97fd933f660c2274cb4a473ede7a98968d24eaf0dd3ae45ace1e1176cd0d`.

Exclude ALL 16 previously timed NOR46–48 source identities from replicate blocks **[100,104,108,112]**. Test ONLY 48 previously untimed source identities from the **12 original blocks [101,102,103,105,106,107,109,110,111,113,114,115]** × observed degrees D512/D1024 × dense/block-regime input families, four cells per block. This source roster, exact SHA list and original signed-program roots are frozen by a separate **public-only cohort selection job**, before any timed verifier arm executes. No new native true program or source identities are generated in NOR49. This is timing OOD relative to a previously timed source panel; all NOR42 native programs still share the same RRC25 grammar and public test keys.

## Original fixed engineering comparison

For each selected 48 source IDs, restore only its already original NOR42 clean/hidden truth, and reconstruct the original native authorization statement with independent public-source prior inputs. Produce the original NOR46 full exact trace before timing; keep the same precomputed trace for all trace-based arms. Full signature/session/provider-root checks run in an independent exact suite, and EVERY arm must return the same complete exact integer output and independently recomputed source-bound Merkle root.

Five **seeded and interleaved** repetitions for four arms: **NOR45 direct**, **NOR47 locally injected full GMP exact trace**, **NOR48 any-R dispatcher**, **NOR49 count-R≥2 dispatcher**. The full output+root runtime is included in every measured arm. Session/signature/provider transport is excluded equally as in NOR45–48. Benchmarking shall run on one pinned `gmpy2==2.3.2` Python 3.12 hosted runner, single thread, recording CPU model, full per-arm timings, process memory metadata and exact source IDs.

## Exactly five frozen engineering gates

1. **Primary absolute speed:** median across 48 sourcewise `NOR49_dispatch_ms/NOR45_direct_ms` is strictly **<0.75×**.
2. **Matched improvement over NOR48:** median across the same 48 sourcewise `NOR49_dispatch_ms/NOR48_any_R_dispatch_ms` strictly **<0.95×**.
3. **Deterministic authorized-program-only routing:** exactly every signed active R-count≥2 program chooses GMP, every count 0 or 1 chooses NOR45 direct, including across hidden transform when hidden_beta≠0; any unsigned model hint is ignored. No logged holdout timing is used for a branch choice.
4. **Exact and signed-source parity:** every arm rebuilds the same full exact integer vector and Merkle root; wrong signed authority/source/session/provider root and tampered R trace are rejected, and no-R direct ignores model trace content. Full source historical exclusion is checked.
5. **Reentrant resource controls and reproducibility:** original NOR46 global function remains unchanged, GMP source caps and original signed-integer semantics apply, first 48-source roster is frozen separately before any CPU timing, all five repetitions interleaved with source/repetition-specific fixed seeds, the first raw JSON and all five PASS/FAIL flags archived and independently SHA verified.

The first overall PASS requires all five gates. Any original FAIL remains visible. A descriptive 12-block t interval on **block-mean** sourcewise ratios will accompany the median primary endpoint, with by-Rcount, D and input-family breakdowns. A finite repeated timing run may vary by hardware, so all source-level original results are preserved. This experiment proves neither cryptographic soundness nor real-world provider honesty; external signed-program and trusted-target premises remain in force.

No new NOR49 holdout source identity or first CPU timings may be generated before this first registration commit. If implementation validation finds a pre-data bug, record an explicit amendment before any first holdout timing; change in the primary gates after first timing requires a new cohort.
