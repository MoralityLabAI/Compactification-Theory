# NOR-51: exact reciprocal length/precision CPU crossover

[Frozen original registration](REGISTRATION.md) commit `83fabe1c34c2bedbba35faa928349f41b964115a` precedes all NOR51 public sources and measurements. Continues merged [NOR50](../nor50_memoized_exact/RESULTS.md), where 19 multi-R source programs saw **1.183263×** median GMP/uniquely memoized native timing, with the original registered GMP speed goal failing.

This next experiment isolates the single-R arithmetic problem. [cohort.py](cohort.py) creates 48 previously unused public-only signed integer series, from a fully registered 2×3×4×2 grid of source replicates, lengths 769/1537/3073, sparse 2/8/16-bit and dense 2-bit coefficient profiles, and sign families. The original first source JSON and SHA must be uploaded in an upstream GitHub Actions job before any inverse, witness, or timing result is produced.

The exact [algebra.py](algebra.py) implements three full-output and independent Merkle-root arms: original NOR45 single-inverse recurrence, NOR47 GMP verification of a complete untrusted exact inverse witness, and Newton exact inverse construction with a final independently checked GMP identity. Each step of the Newton producer uses two signed carry-free GMP convolutions, and the final output reproduces the full direct integer vector.

The registered [benchmark_cpu.py](benchmark_cpu.py) makes five source-seeded interleaved comparisons per arm. The independent reference inverse and its canonical witness serialization are generated once outside arm timing, with their exact time/bytes logged separately. The end-to-end Newton arm includes both production and verification time. Corrupted witnesses, incorrect normalization and resource-cap violations are rejected in a source-free unit suite. No model evaluation, hidden native program, or new signed-authorization grammar is assumed.

Five preregistered engineering gates cover 48-source exact output/root parity, longest-length verifier advantage below 0.75×, longest-length Newton+verify below 0.95×, matched length-slope improvement below 0.80×, and source-first exact resource/reentrancy invariants. The original first outcome, PASS or FAIL, must be archived immutably. Secondary grouped intervals may describe variance without altering the first verdict.

NOR51 tests a synthetic exact arithmetic kernel and is **not** a complete authorized native verifier at 3073 coefficients. NOR45 signed authority, external provider honesty, session transport and real-world witness security remain outside the claims.

**First timing state:** the runner is armed only by a source-commit message `NOR51-FIRST-CPU` after the source-free tests pass. The first 48-source roster must be separately uploaded before the dependent CPU measurement begins.

**First original source and CPU trigger:** predata CI passed on the frozen implementation, including exact Newton parity against quadratic inverses and malicious witness/resource checks. This commit starts the first source-only 48-case SHA freeze; only its dependent job may subsequently execute first NOR51 CPU timings. The original PASS/FAIL verdict and raw SHA will be retained without changing the registered gates.
