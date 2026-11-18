# NOR-48: authorized-program R-aware exact verification dispatch

[Design issue #251](https://github.com/MoralityLabAI/Compactification-Theory/issues/251) fixes the next historical-only engineering question after NOR47's merged [original source-locked GMP result](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/nor47_gmp_exact_convolution/RESULTS.md). The exact original NOR47 CPU receipt is **34,928 bytes**, SHA256 `809fd5d18cfb61e4e6d771c4e402b31e3fec99889774f4305ddfa3d6c793f530`. Twelve prior source programs use reciprocal R, where GMP full-trace verification accelerated the exact NOR45 baseline; four have no R, where NOR45 direct computation remained faster.

## Signed-source backend selection

A verifier first checks the NOR45 independently pinned Ed25519 **program-authority signature**, original closed statement schema, program-authority key ID, exact-program semantic version, source query digest and observed output length. It then decides on a backend using **only** the authenticated program's original operation fields `base,u,v` and optional nonzero-β hidden transform. If any authorized operation is `R`, it selects NOR47 GMP exact full-trace verification within NOR46's original signed context and output-root verifier. For a program with no R it selects original NOR45 full exact execution and output-root verification. Both paths independently reverify complete NOR44 session/precommit, provider root, signed program authority and final committed output root.

The model cannot supply a backend selection override. Tampering with a signed program, source query, observed degree, provider manifest or session nonce fails authorization, regardless of backend. In a no-R program the provided trace is ignored and the original authorized program is recomputed; on R paths every supplied unary coefficient and full output is validated. The complete signed root verifies faithful computation of the separately authorized program, conditional on prior signing/provenance assumptions; it cannot establish the semantic truth of a false signed program.

## First historical benchmark protocol

Exactly the original **16 NOR46/NOR47 source identities** from replica groups 100,104,108,112 × D512/D1024 × dense/block-regime. No NOR48 source identities are generated. Each of four fixed arms is run for **five seeded interleaved repeats per source**, on the same hosted CPU:

- NOR45 full signed session, program and authorized execution.
- NOR46 full signed session and pure Python complete trace.
- NOR47 full signed session and GMP-accelerated complete trace.
- NOR48 preliminary authenticated signed-program R dispatcher.

All four arms include the full original signed program/session/provider key checks and output-root construction; transport and source generation are excluded from every timed arm. The registered engineering target is source-median NOR48 dispatcher/NOR45 full signed execution ratio **<0.75×**, with exactly 12 R/GMP branch selections and four no-R/direct branch selections, 16/16 exact output/authorized Merkle root matches and 16/16 parity of the original signed-program negative controls. Archive original first wall times, first receipt SHA, hardware, code versions and original engineering PASS or FAIL unchanged.

A global replacement of NOR46's exact convolution function exists solely in the **single-threaded historical benchmark** and is restored by a `finally` clause. This prototype must not be deployed concurrently; a reentrant explicit dependency-injection interface requires a separate design review. All signing keys and generated historical test sessions are reproducible PUBLIC research fixtures and do not establish real-world authorization.

The study remains a transparent nonsuccinct source-bound execution certificate; no STARK/SNARK, ZK claim or verified external provider honesty is implied.
