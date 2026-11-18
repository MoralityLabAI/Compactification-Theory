# NOR-47: GMP exact signed-integer verifier comparison

[NOR-47 design issue #248](https://github.com/MoralityLabAI/Compactification-Theory/issues/248) freezes a historical-only performance study after NOR46's original 16-source trace correctness PASS and CPU slowdown result.

The fixed 16 source identities are NOR46's original archived replicate blocks 100, 104, 108 and 112, with observed degrees 512/1024 and dense/block-regime inputs. First NOR46 original trace receipt SHA256: `d24bb2fec989a5cbe7422d3aae6e2e2779a9d780268f25f0f23ccf28f56db052`. First NOR46 CPU receipt SHA256: `4e9695f89aaa8acf42c09c6fe7aea9ee47ca0f86f74f6131d3e108e587454a64`.

Four original frozen benchmark arms use the same CPU, five repetitions, interleaved seeded arm ordering and identical roots: NOR45 direct full execution, NOR46 original Python exact Kronecker full-trace verification, NOR47 GMP packed verification and NOR47 hybrid Python packed / GMP multiplied verification. Signed program/provider/session checks are tested in the exactness gate and excluded from all timing arms; complete Merkle-root recomputation is included in every timed arm.

[Original first exactness CI #44240546887](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44240546887) passed all five test groups, including exact signed 4097-bit convolution and original 16 authorized-program negative controls with both GMP backends.

The frozen engineering targets are median per-source GMP/NOR45 execution+root <1.0 and GMP/NOR46 Python trace+root <0.50. A negative first outcome remains a valid archived benchmark result. There are zero new NOR47 source identities or scientific model outcomes. The backend override is single-threaded and benchmark-only; the original verifier remains unchanged.

**First original CPU receipt:** this commit triggers the exact five-repeat, four-arm historical benchmark. Preserve the original first SHA and all measured outcomes, whether speed targets pass or fail.
