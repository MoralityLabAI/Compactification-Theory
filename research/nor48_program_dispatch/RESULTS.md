# NOR-48: source-signed reciprocal-aware GMP dispatch, first historical exact CPU result

**First original engineering verdict: FAIL (4/5 registered gates).** The prospectively frozen [NOR-48 design](DESIGN.md) used 16 previously independently verified signed native programs and a fixed five-repeat interleaved CPU comparison. Its registered primary goal was a median **sourcewise dispatch/NOR45 direct ratio below 0.75×**, with source-root parity and all R/no-R routing constraints. The first hosted [CPU run #44253553267](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44253553267) produced **0.818287×**, so the speed target failed. Every signed-program, root-parity and R/no-R dispatch gate passed.

## Original immutable CPU evidence

| First original receipt | Immutable value |
| --- | --- |
| First five-repeat source-locked benchmark | [GitHub Actions #44253553267](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44253553267) |
| First hosted original CPU artifact | [#14404902340](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44253553267/artifacts/14404902340) |
| First raw CPU [JSON receipt](results/nor48_first_original_cpu.json) | **27,609 bytes** |
| Exact first JSON SHA-256 | `d5cf4b0e3860907eb58c2a37b84ff631f023caf2b69e6e0d0e5a322e2f0b2b99` |
| Original first [archive manifest](results/nor48_first_receipt.json) | Source and gate hashes preserved |
| Previous NOR47 source/time reference SHA | `809fd5d18cfb61e4e6d771c4e402b31e3fec99889774f4305ddfa3d6c793f530` |
| Registered cohort | 16 completed NOR46 original signed programs from replicate blocks **100, 104, 108, 112** |
| First repeat count | Five seeded interleaved repetitions per source/arm |
| Backend | `gmpy2==2.3.2` |
| New source identities and new model evaluations | **0** |

The original [one-shot publisher #44254179705](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44254179705) completed successfully and committed the first CPU JSON bytes unchanged, including negative performance. The permanent [original_archive_audit.py](original_archive_audit.py) checks the raw SHA, selected 16 source identities, five gates and original calculated per-source ratios. No timing run or performance gate was altered after observing the first receipt.

## Full original source-level arithmetic+root timing

Each of the four benchmark arms reconstructs all output coefficients and the complete independent output Merkle root on the same source. Program/session/provider signature checks, network and transport costs are excluded **equally** in these CPU arithmetic timings. The test suite checks signed authority, precommit, provider root, malicious model-provided trace and wrong-program/source controls separately.

| Auditor | Median of per-source wall times |
| --- | ---: |
| NOR45 original exact direct + root | **49.7829 ms** |
| NOR46 original Python exact full trace + root | **227.3625 ms** |
| NOR47 explicit GMP full trace + root | **28.4686 ms** |
| **NOR48 R-aware exact dispatch + root** | **28.4283 ms** |

The median of the 16 **individual dispatch/direct ratios** is **0.818287×**. The ratio of the displayed median runtimes is a different statistic; the prospectively registered decision uses the median individual source ratios.

| Precommitted routing group | Source count | Median dispatch/direct ratio | Median GMP/direct ratio | Median dispatch/GMP ratio |
| --- | ---: | ---: | ---: | ---: |
| Source programs using reciprocal `R` | **12** | **0.503542×** | 0.502952× | 1.000032× |
| Programs without `R` | **4** | **1.004956×** | 1.810600× | 0.553207× |
| **All 16** | **16** | **0.818287×** | 0.817688× | 0.999717× |

The four no-R sources correctly avoid the expensive GMP trace and use NOR45 direct execution; the observed dispatcher overhead is approximately **0.5%**. The twelve R sources retain the acceleration of GMP exact signed convolution without changing the global NOR46 verifier function, including when an injected kernel raises.

### All original 16 sourcewise ratios

| Replicate | D | Public input family | Authorized R? | Dispatch/direct ratio |
| --- | ---: | --- | --- | ---: |
| 100 | 512 | Dense | No | 0.998× |
| 100 | 512 | Block regime | Yes | 1.096× |
| 100 | 1024 | Dense | No | 1.007× |
| 100 | 1024 | Block regime | Yes | 0.398× |
| 104 | 512 | Dense | Yes | 0.193× |
| 104 | 512 | Block regime | Yes | 0.334× |
| 104 | 1024 | Dense | Yes | 0.785× |
| 104 | 1024 | Block regime | No | 1.006× |
| 108 | 512 | Dense | Yes | 0.892× |
| 108 | 512 | Block regime | Yes | 1.078× |
| 108 | 1024 | Dense | Yes | 0.456× |
| 108 | 1024 | Block regime | Yes | 0.551× |
| 112 | 512 | Dense | Yes | 0.240× |
| 112 | 512 | Block regime | No | 1.004× |
| 112 | 1024 | Dense | Yes | 0.423× |
| 112 | 1024 | Block regime | Yes | 0.852× |

Among the 12 reciprocal programs, **five** have dispatch/direct ratios at or above 0.75. Two of these reciprocal programs are even slower than direct execution. Among all 16, **nine** cases have ratios at or above 0.75, including the four no-R cases. R presence alone therefore does not perfectly determine which exact arithmetic backend is faster.

The four replicate-block **mean** sourcewise ratios are **0.87455**, **0.57929**, **0.74435**, and **0.62977**. Their descriptive overall mean is 0.70699 and the four-block t95 interval is **[0.4980, 0.9160]**. This interval describes the replicate-block mean, while the registered primary engineering target is the pooled **median** of the 16 sourcewise ratios. The four small blocks provide weak evidence for generalization to different native programs or hardware.

## Original frozen engineering gates

| Registered first gate | Observed | First verdict |
| --- | --- | --- |
| Median sourcewise dispatch/NOR45 direct less than 0.75× | **0.818287×** | **FAIL** |
| All four authorized no-R programs use NOR45 direct | 4/4 | PASS |
| All four exact-output/root verifier arms agree | 16/16 sources | PASS |
| All 12 authorized R programs use injected GMP exact trace | 12/12 | PASS |
| Process-global NOR46 convolution remains unchanged | Unchanged across all arms | PASS |

The benchmark runner uses an explicit *reentrant* convolution function parameter in the NOR48 trace verifier, with the same full exact no-carry integer verification requirement, signed coefficient caps and all intermediate transform/output checks. The [NOR48 historical signed tests](test_dispatch.py) also verify a complete pinned authority signature and NOR44 verifier precommit/session and NOR43 provider-signed root; the signed routing decision is based solely on the authorized program. A model-supplied unsigned backend hint cannot select the trace path. If the authorized program contains no R, the direct path ignores any malformed model-provided trace; otherwise the R path rejects mutated inverse/output evidence.

The first CPU experiment compares full original NOR46 Python trace, local explicit-injected NOR47 GMP trace, source-authorized program R-aware dispatch and original NOR45 direct computation. Test-only keys are publicly known, and all underlying original source and signed program roots were locked prior to this study. A signed correct execution trace proves that the verifier faithfully ran its authorized native program, **conditional on** cryptographic assumptions and independently pinned signing identities. Actual authority honesty, the provider's true-output correctness, and real secret entropy remain external trust assumptions. The trace contains all exact integer data and is **not succinct nor zero-knowledge**.

## Next experiment

The first measured FAIL identifies a falsifiable next step: learn or analytically derive a **program-and-public-query-complexity-conditioned** exact execution backend selector, with a separate clean prospective validation source panel. A source-only feature could measure public coefficient bit widths and operation count from the signed native program, then choose either local GMP trace or NOR45 direct without accessing the true future, malicious trace, or timing outcomes. A source-specific choice made after seeing the original NOR48 measured runtimes would overfit the completed 16-source cohort. Such a selector needs its **own preregistered source-exclusion list, threshold, cost parity checks and original FAIL archive** before new data.

The historical NOR48 original fixed benchmark is final and must remain unchanged. It is a conditional CPU engineering observation; results cannot be extrapolated from 16 fixed signed RRC25 programs to arbitrary executors or deployed inference services.
