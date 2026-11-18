# NOR60 authorized exact interpreter

NOR60 adds a fixed optional interpreter to the original signed NOR45 verifier.
It infers supported positions from the exact query and computes each needed
unary token once per invocation. The default NOR45 route remains available.

Read [REGISTRATION.md](REGISTRATION.md) for the frozen source panel, thresholds
and measurement boundaries. The complete first result and archive audit are
added after the registered hosted run. Fresh sources stay sealed until
`REGISTRATION_SHA.txt` names the earlier complete registration commit.

## Source-free checks

```sh
python -m unittest research.nor60_authorized_interpreter.test_optimized_execution research.nor60_authorized_interpreter.test_cohort_integrity -v
python -O -m unittest research.nor60_authorized_interpreter.test_cohort_integrity -v
```

The checks use small synthetic inputs, historical hash audits and public test
signers. They do not create a NOR60 source panel. Install the same
`cryptography==46.0.4` and `gmpy2==2.3.2` dependencies used by NOR59.

## Registered execution

The workflow performs the first command in an upstream source-only job and
uploads its artifact before the dependent CPU job begins the second command.

```sh
python research/nor60_authorized_interpreter/cohort.py --public-only-out nor60_first_original_60_public_only.json
python research/nor60_authorized_interpreter/benchmark_cpu.py --public-file nor60_first_original_60_public_only.json --out nor60_first_original_cpu.json --repeats 5
```

The CPU command keeps `nor60_first_original_cpu.progress.jsonl` and writes a
failure receipt if a process-level exception occurs. Existing paths cause
failure before any new measurements. Future replay should use the permanent
read-only archive auditor. A timing rerun requires a separately named result
and cannot replace the first measurements.

## Assurance scope

The optimized interpreter returns the full original exact integer vector and
the verifier checks the original signed full-index Merkle root. It preserves
the program-authority, verifier-precommit, provider-signature and query/session
checks. Every dynamic memo exists only within one evaluation. Public fixture
keys support reproducibility; external honest authority and source premises
remain. This module supplies executable tests and a prose derivation, with
the Lean proof obligation recorded in the registration.
