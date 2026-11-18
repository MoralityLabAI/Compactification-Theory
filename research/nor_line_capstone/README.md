# NOR research capstone

The [paper](../../papers/authorized-exact-computation/paper.md) consolidates the exact-computation branch of the Numerical Orbit Research series and its information/authority background. Its archived results remain attached to their original studies. NOR-60 closes the registered execution sequence with all five gates passed and a combined signed-verifier workload ratio of 0.563961. All 60 exact output vectors and indexed roots are preserved.

## Read the evidence

The [evidence ledger](evidence_ledger.json) records the historical source pin, study-level timing scopes, selected ratios, first outcomes, and SHA-256 locks for every referenced report and archive. [CSV](evidence_ledger.csv) gives a compact analysis table. The [claim map](claims.json) states the evidence type and assumptions for each central claim.

The historical reports through NOR-59 are pinned to commit `714a7f5e1203289ef053dfa5becaa3537d1b946b`. NOR-60 is bound separately to registration `363ef19abeba68e2897c593697ad56e2aaf4fefc`, trigger `a7edfa6850206fc01a9a67650e09ddd31d198766`, and first run 44327994013. Ratio denominators differ between studies, so the selected values describe their own matched comparisons. Original failures and the unscored NOR-57 first run remain in the ledger.

## Audit without generating a new experiment

From the repository root:

```bash
python research/nor_line_capstone/audit_evidence.py --out /tmp/nor_capstone_byte_audit.json
python research/nor59_authorized_merkle/original_archive_audit.py
python research/nor60_authorized_interpreter/original_archive_audit.py
python research/nor57_authenticated_router/original_failure_audit.py
```

The capstone audit checks the byte locks in its ledger and exits nonzero on a mismatch. The study-specific auditors reconstruct their archived checks from the original raw receipts. Each writes only its requested audit output and leaves the first-run evidence intact.

The recorded seed streams and signing keys are public test fixtures. The measurements concern finite synthetic exact-integer workloads. Existing MathCan proofs are conditional on declared execution and cryptographic assumptions; the concrete support-compression and static-tree correspondence arguments in the paper retain their stated proof status.

## Reusable implementations

[Authorized Exact Gym](../authorized_exact_gym/README.md) packages the signed verifier task and public development fixtures. [Method transfer](method_transfer.json) records the source-pinned EvalMath FRE-01 auditor and Control-Harness adapter, with their finite test evidence. Their authority and timing premises remain explicit.
