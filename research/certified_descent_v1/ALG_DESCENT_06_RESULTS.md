# ALG-DESCENT-06: exact interned IDs and cache efficiency

See [DataGoblin PR #89](https://github.com/MoralityLabAI/DataGoblin/pull/89), protocol commit f6836f5. C++17 synthetic public-witness routing. Fresh 56 roots, 2,072 root/arm cases, 209,272 timed executions; 14,504 independent Python receipt fields, 672 equal-slot comparisons and ASan/UBSan replay agree. All cases closed.

Exact root-local representative uint16 fits existing Node padding (56 B unchanged, alignment 8). Cache entry 36 -> 12 B, 3x slots at matched entry allocation; controls 32 B and pending stack 104 B unchanged. OOD banks 22,592 nodes, 5,025 root-local distinct keys; persistent bank not deduplicated. Temporary sort vector peaks 2,352 B; conservative sidecar if no padding 600..2,352 B/root. No process-level memory saving claimed; max native RSS ~23 MiB including banks. ABI mismatch rejected.

OOD corresponding LRU at 72 B: work 921.31 -> 864.53 (-6.16%, 29 wins/3 ties/0 losses), mean root-median warmed time 2.719 -> 2.410 us. At 108 B: 904.72 -> 850.50 (-5.99%, 27/5/0), time 2.643 -> 2.371 us. Corresponding-policy gate passes. Against strongest packed control, best ID work gains 5.76/4.48/3.64% at 36/72/108 B, so no two-capacity 5% gate against strongest packed policy. Pending-policy gate fails. Large ID arrays increase scan time. Equal-slot work and all receipt fields are identical.

Mean interning 39.86 us/root (excludes post-check and destruction); approximate amortization ~129 repeated rollouts at 72 B LRU by ratio of means, ~147 at 108 B. Native load 62.55 ms, Python bank generation/export ~3.46 s, excludes cold speedup claim.

MMS direction: use exact verified state IDs, limit scans to occupied slots, and charge preprocessing per reuse horizon. Cache closure identity and route cost have distinct semantics; two equivalent polynomials may have different candidate banks. Preserve certificate authorization/version and field scope. Risk-family BPD comparison remains prospective. No neural, native MMS/BPD, general proof-search, factoring or DLP claim.
