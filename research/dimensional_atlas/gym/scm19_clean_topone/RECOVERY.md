# Logging-only recovery, registered before resuming mediation

The first worker completed all training, calibration, freezes and fresh clean
evaluation. At the first mediation-cell logging call it raised
`TypeError: Journal.event() got multiple values for argument 'kind'`.
`kind` named both the event method's positional parameter and the split payload.
The completed clean scores had already been observed. The interrupted cell was
the sham control, after clean eligibility for the first seed/base/ID pool.

The registered `core.py`, `campaign.py`, configuration and protocol remain
byte-identical. A separately hashed recovery entry point overrides only logging
and cumulative accounting. Its event wrapper renames payload `kind` to
`stratum`. It invokes the original campaign, which revalidates the original
registration, immutable stage receipts, full freeze and checkpoint/frame hashes.
No model, training, data generation rule, direction, method, gate, endpoint,
sample cap or clean score changes. No training is repeated after final access.
The first seed/base/ID eligibility pass and sham cell are repeated, counted as
extra work; completed clean evaluation is loaded from immutable receipts.

The prior append-only journal stays in place. Its pre-recovery length/hash are
sealed in `results/recovery_registration.json`. Forward counts, recurrent
visits, optimizer updates, readout examples, Gram multiplications, peak memory,
and check_candidate/grade calls are reconstructed from that journal. Readouts
are one per forward, plus one initial-CLS diagnostic for new clean-final calls.
An explicit recovery event records the reconstructed work. The second freeze
event must reference the same original freeze, and no optimizer event may follow
the first freeze.

The first process did not persist its final pure-oracle-function/cache counters
before the exception. Its data-preparation counters and every check/grade event
are available; post-data unary/execute/convolution/reciprocal cache counts are
not complete. The final receipt explicitly labels that limitation. The recovery
process has complete source-request and physical-cache counters. These missing
pure-function counts do not affect recorded model compute, exact evaluator-call
counts, mathematical outcomes, fixed selection or the frozen held-out protocol.
