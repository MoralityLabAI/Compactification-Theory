# Game-design webhook contract

`WebhookDesigner` POSTs one bounded UTF-8 JSON object to the configured broker.
Set `PLUTO_GYM_DESIGN_URL`, `PLUTO_GYM_DESIGN_MODEL`, and optionally
`PLUTO_GYM_DESIGN_TOKEN`. The effort hint defaults to `high`; the broker maps it
to its provider. HTTP is accepted on loopback, and remote endpoints require
HTTPS. Credentials belong in the authorization header. Redirects are rejected.

The `pluto-gym.design.v1` request includes a content-derived `request_id`, also
sent as `Idempotency-Key`, plus offered candidate variants and training-only
scores. For synthetic candidates the payload contains a validated `RuleSpec`.
Native proposals use the four named TensorCraft mod variants and a bounded
description. Native proposal support is a protocol capability; automatic native
curriculum training awaits its action/learning bridge.

The broker must return HTTP 200 and `Content-Type: application/json` with exactly:

```json
{
  "schema": "pluto-gym.design.v1",
  "request_id": "copy-the-received-request-id",
  "variant_id": "one-offered-variant-id",
  "rationale": "Why this candidate should help the next training block."
}
```

A reply cannot set a path, launch command, interface permission or training
budget. Its `variant_id` selects an already validated candidate. Rejected or
timed-out calls consume the call budget. The client performs no automatic retry.
The rationale is retained for audit and is never executed or parsed as a command.

Default host limits are four calls per run, 32 KiB request bytes, 16 KiB response
bytes, 800 requested output tokens per call, and 25 seconds per HTTP worker.
The subprocess timeout covers network waiting, including slow headers and
trickled bodies. Process startup/termination contributes additional overhead.
The full training loop uses a separate cooperative wall-time budget checked
between phases. Provider-side token/currency limits remain broker obligations.

A broker authenticates the request, checks its budget and idempotency key,
passes offered variants and training scores to the selected model, parses the
structured choice, and returns the four allowed fields. Keep provider credentials
inside that service. The same endpoint contract supports a hosted small model
and a local quantized model.

The offline control chooses the largest exact training interface regret.
Its receipts record zero network calls and zero billed cost. The suite includes
local mock-broker tests; socket restrictions caused those cases to skip during
this build. A separate real subprocess test verified timeout termination.

## Later designer training

Each successful curriculum step appends a `pluto-gym.designer-trace.v1` record.
It contains the sanitized prompt, validated response and post-update training
feedback. Held-out scores stay in the experiment receipt. The trace marks its
feedback as training-only and excludes a confirmed-benefit label.

For proposer distillation, split records by game family and mutation lineage
before fitting. Preserve failed proposals as validation examples with their
original status. A later evaluator can attach independently measured benefit
under a frozen protocol; that attachment must retain original request and
candidate hashes. Loading a 1-bit inference checkpoint alone does not specify
how designer weights or adapters will be trained.

