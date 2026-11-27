# DISTILL-02 / v0.2.0

The public design examination library now includes a byte-pinned offline registry of fourteen adapters in four repositories. Nine cross-repository sidecars are copied byte-for-byte from the pinned public PR heads; the registry retains their original repository, full commit, path and Git blob. The five local adapters are unchanged. Registry checks authenticate equality to the recorded adapter blob, not the truth of every source claim or the origin of arbitrary external inputs. Keep source and model rights review separate.

## Difficulty and modular sections

- foundation: minimal examples, prerequisite definitions, explicit assumptions.
- standard: registered-scope derivations and reproducible checks.
- frontier: extensions, counterexamples and two additional obstruction/redesign questions.
- quick/full control common-question coverage. Each difficulty retains the full algorithm step sequence and domain-specific probes.

`split_quiz(packet, max_questions=5)` builds separately answerable sections with deterministic parent packet, ordinal, count and section digest. Sections keep the public context and the question fields. Foundation and standard use the same question IDs with different difficulty instructions; they are separate examinations, not independent statistical roots. Frontier adds two questions. The fixture cases remain fixed until the owner supplies independently generated public cases. No claim of novel problem diversity from repeating a form.

The original v0.1 packets remain readable. New packets add difficulty and optional section metadata.

## Offline inventory and cost envelope

From the repository root:

    python -m Distill.campaign_cli inventory
    python -m Distill.campaign_cli plan --out runs/distill-200 --budget-usd 200 --reserve-usd 10 --input-usd-per-million 4 --output-usd-per-million 20 --input-cap 6000 --output-cap 5000 --section-size 5
    python -m Distill.campaign_cli status --campaign runs/distill-200

Rates above are illustrative operator-supplied inputs, not verified provider pricing. Verify the specific model, batching, output/thinking accounting, minimum/extra fees and credit eligibility before authorizing a live run. The plan is provider-neutral and makes zero requests. It writes separate packet files and a hash-bound plan. At these assumed prices the maximum modeled cost of a request is $0.124 (6k input + 5k total billed output). The planner allocates a round-robin prefix across gym/difficulty sections under the available budget; omitted requests are counted. A tokenizer or provider count must independently establish actual input size before reservation; the cap is not an estimate of actual usage. If a prompt exceeds the cap, do not submit it. The output cap must include billed reasoning and all generated tokens, enforced by the external operator.

## Durable request accounting

After independently counting the entire provider request (including wrappers):

    python -m Distill.campaign_cli reserve --campaign runs/distill-200 --request-id PACKET_ID --counted-input COUNT
    python -m Distill.campaign_cli settle --campaign runs/distill-200 --request-id PACKET_ID --input-tokens BILLED_INPUT --output-tokens BILLED_OUTPUT --provider-receipt UNIQUE_PROVIDER_ID

The SQLite ledger binds the plan digest, checks all packets and reserves full modeled request cost atomically. Duplicate reservations and repeated settlements fail. A crash or ambiguous provider outcome leaves the reservation in place; no automatic retry or release. Settlement records actual provider-reported usage as supplied by the operator and freezes new reservations when a cap is breached. The ledger models the supplied rates and cannot prevent external billing, duplicate submission outside the process, or undocumented fees. Invoice reconciliation and a trusted provider adapter remain operator-side. Raw provider answers should be stored separately; no automatic provider calls or training export.

## Native grading

    python -m Distill grade --packet PACKET.json --response RESPONSE.json --native-checkers --out RECEIPT.json

The first two native integrations are Pluto's pinned deadline capacity reference and COE-01's exact chromatic coloring implementation. Source bytes and design document bytes must match the reviewed pins. The native answer agrees with an independent finite toy oracle. Only the concrete final answer is graded; prose remains pending. Other gyms retain their source-owned verifiers and host callback interface, with no claim of native integration yet.

All output receipts retain reference-only custody and training_admission=false. Review applicable provider terms and obtain any required permission before using provider output for training.
