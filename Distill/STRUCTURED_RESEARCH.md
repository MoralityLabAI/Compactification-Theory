# DISTILL-12B / Fable structured research output preflight

The original 24-packet Fable archive and strict results (7/24 valid, 1/12
exact) stay immutable. Post-hoc diagnostic 12/12 accuracy is separate.
Run a new experimental condition using a closed JSON schema with fixed
question-ID keys. This is an evaluation transport, not model-output
training.

Reference: https://platform.claude.com/docs/en/build-with-claude/structured-outputs

    python -m Distill.structured_research prepare --packet PACKET.json --out runs/wire-preflight/PACKET_ID

Use only prompt.txt and format.json as model input. The authenticated host
sets output_config.format to the parsed format.json and retains its calibrated
effort setting. The schema uses closed objects, required fields, basic
types and nonrecursive unions. The adaptive policy is transmitted as a flat
binary-path node list because recursive schemas are unsupported. The local
adapter checks exact paths and reconstructs the native policy tree without
inferring any missing nodes, changing an answer or normalizing a mistake.
Provider grammar compilation and SDK compatibility still require live
verification. Structured output can still fail due to truncation or refusal.

Select up to three previously invalid decision packets across distinct
families, using only public packet IDs. Include an adaptive case if present
and a nonadaptive case. Use the same model and calibrated generation policy
as the previous run to the extent the budget permits. Verify identity, rates,
and permitted use. Reserve each request through the existing canonical
global ledger, including thinking tokens and extra schema prompt overhead.
Total incremental authorization ceiling is $3; stop before any request
whose reserved maximum would exceed it. Do not reset the ledger or assume a
third request fits. Check actual usage and provider receipts. No automatic
retry and no fallback. Save the original raw response and stop reason.

After freezing all new raw attempts, convert each clean response:

    python -m Distill.structured_research convert --packet PACKET.json --raw RESPONSE.json --provider anthropic --revision MODEL_ID --run-id PROVIDER_RECEIPT --out runs/wire-converted/PACKET_ID

Raw bytes, schema digest and converted v2 response are independently bound.
Run the existing exact grader against the converted response, and review
prose separately. Report planned, attempted, clean, schema-valid, converted
and strict exact counts for the new wire condition. A malformed response or
missing policy node fails; keep the raw attempt. Do not pool these scores
with the original direct-JSON benchmark. No paid calls are implemented by
this module. No provider outputs enter a training dataset.
