# Paid research shortlist: bounded operator workflow

The full proposed budget is $200. Start with an active $80 envelope, retaining
$120 for the other research/criticism tracks and final reserve. An operator
may explicitly choose a different active envelope up to $190; at least $10
always remains reserved. These are local caps, not a provider account balance.
Use one canonical ledger; never copy or recreate it to spend the same credit
again. Coordinate all other API spend outside this ledger against the same
overall $200 authorization.

After teacher triage, use the exact returned shortlist. Verify model ID,
current input/output rates (including all billed reasoning/output tokens and
fees), and permitted output uses before planning. The CLI requires operator
evidence strings and explicit rates; it never guesses Anthropic prices.

    python -m Distill.research_paid plan --help

Supply --triage, --neon, --holdout, --out, --model, --rate-evidence,
--terms-evidence, --input-usd-per-million and --output-usd-per-million.
The default caps are 8192 input and 8192 output tokens per request. Adjust
them to provider-counted prompt lengths and verified output needs. Keep
both sections of each case; truncate the shortlist only at whole-case
boundaries. No clean cases are added to fill the budget. No provider call
occurs in this command.

The existing ledger interface now accepts these research-specific plans:

    python -m Distill.campaign_cli status --campaign runs/fable-paid
    python -m Distill.campaign_cli reserve --campaign runs/fable-paid --request-id PACKET_ID --counted-input COUNT
    python -m Distill.campaign_cli settle --campaign runs/fable-paid --request-id PACKET_ID --input-tokens COUNT --output-tokens COUNT --provider-receipt RECEIPT

The host must enforce the requested output cap and send only public/ prompts.
Reserve before submission. Save original visible JSON answers outside Git.
Reconcile usage against provider receipts/invoices and settle each request.
Unknown calls retain reservations, duplicate requests fail, and cap overruns
freeze the ledger. No automatic retry, paid API client or model fallback is
provided. Local supplied-rate calculations cannot enforce a remote account
cap or prove provider identity. Stop if provider pricing or billing changes.

When every planned request settles within caps, archive one verbatim JSON
file per full packet ID in a separate response directory (empty file for
an omitted answer). Freeze the whole batch before showing any exact grades:

    python -m Distill.research_paid freeze --campaign runs/fable-paid --responses runs/fable-raw
    python -m Distill.research_paid grade --campaign runs/fable-paid
    python -m Distill.research_paid review --campaign runs/fable-paid --directory runs/fable-audit --create

Complete independent review templates and repeat the last command without
--create. Exact checks, review scores and provider evidence are separate.
This plan is reference-only. Training admission remains false, including
for paid provider responses. Actual data use needs explicit terms review
and native DataGoblin admission. No hidden/private reasoning collection.
