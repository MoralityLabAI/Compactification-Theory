# DISTILL-04: free Inkling Mini evaluation pilot

The live lane requests only the exact Tinker free-tier sampling model:

    thinkingmachines/Inkling-2-Mini:peft:65536:free

Pricing source (checked 2026-11-27): https://thinkingmachines.ai/tinker/pricing/ . Free sampling is subject to throttling and prepaid positive-balance eligibility. The operator must verify account eligibility and current provider terms. The paid base model identifier is used only to select tokenizer/renderer metadata; never as a sampling target. No fallback, training, checkpoint creation or Fable calls. The runner is single-request sequential. SDK-internal retries and provider billing are outside local control.

The experiment is a **reference/evaluation** pilot. It does not train or promote an adapter. All 30 generated cases (ten families, three difficulties) and an optional one five-question design section from each of 25 adapters form a deterministic roster. Foundation, standard and frontier case sizes vary. Design sections are partial examinations. The model receives only the public packet; the exact oracle runs after the response is returned. The original public seed namespace is visible and is not a sealed confirmation test.

Use the existing DataGoblin Tinker SDK and renderer pattern (Agonist/tinker_live.py at c06ac62fca6a4b8a6b8a69bba65a6435b463d8dc, blob 3c58ced49e2721ab519da0f21d04b39710360edc). On the authenticated operator host:

    python -m pip install tinker tinker-cookbook
    python -m Distill.inky_free prepare --out runs/inky-pilot --mode live --per-family 3 --include-design
    python -m Distill.inky_free run --campaign runs/inky-pilot --confirm-free-tier --limit 1

The first request verifies the actual live provider and renderer path. Use the existing environment-managed TINKER_API_KEY. Do not put a key in a command, repository, prompt, or result log. After confirming the returned model, actual billing and the receipt, run a further finite batch:

    python -m Distill.inky_free run --campaign runs/inky-pilot --confirm-free-tier --limit 54
    python -m Distill.inky_free replay --campaign runs/inky-pilot

Context: 12,288 rendered input plus maximum generation, hard cap 16,384, output cap 4,096 tokens (includes reasoning tokens if returned by SDK), effort .9 and temperature .3. A prompt exceeding the rendered cap stops before sampling. The actual model weight revision is unknown unless the provider supplies and the operator verifies it. The receipt records the requested model ID, renderer, SDK/Cookbook versions, token counts and latency; no invoice is inferred. Only visible assistant text is retained; reasoning/tool blocks and raw token IDs are not stored. Unclean termination, invalid JSON, exact wrong answers and abstention are separate outcomes. Failed requests remain pending, including uncertain network/timeouts. No app-level retry, no automatic release, no silent reissue. Reconcile pending requests with provider before operating further. A STOP file prevents new requests; it cannot cancel a submitted call. One pending claim at a time via SQLite. A completed campaign can be replayed without calls. Keep its journal and prompts private and do not commit live outputs.

Offline fixture test (no credentials, no provider imports):

    python -m Distill.inky_free prepare --out runs/inky-fixture --per-family 3 --include-design
    python -m Distill.inky_free run --campaign runs/inky-fixture --limit 55
    python -m Distill.inky_free replay --campaign runs/inky-fixture

Fixture answers are deliberately supplied by the exact local oracle; their pass rates are engineering checks, not model competence. The mock SDK tests the exact free model routing, context cap, single sample and text-only extraction without contacting a provider. Live SDK compatibility and actual free-tier rate remain unverified until the operator smoke.

## Semantic review

Exact case grading assesses the requested final answer; it does not establish the correctness of the worked derivation. Independent reviewers use a versioned rubric with four dimensions scored 0/1/2: mathematical validity, checkability, assumptions and scope, explanatory value. A reviewer must attach an evidence note for each answered question. Review records bind the packet and exact response digest. No automatic prose scoring, model-as-judge or training admission.

    python -m Distill.review template --campaign runs/inky-pilot --packet-id PACKET_ID --out review.json
    # Fill reviewer, scores and evidence from the public question and visible worked answer.
    python -m Distill.review validate --campaign runs/inky-pilot --packet-id PACKET_ID --review review.json --out review-receipt.json

The template and journal are local; reviewer identity is claimed, not cryptographically authenticated. Fixture records cannot be submitted through the live review CLI. Review exact failure categories separately from semantic scores and keep all adverse outcomes. Before any training use, verify applicable terms and obtain necessary rights. Fable output is excluded from the pilot.
