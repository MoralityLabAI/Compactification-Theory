# DISTILL-03: seeded mathematical examination cases

Version 0.3.0 adds eleven locally authored design adapters pinned to public design documents or code in six further repositories. The registry contains 25 adapters across ten source repositories. The new adapters live in Distill/registry_new rather than the source repositories. Existing gym code and evaluation data are unchanged.

The generator covers Pluto deadline capacity, COE graph coloring, RLT arithmetic, AUX modular recurrence, EvalMath formal reciprocals, MathCan Lagrange identity, MoLT strict-prefix recurrence, ASM Bayesian decisions, independently anchored parity and HG-07 GF(2) query identifiability. These are finite public-development exercises inspired by the sources, not native gym episodes or held-out evaluations. Size and range depend on difficulty. A SHA-256 counter stream and content hashes support reproducibility and deduplication within the generated roster. Seeds are public reproducibility metadata rather than blinded roots.

Each case has three question slots: an exact answer with derivation and verification, a general mechanism explanation, and a changed-premise witness. The grader reconstructs the packet from the adapter, seed, ordinal and difficulty. It independently checks the final answer; explanation review remains pending. For mathematical ambiguity, answer the question with JSON string "abstain"; ordinary response abstention is a separate status. Preserve the exact source and model metadata.

    python -m Distill instance --gym HG-07 --seed 20261127 --ordinal 4 --out /tmp/hg07.json
    python -m Distill prompt --packet /tmp/hg07.json
    python -m Distill grade --packet /tmp/hg07.json --response /tmp/hg07-answer.json --procedural-checkers --out /tmp/hg07-receipt.json

Offline campaign with illustrative operator-supplied rates:

    python -m Distill.campaign_cli plan --out runs/distill-200-v03 --budget-usd 200 --reserve-usd 10 --input-usd-per-million 4 --output-usd-per-million 20 --instances-per-gym 96 --instance-seed 20261127

The default remains design-only. The augmented planner round-robins design and generated case queues. With section size five the ten generators contribute 960 candidate case requests. Budget omissions are explicit. Distinct cases need not be independent statistical samples. Costs are illustrative ceilings, not actual billing or verified provider prices. Existing v0.2 packets and plans remain parseable; new plans use distill.batch_plan.v2. Prior ledgers remain bound to their own plan IDs. Do not use multiple ledgers to spend the same provider balance twice.

No live transport, paid requests, training use, or automatic training admission. Applicable provider terms and rights must be verified before any training use. Reference-only custody is preserved. Native verification of the original gym remains distinct from these finite synthetic checks.
