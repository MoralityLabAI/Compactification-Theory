# JSON wire contract v1

UTF-8 JSON, unique object keys, finite numeric values, 1 MiB canonical size bound. Python is the canonical encoder for v0.1. Do not assume another JSON implementation produces byte-identical floating-point or Unicode encoding.

## Adapter

See the five registered distill_exam.json files for complete examples. Required keys: schema_version (distill.adapter.v1), gym_id, gym_version, source (repository, commit, path, blob), objective, context, steps (ordered id/name/description objects), packs, specific_questions (id/prompt objects), example_case, case_question, native_verifier. Both case fields are null for design-only adapters. Source commit and blob are full Git SHA-1 identifiers; the adapter records the design source, not the current code revision of an arbitrary runner.

## Packet

The compiler emits distill.packet.v1, library_version, pack_version, profile, gym_id, gym_version, source, adapter_sha256, purpose, objective, context, public_case, questions and packet_id. Each question has id, category, prompt and ordered fields. A case is null or contains exactly classification (public_development), case_id, kind and data. Only explicitly reviewed public inputs belong in packets.

## Response

    {
      "schema_version": "distill.response.v1",
      "packet_id": "COPY_THE_ACTUAL_PACKET_ID",
      "model": {"provider": "fixture", "revision": "fixture-v1", "run_id": "run-001"},
      "answers": [
        {"question_id": "core.ontology", "status": "answered",
         "fields": {"claim": "...", "derivation": "...", "check": "...", "limitations": "..."}},
        {"question_id": "core.first_principles", "status": "abstain",
         "fields": {"reason": "insufficient premises"}}
      ]
    }

This is an abbreviated illustration, not a valid complete response. Every compiled question must be present exactly once. Answered fields must match that question exactly; abstention has only reason. Values are nonempty strings. Step questions require what/how/why/invariant/failure_and_check. Case questions require answer/worked_solution/check/assumptions. Model metadata is claimed until the external runner binds it to provider records.

## Receipt

distill.receipt.v1 includes packet_id, response_sha256, claimed_model, custody (reference_only), training_admission (false), checks and scope. Each check has question_id, answer_status, exact_check (not_checked/pass/fail/abstained/checker_error), explanation_review (pending/not_applicable). No automatic correctness claim for free-form derivations. Only the host supplies native scorer callables, never the model packet. Errors remain failures of evidence, not automatic labels. Archive raw responses separately, with model/run/spend provenance and any verified rights; keep sealed material separate.
