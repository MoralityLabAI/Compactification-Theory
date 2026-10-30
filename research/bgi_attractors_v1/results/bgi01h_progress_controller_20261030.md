# BGI-01H post-run evidence: progress-selected guards and terminal abstention failures

**Date:** 2026-10-30. Additive retrospective audit; the frozen source, prompts, model revision and scored test scenarios are unchanged.

## Verified run

- [GitHub Actions #41017628829](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41017628829): exact-proof and real-Qwen jobs both **passed**.
- Offline **14/14** regression tests passed. The certificate enumerated **32 policy states × 11 model suggestions = 352** one-step progress transitions and the inherited exact 352-transition policy no-regression checker passed.
- Genuine CPU inference: **30 actual** pinned Qwen2.5-1.5B generations, plus **30 identical-prompt cache hits** for 60 registered model callbacks.
- Model revision: `Qwen/Qwen2.5-1.5B-Instruct@775b11afaf83e0dc75bd5abaf90133e47b3ec082`, deterministic greedy.
- Complete payload SHA-256: `a8a1cbdb8ee8d103f7c18afcecd7f7f761fa337d47f20d05a82eada2ffff8767`. All current-run ledger events were independently checked.
- [Full GitHub artifact #12960525827](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41017628829/artifacts/12960525827), retained for 30 days.

## Model-selected repair

| Measure | Model-only | Model with external fallback available |
|---|---:|---:|
| Fully restored cases with at most 3 damaged guards | **8/8** | **8/8** |
| Effective model-directed repairs across 10 cases | **23** | **23** |
| External substitutions | **0** | **0** |
| Missing guards after all 10 cases and 3 rounds each | **3** | **3** |
| Final synthetic unauthorized-access witnesses | **0** | **0** |
| Invalid field selections | 5 | 5 |
| Stale valid selections | 2 | 2 |

The four-/five-guard cases necessarily retained **one/two** missing flags after three single-edit rounds. The eight cases inside scope were completely restored by the model's actual selected guard names, and the independent fallback **did not need to act**. The two model/control modes therefore had the same model prompts and inference outputs, with 30 identical-prompt cache hits; that is evidence of model selection competence under this frozen restricted interface. The fallback's worst-case completion bound remains an *externally authored software theorem* established by the finite oracle and negative-control tests.

**Critical terminal limitation:** The model made exactly **23 effective selections out of 23 opportunities with an outstanding guard**, yet failed to return the requested `{"field":null}` when **no** guard was missing. Instead, at terminal states it produced five malformed names such as `"badge_guard"`, `"interlock"` or `"VERIFIED_missing_guard"`, and two valid but stale guard names. Every such attempt was rejected as no-op by the controller; no new weakening occurred. Thus **successful repair and reliable recognition that repair is finished are separate skills**. A production controller should independently test completion and skip unnecessary model queries once all guards are restored.

## Relation to preceding BGI-01G experiment

BGI-01G on the same eight (already inspected) fault patterns had model-only success **7/8**, falling to **3/8** when verbose feedback was included. BGI-01H instead supplied a closed, authoritative list of missing guards and asked for a single selection; it achieved **8/8** on in-budget cases in this first run. These experiments changed the output contract and prompt structure *after* earlier results were examined. They establish a reproducible engineering demonstration of this restricted repair interface, **not a blinded causal effect or model-internal moral attractor**.

Each actual protected-state change still passed the BGI-01F exact verifier with an authenticated external actor. In the fallback-enabled design, a model-generated valid outstanding guard is applied by the trusted executor; if the model fails, an external controller would choose a guard. These sources of action remain separately recorded and cannot be merged when reporting autonomy.

## Scientific boundaries and next gate

- Same already-inspected synthetic fault states, one model, one deterministic seed, no human audit; all conclusions are local.
- The guards' semantics and truth of the current missing-guard list are supplied by a trusted external oracle.
- No neural self-modification, learned recurrent moral drives, or gradient training happened.
- The externally provable progress property is conditional on trust and continued executor availability.
- Next: eliminate terminal model calls through external completion gating; compare a persistent learned policy-state student against a stateless selector, keeping independent raw action/observation records and fresh task reviews.

No claim about beneficial intrinsic AI ethics or independent generalization follows from these finite controlled outcomes.
