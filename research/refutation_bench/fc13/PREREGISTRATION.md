# FC-13 preregistration: hard-deadline original-Lean transfer

**Registered 2026-11-14 before FC-13 heldout selection, compilation or model evaluation.**
This is a separate experiment from the completed FC-12. FC-12 source
selection, model weights, scored results and replay audit remain immutable.

## Research question

Does the existing FC-12 parameterized goal-state TypedLoop generalize to
genuinely fresh original MathCan source families against a Lean-native search
baseline when both policies obey *the same hard wall-clock cutoff*?

The first FC-12 evaluation reached 5/16 verified proofs for both model seeds
and native search. Two native positive results finished 7–29 ms after the
model-derived per-case time allowance because its deadline was checked only
between subprocess calls. FC-13 makes that boundary part of the Lean process
timeout itself and counts **no proof completed beyond the declared deadline**.

This is a transfer and verifier study; it adds no new training and changes no
model parameter, action generator or previously registered FC-12 result.

## Immutable source and held-out families

Original unmodified `patrickdugan/MathCan` commit
`c84e2025334e9a967c1297425a490f98d5b3d189`, compiler
`leanprover/lean4:v4.35.0`. The test selector consists of **the first four
top-level theorem declarations, in original source order, from exactly these
four previously unused FC-11/FC-12 imported source modules**:

- `MathCan/HookMultiTick.lean`: `distribute_sound`, `three_account_conservation`,
  `before_swap_exact_input`, `specified_credit_restores_debit`.
- `MathCan/Settlement.lean`: `marked_total_conserved`, `exact_five_account_partition`,
  `terminal_pool_floor_budget`, `certificate_gap_identity`.
- `MathCan/TOEJordan.lean`: `metric_vanishes_at_double_root`,
  `mixed_vanishes_at_double_root`, `metric_derivative_vanishes_at_root`,
  `rank_one_root_kernel`.
- `MathCan/TOENonAbelian.lean`: `flavor_exchange`, `signed_color_norm`,
  `signed_color_dot`, `mixed_under_flavor_exchange`.

The module set was identified from the immutable MathCan tree and declaration
names before evaluating any FC-13 policies. FC-13 uses no FC-12 held-out
theorem and none of these four modules entered the FC-11 or FC-12 training
selectors. The source proof bodies remain unchanged; original reference
proofs compile separately as the 16/16 oracle. The reference scripts are
not available to either controller.

Frozen train-only source version and genuine tactic-effect semantics
remain from FC-12:

- train-only source revision `8d192f0b323eade0fc764611f420414d70a0d81b`
- 51 real train states SHA256 `403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0`
- 961 actual compiler candidate effects semantic SHA256
  `bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec`.

## Frozen student and native comparator

Use the existing FC-12 four-loop 62,401-parameter `TypedLoop`, 12 tactic
families and local-context script generation with up to four candidate
scripts per family. The student weights are read-only from the successful
training CI `43547762837`. Freeze the two checkpoint digests:

- seed17 `55bc16117e045279ba0e723c171e45b4fe05b4c6eb8a1b9709dd1b59cc9987a2`
- seed29 `0e11d74cf1be7889caf824671099d052884dc3722211e773dd126d233671a805`.

Native symbolic baseline: exactly the same generated scripts per state, tried
in the FC-12 registered family order:
`omega, rfl, simp_all, assumption, intro, constructor, exact_local,
apply_local, rw_local, rcases_local, induction_list, simpa_local`. It can
respond to compiler feedback exactly as the student can. It cannot read
student logits, selected actions, reference proof, or gold labels.

## Mandatory hard-deadline search and proof gate

For **both** student and native controller:

- Fixed per-theorem **20,000 ms** wall-clock deadline, starting before first
  inference or candidate Lean subprocess.
- At most 3 successful goal transitions; at most **12 Lean subprocesses in
  total**, including untrusted scratch observations, failed candidates, and
  the final separate no-sorry certificate call. The 12 are not 12 pairs.
- Every scratch call receives a process timeout no larger than the
  *remaining deadline*, also capped at 8 seconds. Runtime overhead is
  measured after each call. Expiry stops search and is recorded. A
  subprocess ending beyond the deadline is never counted as certified.
- A candidate is first tried as an untrusted Lean scratch check, with
  `trace_state` and explicit `all_goals sorry`. It counts as a progressing
  transition only if Lean accepts it and its remaining compiler goal changes.
  Any goal count >1 produces explicit `multigoal_unsupported`.
- If scratch closes the goal, the **entire same original theorem** must
  compile again in a separate Lean subprocess with the exact proposed
  script and **no `sorry`** before a proof can be certified. This counts
  toward both deadline and subprocess budget.
- Every candidate must belong to the immutable FC-12 argument grammar,
  which is instantiated solely from the actual compiler-visible telescope.
  The subject's original proof steps cannot drive candidate selection.
- Record all attempts, exact generated source SHA256, compiler outcome,
  source theorem, family, action, time, number of goals and deadline
  status, including failures. Per-theorem elapsed time includes inference,
  all subprocesses, setup, failed attempts and final checks. Store both
  controllers' trajectories as separate JSONL outputs.

A proof is valid only if the second, independent exact full theorem compile
finishes **at or before** the 20-second deadline and has returncode zero
with no `sorryAx` or compiler errors. There is no post-hoc deadline clipping.

For robustness and auditable inference boundaries, the final tests must
also include synthetic Lean declarations with a short proof, a failing
candidate, a two-goal branch, budget exhaustion, and a deliberately delayed
fake Lean executable that exits successfully *after* the remaining deadline.
A stopped process must never release a certificate.

## Fixed outcomes

Primary: number of the **16 original held-out theorems** with exact
sorry-free Lean certificates for seed17 and seed29 versus native, at the
identical **20-second** deadline. Report names and whole results, with
successful, failed, and timeout classifications. The registered success
criterion is **strictly more certified original theorems than native for
both seeds**, with zero false certificates.

Secondary: total measured wall-ms, per-theorem candidate calls, and
milliseconds per certified theorem, with no imputed zero cost for failures.
Compute efficiency succeeds only if cost per certified theorem is lower
for both seeds and primary learning success also holds.

The original source reference proof oracle must compile 16/16, separately.
A CI failure, strict mismatch, timeout or parser error yields a reported
failure/abstention and cannot be repaired by changing the selected
heldout cases. After scoring, any development changes require a new
named post-hoc analysis or FC-14, never a replacement FC-13 score.
