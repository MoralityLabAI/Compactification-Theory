# FC-04: adaptive recurrence with a fresh certified Horn test

[Preregistration](PREREGISTRATION.md) was committed in
`ce7190129617d8387a7acf4ceed3e8d0570cfce4`, before creating the
generator or inspecting FC-04 model scores.

The corpus is intentionally separate from the earlier historical-paper
RefutationBench cases and the FC-02/FC-03 graph families. It has 384
fully checked finite Horn entailment tasks.

| Split | Depth | Count | Valid | Invalid |
| --- | --- | ---: | ---: | ---: |
| `test_same` | 2–4 | 192 | 96 | 96 |
| `test_deep` | 6–8 | 192 | 96 | 96 |

Each test uses the four new families `relay_deadlock`,
`asymmetric_fork`, `stacked_join`, and `gated_detour`, with 24
valid and 24 invalid cases per family per split. Structural fingerprints
must be distinct from all 1,536 FC-02 and all 384 FC-03 problems.

The exact verifier checks 192 complete negative Boolean witnesses.
The 192 positive theorems compile as explicit parameterized proofs
under Lean 4.19.0. Their source and independently regenerated artifacts
are [checked by CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43360749033).
Two `#print axioms` endpoint checks report no extra axioms.

Test generator source frozen for model evaluation:
`1b85412aa237d7522341c40d029654052e1d3267`

Canonical test JSONL SHA-256:
`08e57752f02c886e0947e36687c0572d84cfc815a467e1ea0c2554a1a044c04c`

Generated Lean source SHA-256:
`794480e4427210fc5250016a629565f5b32b1d0d4d52002f651a7aa6c1aa8158`

From repo root:

~~~sh
python -m unittest -v research.refutation_bench.tests.test_fc04
python -m research.refutation_bench.fc04.generator \
  --out /tmp/fc04_cases.jsonl \
  --lean /tmp/FC04.lean \
  --receipt /tmp/fc04_receipt.json
lean /tmp/FC04.lean
~~~

## First frozen-student outcome

The [first preregistered FC-04 results](../results/fc04_adaptive_20261113.md)
show that the validation-trained early-stopping policy met its registered
criteria against the fixed 12-pass baseline for both seeds and both test
depths, with 91–96 exact negative certificates per 96 invalid test cases
and mean compute of 5.0–8.4 recurrent passes. The long-form receipt includes
provisional-positive errors, abstentions and cases where fixed4 is
more certificate-efficient. These numbers are specific to public
finite Horn logic and the registered comparator.

The model trial is implemented in the private `LoopedTransformers`
repository, which trains and selects a stopping threshold on the earlier
FC-02 corpus. The training GitHub Actions job never checks out this new
test generator; the second job loads immutable checkpoint hashes before
scoring these tasks.

A refutation release requires an **exact verified model-proposed
counterexample**. Otherwise the policy can issue an explicitly provisional
positive verdict or abstain. This preserves a clean distinction between
safe mathematical certificate claims and the neural model's classification
accuracy. A formalized real-paper claim still requires an independent
semantics-preserving translation, which this finite benchmark does not
attempt.
