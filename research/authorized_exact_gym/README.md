# Authorized Exact Gym (AEG-01)

AEG-01 turns the NOR45–60 research into a reusable CPU task: decide whether a
signed computation is admissible, then return every exact output coefficient
and its original indexed Merkle root. Each episode has one action. The grader
uses the original NOR45 interpreter and retains its own reference values.

The suite contains separate public training and development panels of 14
tasks each. Both signed degrees, 512 and 1024, appear. Tests cover valid
authority-selected programs, changed signatures, a query substitution,
cross-session replay, a signed provider result that disagrees with the
authorized computation, and expiration. A separately authorized program is
accepted according to its own semantics. That case makes the authority's
external semantic honesty an explicit premise.

The four built-in policies reproduce the NOR60 factorial: original execution,
static Merkle roots, support-compressed execution with invocation-local unary
reuse, and their composition. They all pass through the real signed verifier.
An external runner can consume the same public JSON and submit actions for
independent grading.

## Run on CPU

From the repository root, with Python 3.12, `cryptography==46.0.4`, and
`gmpy2==2.3.2`:

```sh
python -m unittest research.authorized_exact_gym.test_gym -v
python -O -m unittest research.authorized_exact_gym.test_gym -v
python -m research.authorized_exact_gym.gym export --split development --out-dir /tmp/aeg01
python -m research.authorized_exact_gym.gym run --public /tmp/aeg01/public/development.json --policy combined --out /tmp/aeg01/answers.json
python -m research.authorized_exact_gym.gym score --public /tmp/aeg01/public/development.json --referee /tmp/aeg01/referee/development.json --manifest /tmp/aeg01/development.manifest.json --answers /tmp/aeg01/answers.json --out /tmp/aeg01/score.json
```

Every output path must be new. An existing file stops the write. Exporting
creates a manifest with separate public and referee hashes. The scorer checks
these hashes against the manifest that the evaluator retained. Changing both
the manifest and oracle requires the evaluator's trust boundary; a hash alone
cannot establish an independent authority.

## Participant interface

Only `public/<split>.json` goes to a candidate. A public task contains its
opaque `case_id`, `public_sha256`, independently supplied query and trust
context, and untrusted signed envelopes. Labels and expected vectors remain in
`referee/<split>.json`. These development files are public research fixtures;
placing them in separate directories supplies an interface boundary.

Each action has exactly these fields:

```json
{"case_id":"<task id>","public_sha256":"<input hash>","decision":"accept","values_decimal":["1","0"],"root_sha256":"<full root>"}
```

The example vector above illustrates the schema. An accepted answer must
provide all 769 or 1537 coefficients as canonical decimal strings. Use
`decision: "reject"` or `"abstain"` with `values_decimal: null` and
`root_sha256: null` for the other actions. Unknown fields, repeated case IDs,
missing actions and responses to a different public input fail validation.
An answers file wraps the action list with `provenance`, which the score labels
as participant-reported. A future model evaluation needs independently
verified checkpoint and execution provenance.

The one-step Python API is `AuthorizedExactGym(public, referee)`, followed by
`reset(index=...)` and `step(action)`. The observation is copied on reset.
Reward is one for a correct rejection or a correctly authorized full output,
and zero for other decisions. False acceptance and wrong exact output remain
separate counts. The API has no Gymnasium dependency.

## What this run establishes

The committed smoke receipt checks finite, public development fixtures with a
symbolic implementation. It records false acceptance, false rejection and
abstention, together with complete output correctness, source hashes and
backend provenance. A diagnostic elapsed time describes one sequential pass.
The NOR60 registered five-repeat study owns the comparative performance claim.

The AEG-01 receipt covers deterministic symbolic policies on public development
tasks. The package supplies public training and development splits. A future
held-out study must freeze its candidate and compute budget, register new source
content hashes and exclusions, retain its first attempt, and keep the referee
outside the candidate process. CPU smoke tests run on an ordinary host; a model
study must specify its own stopping rule and receipt before entering the compute
queue.

The Ed25519 private keys and nonces here are public test fixtures. The NOR45
verifier is stateless per call. Real deployment would need independently
managed keys and nonce consumption policy. The results concern consistency
with an authorized finite program under the cryptographic assumptions stated
in the [paper](../../papers/authorized-exact-computation/paper.md).
