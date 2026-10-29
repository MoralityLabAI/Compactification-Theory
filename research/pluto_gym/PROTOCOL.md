# IRC-PG-01: interface-relative curriculum and adaptation

Version 0.1, 2026-10-29. Base repository inspected at
`8a79b9668d2fc4f2e2028b8914379aded33285f7`.

## Mathematical contract

The finite state includes static front cues, resolved fronts, the current
screen, elapsed ticks and remaining decision count. An interface fixes
camera/global visibility, command duration and switch duration. The world ends
at its deadline. An operation completed exactly at that deadline counts; an
unfinished operation has no effect.

For fixed rules and interface I, `oracle_value()` enumerates reachable front
orders using public rewards and timing. Every visited front reveals its cue
before the target decision, so an unrestricted controller can attain the computed
weight. The teacher reads only currently visible cues. A strict decision cap
also applies to the oracle after arbitrary zero-cost screen-switch sequences.

For equal weights and a first front already selected, the maximum completed
command count is

    K = min(n, floor((D + s) / (c + s))).

The weighted solver is checked against independent ordered-subset enumeration.
An observation-noninterference test changes every hidden cue while holding public
state fixed and requires identical policy observations, metadata and scheduling.

When J can emulate I with the same clock and information conditions, the measured
deficit decomposes as

    V*_J - V(policy,I) = (V*_J - V*_I) + (V*_I - V(policy,I)).

`evaluate()` records the second term per episode. The synthetic ceiling can be
computed exactly. Native game performance requires a separately justified bound;
the surrogate's score is insufficient to supply one.

`math_ref.py` implements the interface-relative parity/reset rank fixture,
whose minimum retained memory is `min(n, |S| + 1)` bits, and the independent-bit
query lower bound `n * (1 - h2(error))`. Those contracts require compression
before the later query/intervention. The current gym supplies fresh visible
observations each decision, and its student has zero recurrent state. The memory
theorem supplies a separate calibration test, with no conclusion about this MLP.

## Frozen experiment design

The default base rule has four fronts, twelve ticks, equal front weights and
target = cue. Training mutations can reverse the target mapping, shorten the
deadline, change priorities or lengthen the action window. Test combinations
use different deadline/weight combinations. The primitive target relation still
has four possible feature inputs, `(cue, target_xor)`.

Pretraining produces one checkpoint. Each adaptive arm starts from an exact
clone, receives one retained oracle attack example per seeded training episode,
and uses the same update, batch, buffer and diagnostic budgets. The random arm
chooses from the same candidate catalog. The offline designer maximizes training
interface regret; a webhook can replace that choice through the same typed
response contract. The frozen arm receives zero adaptation updates.

Raw oracle collection can create more attack rows than are retained. Record that
cost because an altered deadline changes trajectory length. CPU source, public
scheduler, model storage, host replay and optimizer state have separate entries.
The scheduler's search work remains unmetered, so total inference-FLOP matching
is outside this initial experiment.

Seed lists and rule hashes are frozen in the manifest. Pretraining,
demonstrations, training probes, retention evaluation and held-out evaluation use
disjoint seed partitions. Test combinations are excluded from the candidate
catalog. Held-out results stay out of designer requests and checkpoint selection.
Repeated measurements serve descriptive adaptation curves. A later confirmatory
study needs a fresh final test partition and a registered sampling/power plan
before observing its outcomes.

## Endpoints

Primary records are reward and regret relative to the interface oracle at each
adaptation update count. Original-rule reward measures forgetting. Target error,
Brier score and ten-bin calibration measure the learned decision probabilities.
Exact parameter and replay payloads accompany elapsed time and host RSS.

The curve measures adaptation under a fixed supervision budget. Reward-priority
and timing changes currently exercise the public schedule solver. Strategic
scheduling generalization needs a learned scheduler with its own matched controls.
Luna and Bonsai each require measured runs before making performance claims about
their curricula.

The default smoke run has one optimization seed. Its raw episode records share
a checkpoint and the same four target contexts. They cannot be treated as
independent trained-model replications. No test here opens a statistical
superiority gate for an intelligent curriculum over random choice.

## Failure conditions

- Hidden cues in a policy observation or unlabelled teacher actions invalidate
  the information-interface comparison.
- Reward above the exact oracle under the same interface indicates an environment,
  accounting or oracle defect.
- A changed checkpoint after evaluation invalidates the frozen comparison.
- A mismatched request ID, unoffered variant, extra response field or exceeded
  designer budget stops the run and preserves a failure receipt.
- Source mismatch prevents the TensorCraft adapter from starting upstream code.
- Missing or rejected native artifacts keep native evidence unresolved. Successful
  process exit alone cannot verify a match.

## Native and larger-model work following this package

1. Provision TensorCraft's runtime with owned game assets, Pluto v2.0, BWAPI,
   the observer and replay parser. Complete its native acceptance checks,
   including observed mod values, map/spawn binding and both clocks.
2. Add a native controller interface that intercepts actions and enforces
   visibility, object addressing and timing before the game consumes them.
   Preserve frozen Pluto as a teacher/control with model and opening-bandit hashes.
3. Provide a trainable agent and experience schema with legal observations,
   attempted/executed actions, rewards, masks, model version and clock values.
   Integrate its rollout buffer with the host-memory ledger. Native PPO/self-play,
   target imitation and combat-surrogate runs retain separate evidence scopes.
4. Evaluate the configured Luna broker against random curriculum under matched
   exposure/update budgets and independently generated rule combinations. Expand
   the mod language alongside executable mechanics tests.
5. Train a Bonsai/Qwen designer from validated proposals and measured feedback.
   Pin weights/tokenizer, inference quantization, adapter format, runtime kernel,
   context budget and train/test game families. Place it behind the existing
   loopback broker and repeat the frozen evaluation protocol.

RC-01 GPU/DRAM, IC-01, E10 and TensorCraft draft owners retain their protocols.
Pluto-Gym imports pinned interfaces and leaves those experiment claims intact.

