# Pluto-Gym v0.1

Pluto-Gym runs interface-relative learning experiments for Compactification
Theory and connects them to the existing TensorCraft StarCraft lab.

The runnable learning task is a small synthetic multi-front environment. A
trainable target policy uses a bounded host-DRAM replay buffer, while an exact
public scheduler handles screen choice. TensorCraft supplies the pinned combat
surrogate, native frozen-Pluto farm, and trace/provenance checks. The designer
webhook chooses validated rule variants once per training block.

| Component | Available behavior |
|---|---|
| Exact synthetic gym | Seeded episodes; camera/global observations; command and switch costs; deadlines; weighted objectives; exact attainable reward |
| Compact student | NumPy MLP; actual CPU Adam updates; deterministic checkpoint serialization; bounded DRAM replay |
| Curriculum experiment | Frozen checkpoint, random curriculum, designer curriculum; separate train/test seeds; held-out rule combinations; forgetting and calibration records |
| Designer hook | Provider-neutral HTTP request, high reasoning hint, call budget, response/token caps, subprocess timeout, idempotency key |
| TensorCraft | Pinned surrogate simulation, native A/B preparation/execution, strict trace analysis and optional replay/map binding |
| Gymnasium | Optional Env wrapper with declared spaces and action masks |

## Run it

Use Python 3.10 or later from the repository root. NumPy is the only package
needed for the compact student. The exact environment, mathematical checks,
and packed replay buffer use the Python standard library.

```bash
python -m pip install -r research/pluto_gym/requirements.txt
python -m research.pluto_gym calibrate
python -m research.pluto_gym run --output runs/pluto-gym/first --seed 0
```

Every run gets a fresh output directory. The manifest freezes the source hashes,
rule combinations, interface, seeds, designer configuration and budgets before
training. Outputs include a base checkpoint, final checkpoints, each adaptation
step, a complete receipt, and `designer_traces.jsonl`.

The default student has 33 float64 parameters, occupying 264 weight bytes.
It learns the target relation from `(visible cue, public matchup rule)`.
The scheduling program and supplied observation are additional resources.
Schedule-search work is currently unmetered; this baseline establishes the
experiment machinery and target adaptation. A learned strategic scheduler and
recurrent memory sweep are later experiments.

For the four-front, twelve-tick calibration, the camera profile permits two
successful commands and has reward ceiling 0.5. The assisted and API profiles
permit all four commands. These are synthetic interface specifications, with
no fitted human reaction-time interpretation.

```bash
python -m research.pluto_gym run --output runs/pluto-gym/api --interface api
python -m research.pluto_gym run --output runs/pluto-gym/memory64 --replay-capacity 64
```

## Designer webhook: Luna High first, a local designer later

Use an HTTP broker that implements [the design contract](DESIGNER.md). Configure
its actual model/route ID. Luna High describes the intended route; this package
does not assume that a public provider exposes that exact API model name.

```bash
export PLUTO_GYM_DESIGN_URL='https://your-broker.example/v1/design'
export PLUTO_GYM_DESIGN_MODEL='your-luna-model-route'
export PLUTO_GYM_DESIGN_REASONING='high'
# Set PLUTO_GYM_DESIGN_TOKEN through your usual secret manager when required.
python -m research.pluto_gym run --designer webhook --iterations 4 \
  --designer-max-output-tokens 800 --output runs/pluto-gym/luna01
```

Each iteration permits one call. The default run allows four calls, with 800
requested output tokens per call, a 32 KiB request ceiling and a 16 KiB response
ceiling. A child process performs HTTP; its parent terminates it when the
25-second subprocess timeout expires. The broker must enforce provider billing
limits and honor the requested output cap. There are no automatic retries.

Only training-probe scores enter the request. In the synthetic environment,
each score includes its exact interface ceiling and remaining regret. This
allows a designer to distinguish a controller error from an unreachable command.
The offline control chooses the candidate with the largest interface regret.
Both adaptive arms receive the same update count, batch size, retained-label
budget, replay capacity and number of diagnostic episodes. Raw oracle collection
costs are reported separately because game rules change trajectory lengths.

The future Bonsai/Qwen 27B designer can use the same broker contract over an
explicit loopback endpoint, for example `http://127.0.0.1:8080/v1/design`.
`designer_traces.jsonl` retains sanitized requests, valid proposals and subsequent
training feedback for later proposer distillation. Its training-probe scores
carry no confirmed game-design-benefit label. A model revision, quantized kernel,
training precision and adapter recipe must be pinned when that stage begins.

## Existing StarCraft integration found

The existing lab is
[patrickdugan/TensorCraft, draft PR #5](https://github.com/patrickdugan/TensorCraft/pull/5),
on branch `research/starcraft-headless-farm-v0`, pinned to
`f12f3241b686231d09fbfe40a9f03706ebc7e068`. Its upstream draft stack retains its
current owners. Pluto-Gym adds an external adapter and changes none of those files.

`dependencies.lock.json` records the complete upstream source manifest. The adapter
checks all 70 files by Git blob hash, checks HEAD when Git metadata is present,
and invokes an isolated copy of verified source. Use a clean dedicated checkout;
generated files belong outside it. Owned game installations remain external.

```bash
git clone https://github.com/patrickdugan/TensorCraft.git ../TensorCraft-pluto-gym
git -C ../TensorCraft-pluto-gym checkout --detach f12f3241b686231d09fbfe40a9f03706ebc7e068
python -m research.pluto_gym doctor --tensorcraft ../TensorCraft-pluto-gym
python -m research.pluto_gym surrogate --tensorcraft ../TensorCraft-pluto-gym \
  --scenario zvz_equal_spend_clustered --variant devourer_speed_550 \
  --trials 16 --seed 20261029 --output runs/pluto-gym/surrogate.json
```

The surrogate supports `control`, `devourer_speed_550`, `devourer_cooldown_90`
and `devourer_speed_550_cooldown_90`. Its output keeps TensorCraft's
`stochastic_surrogate_not_observed_games` evidence label.

For native experiments, start with TensorCraft's pinned
[`batch.example.json`](https://github.com/patrickdugan/TensorCraft/blob/f12f3241b686231d09fbfe40a9f03706ebc7e068/starcraft_lab/batch.example.json)
and
[`runtime.virtual-screen.example.json`](https://github.com/patrickdugan/TensorCraft/blob/f12f3241b686231d09fbfe40a9f03706ebc7e068/starcraft_lab/farm/runtime.virtual-screen.example.json).
Populate real local asset paths and hashes. The upstream planner currently accepts
the control/speed pair; four-arm native planning requires a separate upstream change.

```bash
python -m research.pluto_gym native-prepare --tensorcraft ../TensorCraft-pluto-gym \
  --spec /absolute/path/batch.json --runtime /absolute/path/runtime.json \
  --output runs/pluto-gym/native01
python -m research.pluto_gym native-run --tensorcraft ../TensorCraft-pluto-gym \
  --workspace runs/pluto-gym/native01/farm --allow-native
python -m research.pluto_gym trace --tensorcraft ../TensorCraft-pluto-gym \
  --trace /absolute/path/trace.jsonl --output runs/pluto-gym/trace-analysis.json
```

Trace analysis supports `--manifest`, `--replay-report`, `--map` and `--spawn-tiles`
for the existing provenance binder. A syntactically valid trace keeps its origin
unverified until the separate native evidence requirements are satisfied.

## DRAM and training scope

DRAM here means host memory. The ring buffer preallocates feature/label arrays,
overwrites the oldest entries at capacity, and supplies every optimizer batch.
The resource ledger records exact array payload, parameter bytes, optimizer
moments and transient tensor estimates separately from whole-process peak RSS.
GPU/PCIe measurements continue to use the existing
[`gpu_dram_bench.py`](../recurrent_circuits_v1/gpu_dram_bench.py) experiment.

The pinned [Pluto release](https://github.com/tscmoo/pluto/tree/8e78659caf1a987697abe76cd296a8eef9671c97)
provides inference binaries and int8 weights for Brood War 1.16.1/BWAPI 4.4.0.
Its published package lacks the training source needed to update those weights.
TensorCraft's current tournament observer records telemetry and issues no orders.
Native action interception and a trainable native agent therefore remain explicit
integration requirements. Running the synthetic trainer never changes Pluto files.

Pluto v2's default timing can slow simulation when inference runs late. Native
interface comparisons must record simulation frames and wall time separately,
pin that timing mode, and validate the actual command channel. An APM figure or
requested seed label alone cannot establish the IRC deadline contract.

## Verification

```bash
python -m unittest discover -s research/pluto_gym/tests -v
python -O -m unittest discover -s research/pluto_gym/tests -v
TENSORCRAFT_TEST_CHECKOUT=/absolute/path/TensorCraft-pluto-gym \
  python -B -m unittest research.pluto_gym.tests.test_tensorcraft -v
```

The optional wrapper follows the
[Gymnasium Env API](https://gymnasium.farama.org/api/env/). Install it separately:

```bash
python -m pip install -r research/pluto_gym/requirements-gym.txt
python -m unittest research.pluto_gym.tests.test_gymnasium -v
```

Tests cover deadlines, hidden-cue noninterference, exact scheduling, malformed
inputs, memory lower bounds, replay overwrite, actual learning and checkpoint
replay. HTTP loopback tests run where sockets are permitted; this build sandbox
blocked those cases. Its actual subprocess timeout test executed. TensorCraft
integration tests use pinned upstream code with synthetic local assets for
farm preparation and launch no game. Source-hashed smoke evidence is in `results/`.

See [PROTOCOL.md](PROTOCOL.md) for mathematical scope and subsequent native/learner
gates. The empirical result here is a working bounded experiment with synthetic
target learning; StarCraft strategic improvement requires native measurements.

