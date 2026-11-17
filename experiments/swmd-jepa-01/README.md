# SWMD-JEPA-01: paired world views and latent dynamics

A standalone, locally runnable experiment for the **view-selection** claim in
[Balestriero & LeCun, *Proving and Improving the Platonic Representation Hypothesis with JEPAs* (2026)](https://www.alphaxiv.org/abs/2610.39302).
The study tests whether two genuinely different descriptions of the *same world state*
produce representations that decode and transition more reliably than corrupted
same-description pairs or next-token training. It is a small **VICReg-style JEPA proxy**,
not an implementation of the paper's LeJEPA objective or a reproduction of its theorem.

## Four fixed arms

| Arm | Encoder / training signal | Test-time readouts |
| --- | --- | --- |
| `next_token` | Shared GRU encoder and next-token CE over both dialects | Same frozen encoder probes |
| `bad_view` | VICReg on **two independent masks of the same randomly selected caption** | Same probes |
| `good_view` | VICReg on **two independent descriptions of the same symbolic state** | Same probes |
| `good_view_transition` | `good_view` plus an action-conditioned **reusable GRUCell** predicting the next embedding, with stop-gradient target | Same probes, plus direct recurrent-predictor NMSE |

The encoder architecture, dimension, dataset split, seeds, learning rate, epoch
count and evaluation procedures are shared. The task-specific LM and transition
heads add parameters to their corresponding arms; token exposure is recorded so
that differences in training work remain visible. Positive view selection is
therefore tested as a **pilot**, not an exact compute-matched causal estimate.

## Run locally

Python 3.12+ with PyTorch. This repo's root `requirements.txt` already pins
PyTorch 2.9.1, and this folder can be installed alone:

```sh
python -m pip install -r experiments/swmd-jepa-01/requirements.txt
python -m unittest discover -s experiments/swmd-jepa-01/tests -v
python experiments/swmd-jepa-01/run.py \
  --worlds 96 --steps 12 --data-seed 2026 \
  --seeds 11,29,47 --epochs 10 --batch 64 --dim 48 \
  --device auto --out experiments/swmd-jepa-01/runs/preregistered
```

A short CPU smoke run:

```sh
python experiments/swmd-jepa-01/run.py \
  --worlds 12 --steps 5 --seeds 11 --epochs 1 --batch 16 --dim 8 \
  --device cpu --out /tmp/swmd-jepa-smoke
```

Outputs: `manifest.json` (configuration, world-count splits, source + code SHA-256,
Torch and Python versions), `seed-<seed>-<arm>.json` (loss history, exposure,
encoder size, metrics), and `summary.json` (paired seed-level differences and
seed bootstrap intervals when running all four arms). Generated outputs remain
untracked by default. All split assignments depend only on `--data-seed`, so
training seeds share exactly the same train/validation/test data.

## What the synthetic world contains

Each world is a deterministic action trajectory. A state consists of five discrete
attributes: room, key, gate, energy and sigil, changed by `move`, `collect`, `flip`,
`rest` and `rotate`. Dialect A uses `room amber; key iron; ...`, while dialect B
uses disjoint words such as `located atrium; holding steel; ...`. Each state
produces two separately shuffled and independently rendered views. The model
never receives the symbolic attributes or the `world_id` at training time.
They are used only to generate the data and score frozen readouts.

Entire **world_id groups** are assigned to train (70%), validation (15%), or
test (15%). Some symbolic states recur in different worlds; receipts therefore
report the fraction of test states absent from training and separate retrieval
and closure metrics on those unseen states. The current experiment evaluates
cross-world generalization; repeated views and the finite attribute vocabulary
limit claims about open-ended world models.

## StoryworldTRM / SWMD adapter

Supply `--input-jsonl FILE` with one transition per line:

```json
{"world_id":"episode-0001","state_id":"s10","next_state_id":"s11","view_a":"room amber key iron gate open","view_b":"located atrium holding steel barrier released","next_view_a":"room blue key iron gate open","next_view_b":"located bay holding steel barrier released","action":"move","attributes":{"room":0,"key":1,"gate":1}}
```

`world_id` must identify a complete, disjoint episode or environment so no
world crosses a split. Use a stable canonical `state_id` for equivalent states.
`next_world_id` may be supplied and must equal `world_id`. All eight text/ID/action
fields shown above are required; `attributes` is optional. Attribute decoding is
reported only for attributes present in *every* train and test record. Vocabulary
is built from training text alone, with an `<unk>` token for unseen words.
For larger corpora, materialize a canonical JSONL export before starting the
experiment and keep its checksum with the receipt. This adapter is a contract;
there is no implicit download from StoryworldTRM or any remote API.

## What is measured

- **`retrieval_r1`**: test caption A's nearest test caption B, cosine similarity,
  counted correct when canonical `state_id` agrees. `retrieval_chance` is the
  exact prevalence baseline including duplicate states; an unseen-state subset
  is reported separately.
- **`state_accuracy_macro`**: mean attribute accuracy of a deterministic ridge
  probe trained on **training embeddings only**, then evaluated on test views.
- **`closure_probe_nmse`**: action-conditioned *ridge* transition readout trained
  on training embeddings only, tested against mean next-state embedding. Error
  divided by test target variance; lower is better. `closure_probe_nmse_unseen`
  evaluates the subset with unseen state IDs, using the same normalization.
- **`view_disagreement_nmse` / `transition_view_gap_nmse`**: normalized difference
  between two descriptions of a state and between their predicted successors.
  Both can be deceptively low under representation collapse; interpret with
  retrieval and effective rank.
- **`trained_transition_nmse`**: direct GRUCell predictor score for the fourth arm.
  This is not numerically comparable to the fitted ridge readout without noting
  the different training protocols.

See [PROTOCOL.md](PROTOCOL.md) for preregistered directions, interpretation,
limitations and an escalation gate. A CPU pilot belongs in the **exploratory**
category. It cannot establish universal latent identifiability or compact AGI.
