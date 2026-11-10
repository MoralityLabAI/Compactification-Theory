# RRC-24: executed late-divergence candidate-ordering protocol

**Status:** Exploratory CPU pilot, 2026-11-10. Earlier RRC-01 through RRC-23
results informed this design, and the primary held-out roster is exposed by
this release. The protocol was completed during implementation, not externally
preregistered. Additional seeds are post-primary robustness checks.

## Fixed task and data

- DSL: four base operations `I,N,R,S`; eight delays `{64,80,96,112,128,144,160,176}`;
  amplitudes `{-2,-1,1,2}`; four null spellings. Total 132 literal programs.
- Every sample has two independent unit-series demonstration pairs through
  degree 192 and one query output through degree 256. The model observes only
  degrees `0..16` from both demonstration inputs and outputs.
- Training: 320 cases each from signed integer, periodic-partition product and
  sparse signed source families; 960 cases total. Development: 64 cases each,
  192 total. Training uses `dataset(include_test=False)` and never creates the
  rational, squared or late-source generators.
- Primary evaluation: 40 new cases from each of the six families, 240 total.
  Additional controls comprise 12 inconsistent visible demonstration pairs
  selected after all-four-base exact rejection and eight null/delayed future
  collision pairs (16 records).
- The two extra 24-per-family seeds were generated **after** primary results
  and are recorded as robustness checks. All rosters and checkpoints have
  exact SHA-256 locks in the deterministic receipt.

## Models and policies

A 609-parameter GRU with 11 hidden units scans seventeen four-channel input
steps. A parameter-near 588-parameter feedforward network with eight hidden
units sees all 68 features in one pass. A 609-parameter shuffled-label GRU
is the negative control. Both trained models use 340 Adam steps, batch size
96 and learning rate .006, one CPU thread and seeded minibatches. The
shuffled control preserves the label histogram.

The neural models rank the **four base operations only**. After each
proposed base, an exact test-time algebraic algorithm recovers a candidate
`(delay,amplitude)` by its first residual degree and verifies the complete
public demos; at most two of these *macro queries* are allowed in the strict
policy. Four queries provide a class-complete factorized reference on the
in-class positives. Every candidate rejected consumes its full charged base
verification, including the expensive `R` operation if used.

Controls: fixed base order with two macro queries; an exact degree-16
base-ordering comparator with two or four full macro queries (and eight
charged short-degree unary executions to prepare the ranking); a full
four-base factorized solver; and a 132-program literal syntax search
with progressive prefix checks on a **separately identified twelve-case**
diagnostic subset. The literal-budget and factorized macro-budget query
units have different computational powers; report them independently.

## Registered interpretation and validation

1. All 32 delayed choices for a fixed base must agree below degree 64.
   The first residual-degree identity must reconstruct each allowed delay
   and amplitude from exact integer coefficients.
2. Frozen models must receive no hidden source graph, future query output,
   unseen-family name or test-time candidate executions. A patch that makes
   the interpreter unavailable before proposal must leave model logits intact.
3. Every accepted factorized program must exactly match **both** observed
   degree-192 demonstrations. Reject out-of-grammar delays and mismatched
   amplitudes. The full factorized baseline must recover all primary
   positive cases and reject the twelve constructed inconsistent inputs.
4. Both members of each collision pair must have identical visible input
   and search traces, with exactly eight future misses for full search.
5. Macro queries, direct syntax checks and preliminary degree-16 checks
   must be separately accounted for. Neural dense MACs, exact integer-loop
   proxies and measured wall-clock times remain separate units.
6. The 17 new tests, source-bound frozen semantic receipt, independent
   PyTorch checkpoint reproduction and prior RRC-22/RRC-23 regression checks
   constitute the audit. CI runs standard-library tests and exact replay;
   no external model download or CPU training is needed.

**Success gate:** A learned ranker must improve over the static order with
two macro checks, and the comparison against a direct exact short-prefix
sorter must remain visible. A strong learning advantage would additionally
beat the exact baseline in measured time at equal future-prediction quality.
Any such gate failure remains in the report.

## Reproduction

From the repository root:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_delay24.py' -v
python research/rrc_compactification/bench24.py --check
python research/rrc_compactification/bench24.py --profile --cases 36 --rounds 2 --out /tmp/rrc24_host_profile.json
```

Optional CPU retraining requires PyTorch:

```bash
python research/rrc_compactification/train24.py --out-dir /tmp/rrc24-retrain
for name in recurrent feedforward shuffled; do
  cmp /tmp/rrc24-retrain/rrc24_${name}.json research/rrc_compactification/results/rrc24_${name}.json
done
```

Physical profile nanoseconds and memory depend on hardware and are excluded
from byte-exact CI thresholds. The complete semantic receipt contains hashes,
counts, candidate traces, syntax diagnostics and the post-primary replications.
