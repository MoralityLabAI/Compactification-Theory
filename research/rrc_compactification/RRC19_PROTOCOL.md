# RRC-19: family-balanced source robustness, executed protocol

**Execution:** November 10, 2026. One exploratory CPU training seed and a fixed 12-candidate exact-verifier budget. This protocol was finalized during implementation after the source-family idea was selected and after the first model's development scores were available. Prior RRC-18 test outcomes were public. Two additional robustness seeds were run after the primary held-out evaluation and are labelled **post-primary checks** throughout.

## Data and access boundary

Training uses three exactly known families: 1) signed coefficient units with each degree independently chosen from \([-2,2]\), 2) unrestricted periodic arithmetic-part product series, and 3) sparse signed unit series with selected nonzero degrees. Every family appears with each of the four existing topologies in 300 training cases per cell, 3,600 cases total. Development uses 35 fresh cases per topology/family cell, 420 total.

Five test groups have 140 unique demonstrations each, 35 per topology. Three groups repeat training-source generators on new instances. The two **generator-disjoint** groups are reciprocals of finite sparse polynomials and convolution squares of sparse unit series. Twenty test-only inconsistent demonstration pairs are verified to be outside the full 64-graph grammar through degree 12. Eight future-collision pairs duplicate all observable evidence, changing one degree-15 query output. No train/test visible-prefix aliases are allowed, except the deliberate within-test future pairs.

`family19.roster(include_test=False)` creates **only** train and development records. The held-out source generators and collision/negative selectors are never called during training. The source family and hidden graph labels are absent from the model features. The student sees 13 positions of normalized raw input/output coefficients and a one-hot public topology; feedforward uses the same coefficients flattened. Labels are canonical representatives from RRC-18's **exact** degree-12 polynomial quotient, used only during optimization.

## Matched models and inference conditions

- Balanced GRU: 6,240 FP32 parameters, hidden width 32, 700 single-thread CPU Adam steps with learning rate 0.003 and batch 128. Each sampling draw chooses uniformly from 12 family/topology cells.
- Skewed GRU: same architecture and **identical initialization**, same roster and total training-example visits; family weights 80% signed, 15% product, 5% sparse, topology uniform.
- Balanced FF: full-prefix 6,235-parameter two-layer feedforward control, same optimizer/token budget and balanced draw sequence.
- Shuffled-label GRU: the balanced architecture and roster with targets shuffled **within** each family/topology cell, preserving cell frequencies.
- Frozen RRC-18 recurrent and feedforward checkpoints: historical single-source comparators, scored on the fresh RRC-19 cases without retraining.
- Nonlearned controls: certified class-size prior, first twelve canonical syntax graphs, and full class search. An additional hybrid fixes six neural candidates followed by up to six previously unseen class-size representatives. The hybrid ratio is a declared default rather than tuned on the test.

Every arm proposes a list before exact candidate execution. All neural arms use zero graph execution previews. A verifier tries at most twelve distinct programs, executing **both** observed demonstrations per query and recording the full integer-loop proxy. The full-class reference can make up to 33 queries and must be labelled separately. Future coefficients are inaccessible before independent scoring.

## Gates and controlled failures

1. Every source is a normalized integral unit series; independent exact product, reciprocal and convolution oracles support generation. Training-family cell counts are exact.
2. Every checked program is independently executed through degree 12. No candidate outside the finite classes can count as an acceptance, and positive full-class cases must be accepted.
3. Normal and deliberately corrupted examples receive the same verification rule. All 20 negative cases must fail the full grammar, and eight future pairs must produce identical traces with exactly eight full-search future false acceptances.
4. Portable no-Torch inference must agree numerically with PyTorch for both matched architectures. Inference must be blind to future output, source graph label and unsampled graph executions.
5. Recurrent-source robustness requires exceeding both the skewed GRU and parameter-matched FF on the **combined two novel families**, and exceeding the certified class prior under equal query budget. A gain limited to products, already a training family, is reported narrowly. Any failure is retained.
6. Exact source and checkpoint SHA-256 digests, trajectory hashes, complete per-split metrics and two post-primary seed checks are stored in the receipt. Independent training must reproduce all four checkpoint JSON files byte-for-byte before a reproducibility claim.
7. RRC-18 finite polynomial normalizers and RRC-17 quotient tests remain green. CI executes frozen standard-library checks only; no model downloads, live training or GPU spending.

## Actual resource boundary

Each GRU inference costs an estimated 53,216 dense MACs and each FF inference 6,120. Each candidate check executes two exact demo series and charges two instances of the topology-specific integer-loop proxy. Four 700-step training jobs each see 89,600 samples and consume additional backpropagation work. Prior certified quotient compilation, target-label generation, train data creation and frozen historical RRC-18 training are external resources. No meaningful comparison adds floating MAC counts to integer coefficient-loop proxies.

## Reproduce

From the repository root on Python 3.10+:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_family19.py' -v
python research/rrc_compactification/bench19.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_certify18.py' -v
python -m unittest discover -s research/rrc_compactification -p 'test_quotient17.py' -v
```

Optional deterministic independent retraining with CPU PyTorch:

```bash
python research/rrc_compactification/train19.py --out-dir /tmp/rrc19-retrain
for role in balanced_recurrent skewed_recurrent balanced_feedforward shuffled_recurrent; do
    cmp /tmp/rrc19-retrain/rrc19_${role}.json research/rrc_compactification/results/rrc19_${role}.json
done
```

Any source/weight drift requires an explicitly regenerated receipt and new audit.
