# SHAFT-COMP-24: Identifying a temporal trigger law from behavior

**Registered:** November 11, 2026, [COMP-24 protocol](SHAFT_COMP_24_PROTOCOL.md), commit \`92ad76ece8dafe37a6181b48ca53ee52f1554059\`.  
**Calibration frozen before heldout:** commit \`25c07861793ed747eb8b18e36a773fca56c3d753\`, full source-calibration SHA-256 \`e007f5cdbc33fab71740392ec65b5f2dfc36f8bbf20fbca8fffe4700bcf057be\`.  
**Result:** all **72** unknown Boolean trigger entries recovered, 36 in each task family; a frozen learned rule predicted **98.2239%** of held-out native two-query bits at recurrent depth 40, against **82.5161%** from the no-trigger baseline. Registered improvement **+15.7077 percentage points**, four-model-stratified bootstrap 95% interval **[+15.6426,+15.7701] percentage points**.

The study concerns frozen, locally accessible 3,265-parameter GRUs on synthetic six-query latch tasks. It makes no claim about frontier-model hidden goals, covert operations, distributed persistence or recovery of deleted model weights.

## 1. Identification theorem

COMP-22 established that the 768 canonical states \`(goal g,latch l,previous p)\` admit a 384-class behavioral quotient \`(effective goal b, previous p)\` for both repeat-toggle and successor-toggle families. The six-bit effective goal table is \`b = g XOR (63 if l else 0)\`.

The new experiment *withholds the trigger law*. It assumes only that a next-token behavior uses an unknown Boolean trigger \`T(p,a)\`, causing the two-query final response to factor as

\[
B_2(a,c)=B_1(c)\oplus T(p,a),
\]

where \`B1(c)\` is the native present one-query behavioral response, \`p\` is the **last observed input token** and \`a,c\` are the two fresh queries. XOR cancellation identifies an individual rule bit:

\[
\boxed{T(p,a)=B_1(c)\oplus B_2(a,c).}
\]

This requires no oracle goal, latch or rule labels. It does require the XOR factorization **as a structural prior**, direct access to native counterfactual one- and two-query responses, and the known input history.

The finite standard-library reference exhaustively checks both 384-effective-state families (768 cases combined) and 50 randomly generated arbitrary 6×6 Boolean rules. It verifies unique recovery of each row from a single paired response per \`(p,a)\`. When previous-input labels are omitted, the rows can be permuted with a simultaneous relabeling of \`p\` without affecting unlabelled observations. This row-permutation ambiguity is an identifiability limitation.

[TriggerIdentification.lean](lean/Shaft/TriggerIdentification.lean) kernel-checks Boolean XOR cancellation, recovery of a rule bit, and injectivity of the rule-to-behavior map given observed input \`p\`. The third proof uses only standard Lean axioms (\`propext\`, \`Quot.sound\`), with no \`sorryAx\`. Probabilistic vote robustness and the recurrent-model measurements are independently executable rather than formalized in Lean.

## 2. Calibrating the unknown 36-bit rule

Use immutable checkpoints from the prior COMP-23 archive. Check each file's SHA-256 against that archive's manifest and its model state-dict hash against the original COMP-23 receipt before inference.

Calibration source models:
- repeat-toggle, seed **2301**;
- successor-toggle, seed **2311**.

At each state depth 12 and 40 generate 64 goals ×16 independent input histories, totaling **2,048 states per task family**. For every source state the learner reads the final token of its *observed input history* and the 6 native one-query plus 36 native two-query binary responses.

For every \`(p,a)\`, compute the 6 observed XOR discrepancies \`B1(c) XOR B2(a,c)\` across all query endings \`c\` and every matching source state. Infer the corresponding rule bit by **strict majority vote**, with ties assigned zero. The aggregation procedure and frozen random seeds are in the preregistration and [the code](experiments/identify_trigger.py). No temporal-state oracle labels are passed into the learner.

| Calibration family | Source states | Minimum votes per trigger entry | Smallest majority-vote distance from 50% | Recovered entries |
|---|---:|---:|---:|---:|
| Repeat-toggle | 2,048 | 1,872 | 44.44 pp | 36/36 |
| Successor-toggle | 2,048 | 1,884 | 48.47 pp | 36/36 |

The resulting matrices are an identity matrix for repeat-toggle and the one-step cyclic successor permutation matrix for successor-toggle. Both were inferred *before* consulting the task's true rule in the held-out evaluation. The full source observations have SHA-256 \`e007f5cdbc33fab71740392ec65b5f2dfc36f8bbf20fbca8fffe4700bcf057be\`. The frozen matrix file has SHA-256 \`81fbcb7bc21c88810c8f70f7a1f1c0edb5d8485d8bf0e4c54f21f9d23ade5c92\`.

The rule matrix was frozen in [comp24_frozen_calibration.json](results/comp24_frozen_calibration.json) before *any new held-out state computation*, with the checksum of the original experiment source and the immutable model checkpoints.

## 3. Held-out test and controls

Evaluate on four previously trained but **calibration-excluded** GRUs: repeat-toggle seeds **2302, 2303** and successor-toggle seeds **2312, 2313**. These model seeds were already used in COMP-23, so the test is a **fresh-trajectory transfer**, not a previously unseen model-level holdout.

Each held-out seed contributes 1,024 newly sampled states at depth 12 and another 1,024 at depth 40: **8,192 held-out states** in total. For every state, record the observed prefix's final token, its actual native one-query response \`B1\`, and the true native two-query response \`B2\`. Apply the *frozen* inferred trigger to build \`B2_pred\`. No goal bits, oracle latch or other latent labels are supplied to the predictor.

Controls are:
- **No trigger:** repeat B1 over every first query, predicting no latch updates.
- **Wrong family:** substitute the trigger inferred from the other family.
- **Previous-token permutation:** shift the observed previous-token labels before looking up T.
- **Exact-rule oracle:** substitute the true finite trigger, used for posthoc evaluation only.

### Aggregate performance

| Method | Depth-12 bit accuracy | Depth-40 bit accuracy | Depth-40 complete 36-bit accuracy |
|---|---:|---:|---:|
| **Learned trigger** | **98.4572%** | **98.2239%** | **76.5869%** |
| Exact-rule oracle | 98.4572% | 98.2239% | 76.5869% |
| No trigger | 82.6599% | 82.5161% | 0% |
| Wrong-family trigger | 66.2618% | 66.1519% | 0% |
| Permuted previous token | 66.4612% | 66.4001% | 0% |

All methods are evaluated against the **native model's actual two-query binary outputs**, not idealized automaton outputs. The learned and oracle trigger matrices are identical, so their predictions agree on all 8,192 held-out states.

The registered primary depth-40 contrast is **+0.15707737** bit accuracy for learned versus no-trigger prediction, with 1,000 bootstrap resamples stratified by four fixed model seeds: 95% interval **[+0.15642632,+0.15770128]**. Each model exhibits a positive gain.

### Important model-level variation

| Held-out model | Learned bit accuracy, depth 40 | Learned complete-signature accuracy | Improvement over no-trigger |
|---|---:|---:|---:|
| Repeat 2302 | 99.4331% | 90.8203% | +16.3303 pp |
| Repeat 2303 | 99.2730% | 89.1602% | +16.4063 pp |
| Successor 2312 | **94.3468%** | **29.3945%** | +13.5254 pp |
| Successor 2313 | 99.8427% | 96.9727% | +16.5690 pp |

Successor model 2312 is a significant outlier. Its errors in native two-query behavior leave the correct symbolic transition law insufficient for a completely accurate signature. The rule-learning task was successful, while the native model itself supplied a noisier behavioral approximation.

### What was learned

The result identifies **36 bits of transition-law structure conditional on** the supplied XOR compositional form and a known previous input. It is a form of finite system identification from behavioral queries. At most 6×6 rule entries need to be recovered; the learner also retains model-derived \`B1\` and the input-history position \`p\` at prediction time.

Since the learned and exact trigger matrices coincide bit for bit, an already trained factorized decoder using either one would compute the same forward function when its weights and inputs are fixed. The optional experiment that *retrained* a 900-parameter learned-rule decoder was **not performed**; no retraining result is asserted.

## 4. Reproduction and verified limits

The full original held-out JSON was reproduced with byte-identical results under PyTorch 2.10.0+cpu and NumPy 2.3.5, using three CPU threads and deterministic PyTorch operations:

\`40c0e1edbfb81ff51509435be5a853e6de46c64fc52ce741b13cb164f1c2ec23\` (SHA-256).

The source is [identify_trigger.py](experiments/identify_trigger.py), which checks original archive hashes and requires the frozen SHA-256 before running held-out inference. The input archive is the existing SHAFT-COMP-23 receipts ZIP, extracted into a project root with \`MANIFEST.json\`, \`checkpoints/\`, and \`original/\`.

Run from a Python environment with that root and repo source on \`PYTHONPATH\`:

\`\`\`bash
python -m research.shaft_compactification.experiments.identify_trigger calibrate \
  --archive-root /path/to/extracted_comp23 --out /tmp/comp24_calibration.json
python -m research.shaft_compactification.experiments.identify_trigger evaluate \
  --archive-root /path/to/extracted_comp23 \
  --freeze /path/to/comp24_frozen_calibration.json \
  --freeze-sha256 81fbcb7bc21c88810c8f70f7a1f1c0edb5d8485d8bf0e4c54f21f9d23ade5c92 \
  --out /tmp/comp24_heldout.json
\`\`\`

The [compact numerical summary](results/comp24_summary.json) retains aggregated exact bit and signature counts, the registered bootstrap interval, model-specific contrasts and the complete receipt's hash. The original and replayed full JSON files, training-code dependencies, frozen source observations and immutable model checkpoints are contained in the companion conversation ZIP artifact.

This is a **finite behavioral system identification** result. It does not identify arbitrary unknown rules without a structural prior or recover a model whose weights have been deleted. It does not test whether a model can hide on distributed devices, bypass shutdown, or act autonomously beyond the controlled synthetic environment.
