# OIC-22: Executed fault-aware adaptive sensor monitoring

**11 November 2026. Status: completed synthetic CPU training and exact finite reference. Acceptance gate: FAILED.**

## What changed

OIC-21's clean-trained sensor selector used an exact cyclic candidate belief and a 145-parameter learned address scorer. A single adversarial substitution changed the action-history branch and frequently caused a wrong shift estimate. OIC-22 adds exact **zero/one mismatch** masks for inference and a fault-aware policy teacher. The full finite Bellman oracle is solved for n=3,4,5; at n=6 and off-tree training contexts the teacher uses the frozen OIC-21 policy as an explicitly privileged fallback. This preserves the registered architecture size and gradient budget while adding extra teacher compute and a larger runtime belief.

The new student is trained for 500 Adam steps with batches of 96 distinct-context draws, using 120 generated trajectories per training phase size n=4,5,6. Its seven per-sensor features are computed from an exactly updated Bayesian candidate posterior with a 50/50 clean/one-flip prior. Ten models cover seed 0–4 and two phase splits (full and initial phases 1,n−2 withheld). Source policy scores are rounded to four decimal places before taking the smallest-index argmax, ensuring deterministic tie handling across NumPy and pure scalar replay.

All simulator populations for n=4,6,8 are enumerated. For each learned model, each n evaluates n² clean episodes, n²(n+2) episodes with exactly one substituted bit at every location, and the same number with a single marked erasure. The decision tree **replans** after each received report. The total archive contains **21,640 evaluated episodes** across 90 condition panels and 10 learned models.

## Matched held-out-phase results

| Cycle n | Policy | Clean | One substituted bit | One marked erasure |
|---:|---|---:|---:|---:|
| 4 | Original OIC-21 | 93.75% | 40.83% | 86.46% |
| 4 | OIC-22 learned | **100.00%** | **45.00%** | 86.25% |
| 6 | Original OIC-21 | 92.22% | 43.75% | 80.62% |
| 6 | OIC-22 learned | **93.33%** | **48.26%** | 80.90% |
| 8 | Original OIC-21 | 81.56% | 48.94% | 74.38% |
| 8 | OIC-22 learned | **82.19%** | **50.50%** | 74.19% |

Percentages pool the five seeds and all shifts, initial phases and fault locations. The original OIC-21 panel is the frozen previously executed benchmark; all archived baseline counts and source digests remain pinned. The measured flip improvements are +4.17, +4.51 and +1.56 percentage points. **The preregistered requirement of a ten-point gain at both n=6 and n=8 was not met.** Clean accuracy changes satisfy the five-point-loss constraint. Exact candidate dynamics are supplied to the student; no unprivileged latent transition learning is claimed.

### Full-phase split

| n | Original clean/flip | OIC-22 clean/flip |
|---:|---|---|
| 4 | 98.75% / 41.25% | 97.50% / 46.88% |
| 6 | 92.78% / 45.62% | 93.89% / 46.53% |
| 8 | 82.50% / 48.97% | 82.81% / 49.44% |

## Exact oracle and receding lookahead

The finite exact oracle for an equally weighted clean/one-flip experimental mixture gives 71/90 at n=3,Q=5, 147/192 at n=4,Q=6, and 262/350 at n=5,Q=7. With one flip occurring at a uniformly selected query index and no clean episodes, the best expected accuracies are 34/45, 73/96 and 126/175, respectively. An independent fixed-schedule enumeration checks the branch accounting, which preserves leaf multiplicity when different receipts yield the same belief state.

A separate **exploratory** four-step receding planner uses one clean versus three one-flip objective units, evaluating the full true population at Q=n+2. It yields:

| n | Clean | One flip | Erasure | Planning cost |
|---:|---:|---:|---:|---|
| 6 | 88.89% | 55.56% | 76.74% | exact cyclic belief and 4-ply tree search per decision |
| 8 | 82.81% | 53.44% | 74.22% | exact cyclic belief and 4-ply tree search per decision |

At n=6, this planning reference clears the ten-point flip improvement gate against OIC-21 with a 3.33-point clean drop, but at n=8 it remains short. It has much higher inference computation than the learned scorer and is therefore a **resource-unmatched upper-effort comparator**, not a replacement for the main matched training result.

## Exact worst-case obstruction and communication costs

OIC-T36 proves that universal full-shift identification under at most e substitutions and r marked erasures needs at least Q=n(e+1)+r one-bit queries. At the present n+2 budget this is impossible for a single substitution and n≥4 regardless of the learned policy. The theorem is a lower bound; sufficiency at Q=2n is open for this interface.

| n | Registered Q | Required Q for one-flip guarantee (necessary) | Online response+address bits |
|---:|---:|---:|---:|
| 4 | 6 | ≥8 | 18 |
| 6 | 8 | ≥12 | 32 |
| 8 | 10 | ≥16 | 40 |

The address costs assume direct online transmission of each selected marker label. The 145 FP32 learned weights occupy 4,640 parameter bits; two exact mismatch masks occupy 2n² abstract live bits. Teacher search, numeric feature workspace, and repeated inference operations are accounted separately. Matching 500 optimizer steps is insufficient to establish equal total resource cost.

## Verification

- `oic22_exact.py` contains the exact finite Bellman solver for one substitution, one marked erasure, and their registered clean/flip mixture.
- `oic22_reference.py` writes and checks the pinned finite oracle receipt, with exact integer numerators.
- `oic22_filter.py`, `oic22_train.py`: exact mismatch belief, stable deterministic action tie rule and 145-param model training.
- `results/oic22_fits.json.gz.b64` preserves weights, train metadata and all registered panel totals for ten fits. `results/oic22_summary.json` records source digests, population denominators and the frozen OIC-21 control comparison.
- `oic22_verify.py` independently reconstructs policy decisions **without PyTorch or NumPy**, replays every registered trial, checks source and receipt SHA-256 digests, and recomputes all panel totals. Full deep replay covers 90 panels; the lightweight CI path replays 12 panels selected before inspection.
- `test_oic22_exact.py` independently enumerates the received-code histograms of fixed schedules, tests exact fault-mask states and checks the all-zero lower-bound witnesses. `test_oic22_receipt.py` validates model receipts and phase accounting.
- `oic22_lookahead_run.py` stores the optional computationally costlier receding-planner control in `results/oic22_lookahead.json`.

Run from repository root:

```sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic22*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic22*.py' -v
python research/observational_inwardness_v1/oic22_reference.py --check
python research/observational_inwardness_v1/oic22_verify.py --check
# More exhaustive independent scalar replay:
python research/observational_inwardness_v1/oic22_verify.py --deep
# Requires NumPy for high-compute optional planning reference:
python research/observational_inwardness_v1/oic22_lookahead_run.py --check
# Requires torch and NumPy; repeat each registered seed/split for optimizer reproduction:
python research/observational_inwardness_v1/oic22_train.py --seed 0 --split heldout --out /tmp/oic22_fit.json
```

## Interpretation

The OIC-22 learned policy improves average resilience modestly while retaining clean classification. The exact game supplies a stronger teacher than the student captures, and the fault channel doubles the necessary worst-case query count before universal success is even possible. The architectural known cyclic hypothesis model remains a major prior. Results concern the synthetic finite observation game and establish nothing about deployed neural models or phenomenal consciousness.
