# DCF-16 results

## Dataset

920 verified records (train 600, val 120, heldout 200). `verify16.json` records 0 re-derivation mismatches and 0 canonical-hash overlaps between splits. Six families occur only in heldout; the shared families occur in heldout only at n = 48 and 64. Hard negatives, where the unary-evidence or one-hop representation, or BP, decides differently from the exact posterior: 324 train, 62 val, 117 heldout.

The label space shifts on heldout: `gf2` (exact, noise-free parity systems) and `particle` occur as selected operators only in heldout. A model trained on train/val has never seen them as targets.

## Integrations (pushed)

**Data Goblin**, `master` @ `d7d49bd`:
- `schemas/dcf16_decision_trace.schema.json` (draft 2020-12; validates all 920 records);
- `src/datagoblin/datasets/dcf16.py`, a loader that raises `SplitHygieneError` on held-out families in train/val, sizes above 32, mismatched split tags, or ids repeated across splits;
- seven supervision views that never read heldout unless explicitly allowed;
- docs, config and fixtures; 12 tests pass.

**EvalMath**, `main` @ `43f08e5`:
- `src/spiral_ln/evaluation_dcf16_decision.py` scores a predictions file against heldout, with exact `Fraction` arithmetic from the stored floats;
- metrics: verified EU, realized utility, regret, coverage, selective risk, algorithm accuracy, budget compliance taken from the record (not the prediction), Brier/ECE, the hard-negative slice, per-family results, and paired comparison;
- receipt `output/evaluation_dcf16_decision/dcf16_eval_v0_1.json` records the heldout SHA-256; 17 tests pass and `check_receipts.py` verifies 32/32 receipts.

## Reference predictors on heldout (EvalMath receipt)

These use record fields only: no reference posterior, no truth bit.

| Predictor | Verified EU | Coverage | Selective risk | Algorithm accuracy | EU on hard negatives |
|---|---:|---:|---:|---:|---:|
| oracle (verified label) | .4455 | — | — | 1 | — |
| symbolic: cheapest exact within budget, else mini-bucket, else abstain | .3524 | .535 | .047 (5/107) | .505 | .5415 |
| always abstain | 0 | 0 | — | .510 | 0 |
| unary-evidence shortcut | .0097 | .075 | .200 | 0 | −.0529 |

Symbolic regret against the oracle is .093. Symbolic beats the shortcut 92 to 10 in paired records (98 ties). On the hard-negative slice the shortcut has negative utility while symbolic earns .54: the slice isolates exactly the cases where plausible-looking local reasoning is insufficient. Algorithm accuracy is a poor headline metric here, since always-abstain scores .51 on it; verified EU and selective risk are the metrics to report.

No model was trained on the curriculum in this session. The baseline to beat on heldout is symbolic EU .352 at selective risk .047.
