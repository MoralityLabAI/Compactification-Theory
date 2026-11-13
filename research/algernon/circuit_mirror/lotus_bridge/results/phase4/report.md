# Algernon Phase 4 summary (Phase 5 report): constraint-guided correction of D1's latent slots

This report collects the seven Phase 4 runs (H-A3 to H-A3g, 2026-11-06 to 2026-11-12) into one account.
- Every number here comes from the runs' committed receipts.
- [metrics.json](metrics.json) is regenerated from those receipts by `lotus_bridge/phase4_summary.py`. A CPU test checks that it still matches them.
- Each run was registered in [EXPERIMENT.md](../../../EXPERIMENT.md) before it ran. Its own report gives the details.

## What Phase 4 asked

The brief's recurrence is h_{k+1} = F(h_k) + G(c(pi(h_k))). It asks three things:
- whether a correction driven by the machine's own transition rule repairs a latent state after damage, compared with plain recurrence at matched compute;
- that surrogate penalties stay separate from exact finite-field verification;
- that no correction signal reveals the gold successor unless an equivalent control receives the same information.

**F** is 02D's arm D1, frozen (`80eda0824cff`): GPT-2 small, trained with LOTUS's recipe to refine six fixed latent slots once per loop. It answers every roster case at depths 1-16 (02D) and its slots are causal (02E).

**pi** is D1's own LM head read on slot 2: a distribution p over the six states.

**Damage:** isotropic Gaussian noise on all six slots after loop j in {4, 8, 12}, four seeds per case.

**The arms.** All four run F for the same 17 passes (initial + 16 loops). A corrector adds one linear map per loop: 27,648 to 59,904 parameters against GPT-2's 124M.

| arm | correction input | what it knows |
| --- | --- | --- |
| P | none | plain recurrence |
| C | e - p, where e = p_prev M is the public map applied to the previous loop's decode | the transition residual |
| I | [e, p], combined freely | the same information as C. G(e - p) is G([e, p]) with tied weights |
| N | p | the decode, without the map |

**Information.** e reveals the successor of the previous decode. That equals the gold successor only when the previous decode is right. I receives exactly the same e and p, so C's comparison with I (rule R2) is information-matched. N is the control without the map.

**Exact verification.** The exact checker is a BN254 R1CS built from the public machine (`spec.build`), evaluated on pi's 16-step decoded trace (`correct.trace_valid`).
- It is called only in evaluation and never enters a training loss.
- The training losses are surrogates: pi's cross-entropy at loops j and j + 1, plus, under `--loss answer`, the answer digit's cross-entropy.

**Rules** (case bootstrap over 72 cases, lower bound > 0 and difference >= margin):
- R1: C - P >= 0.10
- R2: C - I >= 0.05
- R3: C - N >= 0.10
- R4: I - P >= 0.10 (from H-A3d on)

The instrument requires P to answer all 72 clean cases with every clean trace accepted; it held in every run. The 0.97 clean floor is reported, not scored.

## The seven runs

Primary accuracy, 864 runs per arm (72 cases x 3 noise loops x 4 seeds):

| run | change from the previous run | sigma | P | C | I | N | clean C / I / N | rules met |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| [03](../algernon-03/report.md) | first registration: unit-scale correctors with bias, local loss | 16 | 0.494 | 0.487 | 0.505 | 0.500 | 1.000 / 1.000 / 1.000 | none |
| [03b](../algernon-03b/report.md) | corrector output in noise units (kappa = 110.78) | 16 | 0.494 | 0.527 | 0.528 | 0.501 | 0.556 / 0.556 / 0.681 | none |
| [03c](../algernon-03c/report.md) | bias-free correctors | 16 | 0.494 | 0.461 | 0.888 | 0.499 | 1.000 / 0.889 / 0.722 | none |
| [03d](../algernon-03d/report.md) | one fixed noise scale for every arm | 16 | 0.500 | 0.487 | 0.455 | 0.506 | 1.000 / 1.000 / 0.917 | none |
| [03e](../algernon-03e/report.md) | sigma = 1, where noise damages the state without wiping it | 1 | 0.626 | 0.639 | 0.804 | 0.546 | 1.000 / 0.958 / 0.986 | R4 |
| [03f](../algernon-03f/report.md) | answer digit's cross-entropy added to the training loss | 1 | 0.626 | **0.811** | 0.714 | 0.561 | 1.000 / 0.944 / 0.972 | **R1, R2, R3** |
| [03g](../algernon-03g/report.md) | 03f at sigma = 16 | 16 | 0.500 | 0.481 | 0.528 | 0.505 | 1.000 / 0.764 / 0.528 | none |

- 03 to 03c share one noise scale that follows each run's own slot RMS. 03d on use one fixed scale, so P's accuracy is 0.494 in the first group and 0.500 at sigma = 16 in the second.
- Within each noise setting, P's records are identical across runs.

**The noise-by-loss grid** (fixed noise, bias-free correctors in noise units):

| | local loss (pi on slot 2) | answer loss (local + answer digit) |
| --- | --- | --- |
| sigma = 16 (plain at its floor, 0.500) | 03d: no rule met. C 0.487, I 0.455 | 03g: no rule met. C 0.481, I 0.528 |
| sigma = 1 (plain 0.626) | 03e: R4 met. C 0.639, I 0.804 | 03f: R1, R2, R3 met. C 0.811, I 0.714 |

## What the runs establish

1. **The transition residual repairs the answer when the damage leaves part of the state and the training loss reaches the answer** (03f).
   - C - P = +0.185 [+0.141, +0.231].
   - C answers all 72 clean cases.
   - It beats the information-matched I by +0.097 [+0.041, +0.154], so the gain comes from the residual form, not from extra information.
2. **The residual repairs pi's read before it repairs the answer.**
   - Under the local loss at sigma = 1 (03e), C's decoded trace passes the exact checker in 0.987 of primary runs, against P's 0.304. Its answer stays at plain level (0.639 against 0.626).
   - The local loss reads only slot 2, and the slots attend causally, so the corrector weights for slots 3-5 never moved from zero.
   - The answer loss trains them (03f). C's answer then follows its trace: 0.813 of its valid-trace runs are right, against 0.638.
3. **No corrector repairs the state at sigma = 16 under either loss** (03d, 03g).
   - A rank-6 additive write does not undo isotropic noise that wipes the slots.
   - The answer loss reaches all six slots there too (slot 3-5 weight norms 2.57 / 3.55 / 2.77) without changing the outcome.
4. **C is the only corrector that never damages clean runs once the bias is removed** (03c-03g: 72 of 72 clean cases every time).
   - e - p vanishes on clean loops, so C pushes only where the decode is wrong.
   - I and N push on every loop. Whenever I helps (03c, 03e, 03f), it costs clean cases (0.889, 0.958, 0.944).
5. **Three confounds were found and removed in turn.**
   - The correctors were about 100x too small to reach the noise (03 → 03b, output in noise units).
   - The bias became a constant push that locked the decode and damaged clean runs (03b → 03c, bias-free).
   - The noise scale followed each arm's own slot RMS, which let I shrink its noise (03c → 03d, fixed scale). I's 0.888 in 03c did not survive the fix: 0.455 in 03d.

## Failures and limits

- **Registered predictions often failed.**
  - In 03f only one of four held (P's records). The other three bet against R1 and R2 being met and on R4.
  - In 03e the prediction that R4 is not met failed.
  - The rules decide the results, and the failed predictions are reported in each run's report.
- **The path was adaptive.**
  - Each run was registered before it ran, but the next run's single change was chosen after reading the previous evaluation: the midpoint noise level after 03d, the answer loss after 03e.
  - Seven runs share one roster of 72 cases, with no correction for multiplicity across runs. 03f's R1 lower bound (+0.141) and R2 lower bound (+0.041) are each taken on their own.
- **One F, one seed, one synthetic task.**
  - D1 is GPT-2 small on 6-node pointer maps.
  - The damage is synthetic isotropic noise at one loop, not an error the model makes on its own.
- **The breakdown point lies between sigma = 2 and sigma = 8** (post hoc):
  - In 03f at 2 sigma_e = 2, C - P = +0.181 [+0.106, +0.264].
  - In 03g at sigma\*/2 = 8, C is 0.495 against P's 0.505.
- **C still answers wrong in about 1 run in 5 in which its decode is right** (03f: decode 0.985, answer 0.811).
- **The exact checker certifies the decoded trace, not the generated answer.** The two diverge (point 2), and the registered rules score the answer.
- **Causal reading.**
  - The corrections are interventions on the slots at loop j, and the arms differ only in G's input and weights, so the comparisons are causal for this F and this noise.
  - The account of why C's answer follows its trace under the answer loss comes from post-hoc reading and a 12-case CPU check, not from a registered test.
  - So does the candidate reason I lost its trace repair under the answer loss (its slots shrink to about half of F's RMS, so the fixed noise is larger relative to them).
- **No Circom binary, witness generator or SNARK proof was executed.** The finite-field check runs the R1CS constraints in Python.

## Receipts and reproduction

| item | where |
| --- | --- |
| per-run manifest: code commit, F and roster hashes, corrector hashes (weights outside git), arguments, seeds, precision (fp32), environment, hardware, commands | `../algernon-03*/manifest.json` |
| per-case records with intervention identities (arm, condition, noise loop j, seed, sigma), answer, prediction, decoded trace, exact-check verdict: 5,472 per run | `../algernon-03*/eval.json`, `result.records` |
| accuracy, trace validity, bootstrap intervals, decisions, corrector sizes, receipt hashes, for all seven runs | [metrics.json](metrics.json) |
| training loss per step | `../algernon-03*/train.json` |

The records stay in each run's eval.json rather than being copied into a second `records.jsonl`. metrics.json holds each eval.json's hash, checked against the run's manifest.

**CPU regression tests** (no GPU). They include `test_phase4_summary`, which regenerates metrics.json from the receipts and compares:

```bash
python -m unittest discover -s research/algernon/circuit_mirror/tests -v
```

With `ALGERNON_LOTUS_UPSTREAM` pointing at the pinned LOTUS checkout and the GPT-2 snapshot cached, the same command also runs the model-path tests on CPU. These cover:
- plain-arm parity with the unhooked model;
- zero-initialised correctors changing nothing;
- noise placement at loop j only;
- the output scale;
- the local loss's gradient path through F.

**Regenerate or check the roll-up:**

```bash
python -m research.algernon.circuit_mirror.lotus_bridge.phase4_summary --check
```

**GPU smoke** (two training steps per arm under the H-A3f recipe, in a scratch directory holding a copy of H-A3's `calibrate.json` and `scale.json`):

```bash
vramorama run --vram 4G -- python -m research.algernon.circuit_mirror.lotus_bridge.correct --upstream <lotus checkout> --base <gpt2 snapshot> --checkpoint <D1 final.pt> --roster research/algernon/circuit_mirror/lotus_bridge/frozen_roster_6x16.json --out <scratch> --out-scale noise --corrector-bias off --noise-ref fixed --sigma-rule midpoint --loss answer --steps 2 train
```

Run 2026-11-13 on the RTX 5080: exit 0 after 224 s, including the wait for the card. C, I and N each wrote a checkpoint after step 2, with losses 9.98, 10.03 and 10.13. The smoke covers training only; `eval` has no small mode and runs the full roster.

Full runs use the commands in each run's manifest: `train`, then `eval`.
