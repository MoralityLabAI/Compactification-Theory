# COE-04: measured recurrent quantizer capacity on frozen tied attention

**Executed:** 2026-11-15. **Status:** CPU-trained small causal-attention checkpoint, frozen before probe fitting, with exact offline replay. Companion mathematical theorem in `patrickdugan/MathCan/research/compactification/TEMPORAL_OBSERVER_CAPACITY.md`.

## Why this experiment

COE-01 to COE-03 show that symbolic observer width depends on evaluator edges, metric constraints, precision and the distinction between soft and binary cuts. COE-04 asks whether successive residual states of one *frozen shared-weight attention model* can supply useful binary features from **one tied readout**, and whether that benefit survives a matched total-bit comparison.

The source design follows the pre-norm, tied-block causal visit-state equations in `MoralityLabAI/LoopedTransformers/research_gym/neural/looped_decoder.py`: a 24-dimensional two-token embedding, three attention heads, one physical shared block, pre-norm attention plus GELU MLP, four unrolled visits, `alpha=1`, `beta=0.7`. `run_frozen_loop.py` is an independently executable small implementation. This is a **locally trained synthetic checkpoint**, rather than the external TRM, Huginn or Nanbeige weights discussed in LoopedTransformers. We examined their public experiment receipts; the current environment had no directly loadable pretrained model-weight artifact.

## Frozen task and split

The training task is to copy the first symbol `a` from input `(a,b)` with `a,b in {0,...,7}` into the model's final-token logits. Every one of the 64 symbolic pairs is trained for 160 full-batch AdamW steps at fixed learning rate `0.01`, with independently fixed random seeds `2406,2407,2408`. This is a small supervised associative task; it is **not a held-out language task**. All 64 predictions were correct for each seed before probes were selected. The model weights were frozen and replayed to produce hidden states of shape `[64,4,24]` per seed.

For the *separate probe fit*, distractor contexts `b in {0,1,2,3}` form the training set; contexts `{4,5,6,7}` are held out. The evaluator graph `K8` is fixed by the source-token identity `a`, independently from readout codes. At each held-out context, the 28 unordered pairs of different source labels are the required distinctions. Each seed therefore has exactly 112 held-out evaluator pairs.

A shared readout is one fixed linear threshold `(w,theta)` applied at the query-token residual on visits 1,2,3,4, giving a four-bit transcript. `temporal_observers.py` generates a deterministic pool of hyperplanes from *probe fitting data alone*, maximizes the number of correctly distinguished training pairs, then evaluates the unchanged head on heldout contexts. The last-visit 1-bit baseline uses the same candidate family and selection objective. The final 4-bit control greedily chooses four readouts only from the fitting data; an untied per-visit control chooses four separate heads. Each result is evaluated on identical held-out contexts. Selected heads and raw integer counts are stored in JSON receipts.

### Source provenance and exact replay

- Synthetic model source: `MoralityLabAI/LoopedTransformers` `research_gym/neural/looped_decoder.py` at source commit `bb51f116acf7ebd4a8f337ec9ab54475f3e0cb1d` (November 15, 2026). This self-contained reproduction follows its tied pre-norm block equations and causal attention. It is not an invocation of the upstream class.
- Training runtime: Python 3.13, `torch 2.10.0+cpu`, deterministic CPU operations, one thread.
- `frozen_loop_weights.json.gz` and `frozen_loop_activations.json.gz` store all three original unrounded float32 weight arrays and all 768 residual vectors.
- Source checkpoint SHA256 values (canonical JSON of each state_dict): `2406`: `13b9c3c63f7d7d27ef9aac2144fd2dd5469e144d940bee97e0a975a38e1d1841`; `2407`: `46a99c7a348d2d0250ff9f7289ed2c3078cd847f506a65941d24dd5213e217ee`; `2408`: `5bb8809a753e08f5ade9f3ff40b3ec50cecdb2aa8085c3a2624506d201196a88`.
- Optional torch audit reloads the frozen weights and obtains every stored residual to *exactly zero absolute difference* under the original CPU environment; no probe training is involved in this audit.
- CI runs the Python standard-library verifier against compressed artifacts, signed fit-only readout vectors, full held-out evaluator enumeration, train/test split integrity, tied transducer constructions and integer code optimality cells. The costly upstream checkpoint regeneration is optional (`python run_frozen_loop.py`, then `python audit_model_alignment.py`).

## Primary measurements

| Seed | Last visit, 1 bit | Tied head, 1 bit × 4 visits | Tied head, 2-bit uniform quantization × 4 | Final, 4 independently fitted bits | Untied heads, 1 bit × 4 visits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2406 | 64/112 | **90/112** | 102/112 | 112/112 | 105/112 |
| 2407 | 62/112 | **72/112** | 86/112 | 106/112 | 111/112 |
| 2408 | 61/112 | **79/112** | 97/112 | 111/112 | 111/112 |
| Pooled | **187/336 = 55.65%** | **241/336 = 71.73%** | **285/336 = 84.82%** | **329/336 = 97.92%** | **327/336 = 97.32%** |

The repeated-last snapshot control gives exactly the same edge retention as last visit 1-bit; repeating a frozen bit cannot increase its distinctions. The tied temporal observer beats that per-tick baseline for all three model seeds. Under the **matched total output-bit bandwidth** of four, a final-layer readout with four independent heads performs much better. A temporal tied head has one fitted direction whereas the four-head final baseline has four, and four temporal observations require storing a four-bit history. These are different engineering budgets and are reported separately.

### Post-hoc diagnostics, executed without refitting on held-out data

1. **Count-only collapse.** For every selected tied one-bit head on each seed, reducing its four ordered observations to the count of positive readings preserves *exactly the same* held-out edge set: `90,72,79` hits. The order provides **zero additional pairwise distinctions** in these cells. This narrows the interpretation of the temporal benefit to a coarse crossing-count signal. In the general theorem, a count-only `T=4` bit observer has at most five possible output categories, compared with 16 distinct ordered histories. For an eight-state complete evaluator it can preserve at most `25/28` edges per context. These finite observations fall below that bound.
2. **Hard Hamming diameter-two constraint.** The main temporal heads reach a maximum pairwise Hamming distance of **four**, and therefore do not satisfy the COE-02 global diameter-two condition. A separate fit-only re-selection imposed `d_H <= 2` for all training evaluator pairs. Three resulting probe heads retained `79,59,74` held-out edges respectively, and also respected maximum diameter two on all held-out contexts. `cap2_receipt.json` stores the independently scored witnesses. This is explicitly post-hoc.
3. **Layer separation.** Across held-out evaluator pairs, mean Euclidean separation increases with visit depth in all three local checkpoints (seed 2406: `11.47,26.89,43.74,60.96`; seed 2407: `11.85,26.70,42.28,58.04`; seed 2408: `14.52,32.72,51.88,71.28`). These data do not establish an alpha-contractive recurrence, so the common-fixed-point late-observation theorem does not apply to this run.
4. **One-bit final versus temporal precision.** A two-bit quantizer fitted with the same temporal direction achieves `102,86,97` hits; the final-step two-bit baseline achieves `82,83,83`. Its quantization bins were calibrated on training data alone. Comparisons of bits and readout parameters remain separate.

## Model and proof separation

The task-domain labels are known during fitting, and all contexts appeared during base-model training. This checks **probe generalization across contexts on a fixed small synthetic model**, rather than generalization of a pretrained model to unfamiliar tasks or causal control of a trained language model. There are three model seeds and the evaluated pair indicators are correlated; pooled fractions are descriptive, with no independence-based p-values. A learned threshold readout is an observational probe and does not establish causal necessity. The metric-budget theorem is a conditional property of the codes; its hypotheses are not silently asserted of the unconstrained trained heads.

## Reproduce

From the Compactification-Theory root, standard library Python 3.10+:

```bash
python -m unittest discover -s research/chromatic_observer_gym_v1 -p 'test_*.py' -v
```

Regenerate the probe receipts from the frozen states (CPU, standard library only):

```bash
cd research/chromatic_observer_gym_v1
python temporal_observers.py --input frozen_loop_activations.json.gz --out /tmp/coe04_temporal.json
python run_temporal_baselines.py --input frozen_loop_activations.json.gz --out /tmp/coe04_equal.json
python run_cap2_control.py --input frozen_loop_activations.json.gz --out /tmp/coe04_cap2.json
```

Optional CPU torch replay (`torch 2.10.0+cpu` is the reference version):

```bash
python run_frozen_loop.py --out frozen_loop_activations.json
python audit_model_alignment.py
```

The regeneration script overwrites uncompressed local outputs only; review hashes before updating frozen Git artifacts. The references to native LoopedTransformers are architecture/provenance links; this study does not claim checkpoints were downloaded from or trained by that repository.
