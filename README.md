# Compactification Theory

The new [Pluto-Gym research package](research/pluto_gym/README.md) adds a runnable
interface-relative gym, a compact CPU student with bounded DRAM replay, and
curriculum experiments with a configurable designer webhook. Its pinned
TensorCraft adapter reuses the existing StarCraft surrogate, frozen-Pluto farm
and trace checks.

How many bits does a trained adapter store per parameter, and what limits it? This repository tests a theory note on that question against LoRA adapters on Qwen3-0.6B. Everything runs on CPU in RAM.

The note, [`note/bits-depth-lipschitz.html`](note/bits-depth-lipschitz.html), works out these claims for ReLU networks approximating Hölder functions:

- **Total bits bound capability.** W weights of b bits can store at most W·b bits, so error ε ≳ (W·b)^(−s/d) for any model. No Lipschitz assumption is needed.
- **Per parameter, capability is not bounded by itself.** It is bounded only through the effective bits per weight, log2(R·K/ε). Here K is the Lipschitz constant of the map from weights to function.
- **For bounded weights, that effective precision is a depth quantity.** The classical W^(−s/d) barrier holds exactly when effective precision is O(log W). The W^(−2s/d) ceiling for ReLU comes from VC dimension, not from any Lipschitz constant.

Open the HTML in a browser; its maths is typeset with MathJax from a CDN. Every claim is labelled as proved, cited (opened or recalled) or conjectured. Its §4 lists predictions for small-network experiments, and its §5 reports this repository's results.

## The experiment

**Task.** M random keys (`"482910 ->"`), each labelled A or B at random. The labels are M fair bits, the language-model version of the note's random-sign packing. An adapter gets an item right when the logit of its label beats the other one.

**Exact and in RAM.** The frozen model runs once per item to cache the hidden state the adapter needs, and is then freed. Training and every measurement recompute only the adapted part of the model, which is that part's exact computation.
- [`src/adapter_bits.py`](src/adapter_bits.py) adapts the last layer's MLP at the answer position. Its output matches the full model's margin to 2e-6.
- [`src/adapter_bits_deep.py`](src/adapter_bits_deep.py) puts last-token LoRA on `gate_proj` in any of layers 24 to 27. It caches the keys and values of earlier, unchanged positions in fp16, and matches the full model to 2.4e-4.

**Measured per run:**

- **Stored information** I = M(1 − H2(err)), a Fano estimate of the bits the adapter holds about the labels.
- **I_b**, the same after rounding every adapter weight to b bits, for b from 1 to 16. The counting bound I_b ≤ W·b + 32·(number of tensors) is checked on every run.
- **The knee**: the fewest bits per weight that keep the error within one point.
- **Two predictions of the knee from per-item gradients.** The worst case uses the ℓ∞ Lipschitz constant, as in the note. The typical case treats rounding errors as independent noise on the margins.
- **The rank of the items × parameters gradient matrix.** Cover's theorem caps a linearized adapter near twice that rank.
- **The top Hessian eigenvalue** of the training loss.

Adapters:
- `probe`: a linear readout on k whitened principal directions of the final hidden state, the baseline with known capacity.
- `lora:<matrix>`: low-rank factors on last-layer MLP matrices.
- `deep-gate`: last-token gate LoRA spread over several layers.

## Results

Single seed, Adam, 2,000 to 4,000 steps, CPU. Full tables are in `results/<run>/report.md` and raw records in `results/<run>/results.jsonl`.

### What sets single-layer capacity (`results/sweep-1`, 36 runs)

| adapter | W | best stored bits | bits / parameter | knee (bits/weight) | 2 × gradient rank at 1e-3 |
|---|---|---|---|---|---|
| whitened probe, k = 64 | 64 | 128 | 2.0 | 7 | 128 |
| whitened probe, k = 512 | 512 | 557 | 1.09 | 8 | 1,020 |
| gate LoRA, rank 1 | 4,096 | 256 | 0.063 | 7 | 252 |
| gate LoRA, rank 2 | 8,192 | 381 | 0.047 | 16 | 292 |
| gate LoRA, rank 4 | 16,384 | 324 | 0.020 | 10 | 390 |
| gate LoRA, rank 8 | 32,768 | 493 | 0.015 | 10 | 646 |

- **The counting bound held on all runs at every rounding level**, as it must.
- **The linear probe stores bits in proportion to its size, as Cover's theorem says.** It reaches 2 bits per parameter.
- **Last-layer LoRA does not.** Eight times the parameters gave about twice the bits, a log-log slope of 0.26.
  - Its stored bits stay within a factor of 1.3 of twice its gradient-matrix rank at 1e-3.
  - The limit is how varied the frozen model's features are at the adapter site. The answer-position residual has an effective rank of about 12 across keys.
- **Precision goes to conditioning, not information.** LoRA needs 7 to 16 bits per weight to keep at most 0.06 bits per parameter.
- **Overloading costs information the adapter could have kept.** With 2,048 to 4,096 labels a LoRA adapter ends with 40 to 240 bits, less than it keeps with 256 to 512.
- **The knee follows first-order sensitivity.** The typical-noise model predicts the probe's knee within 2 bits on 12 of 12 runs, and exactly on 7. The worst-case ℓ∞ prediction overshoots the probe's full-fit runs by 3 to 4 bits, which is ½·log2 W, the gap the note predicted.

### Placement and spread at fixed W = 16,384 (`results/deep-2`, `results/deep-3`)

All rows use last-token gate LoRA.

| placement | stored bits at M = 512 | at M = 2,048 | at M = 8,192 | knee | top Hessian eigenvalue | gradient rank at 1e-2, init → trained |
|---|---|---|---|---|---|---|
| rank 4 in layer 27 | 220 | 106 | — | 8 | 1,297 | 22 → 16 |
| rank 2 in layers 26, 27 | 512 (all) | 782 | — | 10 | 141 | 28 → 207 |
| rank 4 in layer 24 | — | 2,048 (all) | 848 | 8 | 5 to 20 | 37 → 509 |
| rank 1 in each of layers 24–27 | 512 (all) | 2,048 (all) | **3,284** | 7 to 8 | 8 to 51 | 35 → 463 |

`deep-2` ran 2,000 steps. `deep-3` (the rows with M = 8,192, and layer 24 alone) ran up to 4,000. Neither M = 8,192 run converged before that cap, so the comparison there is at equal compute.

- **Placement matters most.** The same adapter moved from layer 27 to layer 24 goes from 106 bits to all 2,048 labels. Layer 27's features at the answer position are too uniform across keys for an adapter there to separate them.
- **Spreading adds more on top.** At M = 8,192, rank 1 in each of four layers stored 3,284 bits (0.20 per parameter), 3.9 times the 848 bits of the same parameters in layer 24 alone.
- **This is not the theory's route through depth.** Bit extraction gains capacity through more bits per weight and sharp, high-K solutions. Here every high-capacity solution needed 7 to 8 bits per weight and was 25 to 250 times flatter than the layer-27 one.
- **What grew was the gradient matrix's rank**, from 16 in layer 27 to 385 to 509 whenever layer 24 is adapted.
  - In `deep-2`, stored bits track twice that rank, as in `sweep-1`.
  - In `deep-3` the rank at 1e-3 reached the 512 items sampled, so this check cannot tell those runs apart. A larger gradient sample is needed.
- **The note conjectured that trained adapters would not use depth.** That is wrong as stated, since spreading helped. It holds only in the narrower sense that they do not spend depth on precision.

### Follow-ups in progress

[`docs/followups.md`](docs/followups.md) records predictions written and pushed before running them:
- whether plain gradient descent's stability limit (sharpness ≤ 2/η) caps what an adapter can store;
- seeds 1 and 2 of the main comparisons;
- where the four-layer adapter saturates.

### Pilots

- **`results/pilot-2`** used the template `"Item {key} belongs to group"` and a probe on raw random directions.
  - Its answer-position features have effective rank 4.7. Plain Adam could not fit even M = k with that probe, though Cover's theorem guarantees the labels are separable.
  - That led to whitening the probe and to switching to the template `"{key} ->"`. The template comparison is `src/adapter_bits_templates.py`.
- **`results/pilot-3`** is the first run on the current setup.

## Running it

Python 3.12 with the packages in `requirements.txt`. The model is `Qwen/Qwen3-0.6B` at revision `c1899de2`, loaded in fp32 on CPU, about 5 GB of RAM while caching.

```bash
pip install -r requirements.txt
python src/adapter_bits.py --pool 4096 --steps 3000 --lora-lr 0.03 --name sweep-1
python src/adapter_bits_deep.py --first 24 --pool 2048 --items 512,2048 --specs "27:4;24:1,25:1,26:1,27:1;26:2,27:2" --lr 0.03 --name deep-2
python src/adapter_bits_deep.py --first 24 --pool 8192 --items 2048,8192 --specs "24:4;24:1,25:1,26:1,27:1" --lr 0.03 --steps 4000 --name deep-3
python src/adapter_bits_report.py runs/sweep-1
```

- Runs go to `runs/<name>/results.jsonl`; set `ADAPTER_BITS_OUT` to change that. Set `ADAPTER_BITS_HF_CACHE` to use an existing Hugging Face cache, and `HF_HUB_OFFLINE=1` to stay offline.
- `--plan` takes entries like `probe:256:512,lora:gate:1:2048` (adapter, size, items).
- `--cache-file` keeps the residual cache on disk between invocations.
- `--wait-commit-gb` makes a Windows job wait for memory headroom before loading the model. This is for shared machines.
- Expect about 3 hours for `sweep-1`, 2 hours for `deep-2` and 3 hours for `deep-3` on an 8-thread laptop CPU.

## Layout

```
note/bits-depth-lipschitz.html   the theory note (open in a browser)
src/adapter_bits.py              single-layer adapters, all measurements (measure())
src/adapter_bits_deep.py         last-token LoRA over several layers, reuses measure()
src/adapter_bits_report.py       results.jsonl -> report.md
src/adapter_bits_templates.py    prompt-template comparison by feature spread
src/config.py, src/qwen_suffix.py model pin, paths, model loading, layer runner
results/<run>/                   raw records and reports of the runs described above
```
