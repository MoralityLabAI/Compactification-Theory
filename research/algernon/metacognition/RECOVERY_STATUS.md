# H-M0/H-M1 executable recovery status

This branch is the direct import of the preserved ChatGPT sandbox experiments into Algernon. GitHub blob IDs for the files below were checked against the original ZIP bytes, not rewritten implementations.

| Experiment source | Original Git blob SHA | On GitHub |
| --- | --- | --- |
| `experiments/toy_contracts.py` | `16a92153570e943c4b2e4287bcd7f8ab66c309bb` | yes |
| `experiments/tests/test_toy_contracts.py` | `21805955f4ab283cc0501d608081b177fba57192` | yes |
| `experiments/h_m0_pruner.py` | `d62a43c27f5fa6c1f67208c5f52ae5f5678bbd9f` | yes |
| `experiments/h_m0_terminal.py` | `8e601a330f6e11610ee89e63a4a57689f22ee3ad` | yes |
| `experiments/h_m0_fair_baselines.py` | `36ce308cbc1a22b014d87578be8a864a5d8c9760` | yes |
| `experiments/h_m1_revision.py` | `d438519c0a79aad649302d27704be1a6200efef7` | yes |
| `experiments/tests/test_h_m0_pruner.py` | `f94a8a31cb9277f7b8323cc9241f64ac3ef88c90` | yes |
| `experiments/tests/test_h_m0_terminal.py` | `f71f94145fe7ae1b9cf11dc6786d6fa4e9ecc4cb` | yes |
| `experiments/tests/test_h_m0_fair_baselines.py` | `9628e47be2cfce37ace36770031de16547c31284` | yes |
| `experiments/tests/test_h_m1_revision.py` | `5e1eaec37744423424466e8061638179f560e49c` | yes |
| `experiments/benchmark_terminal_runtime.py` | `ffb3d462bf813cd6ad99b9ff51e73193700d4744` | pending; optional timing script |

The original archive is SHA-256 `88d8e23581f0445aca9ac7b7b9a9c7f8d7df34f00bbc5126fced84511053f7b4`. H-M0/H-M1 original binary weights and full JSONL records remain outside GitHub until a separately verified artifact transfer. **H-M2 remains unverified and requires a new frozen protocol.**

## Commands (CPU)

Use a detached checkout of this branch. From the repository root:

```powershell
$env:PYTHONPATH = "research/algernon/metacognition"
python -m unittest discover -s research/algernon/metacognition/experiments/tests -v

# H-M0 initial policy; creates original predecessor checkpoint in the expected location.
python -m experiments.h_m0_pruner --output research/algernon/metacognition/results/h_m0_01 --train-states 4000 --test-episodes 1000 --epochs 18 --seed 20261103

# H-M0 terminal policy. Expected predecessor checkpoint is results/h_m0_01/policy.pt
# under metacognition/, so copy or symlink the generated initial checkpoint into that layout.
python -m experiments.h_m0_terminal --output research/algernon/metacognition/results/h_m0_02 --train-worlds 700 --validation-worlds 150 --test-worlds 1000 --oracle-test-worlds 100 --rollouts 2 --epochs 18 --seed 20261104

# Second independent held-out stopping comparison.
python -m experiments.h_m0_fair_baselines

# H-M1 grammar-revision study, self-contained and independent of H-M0 checkpoint.
python -m experiments.h_m1_revision --out research/algernon/metacognition/results/h_m1_01 --train 2400 --val 350 --test 1200 --confirm 1000 --epochs 28 --seed 20261103
```

The source imports use `experiments`; `PYTHONPATH` makes that package visible. The generated outputs are research artifacts and should be excluded from normal Git commits. Preserve exact seed ranges and do not tune on the final test/confirmation ranges.

**Scientific status:** the code and tests are on GitHub; results require independent rerun, and old observed measurements await raw-record publication. These are synthetic CPU studies only, distinct from Qwenputer and Qwen3.5 graft benchmarks.
