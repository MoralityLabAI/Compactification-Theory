# Bonsai 2 27B on Prime Intellect v2 Verifiers: Baseline vs. Goose 1.53.0

## Hardware & Provenance
- **Model:** `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (native 1.58-bit ternary, `5.54 GB`, SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`)
- **Runtime:** `llama-prism-b10743` (`0.2.0-dev`, commit `adfffbe41`), **100% pure GPU** (`-ngl 99`, `0` CPU offload, `~6.5 GiB` VRAM on RTX 5080 Laptop 16 GB, `-ctk q8_0 -ctv q8_0 -fa on`, greedy `temp=0`, `seed=523`)
- **Harnesses:** Official Prime Intellect `verifiers 0.4.1` (`verifiers.v1` Tasksets) and `verifiers 0.1.11` (`vf-eval`, `c=1`, `r=1`), paired across identical task indices for `/raw` (bare baseline) and real `goose.exe 1.53.0` (`--no-session --no-profile`, with `python_calc` MCP on quantitative/math Tasksets).

## Matched Baseline vs. Goose 1.53.0 Results

| Hub Benchmark | Runner | Goose Mode | N | Baseline Avg Reward | Baseline Solved (Wilson 95% CI) | Goose Avg Reward | Goose Solved (Wilson 95% CI) | Delta Reward |
|---|---|---|---:|---:|---|---:|---|---:|
| `dave-quant/backtest-qa-v1` | verifiers 0.4.1 (verifiers.v1 Taskset) | `goose+python_calc` | 12 | 1.0000 | 12/12 (100.0%, [0.76, 1.00]) | **1.0000** | **12/12** (100.0%, [0.76, 1.00]) | +0.0000 |
| `primeintellect/aime25` | verifiers 0.4.1 (verifiers.v1 Taskset) | `goose+python_calc` | 4 | 0.0000 | 0/4 (0.0%, [0.00, 0.49]) | **0.2500** | **1/4** (25.0%, [0.05, 0.70]) | +0.2500 |
| `moralitylab/bitagent-dag-ops-v2` | verifiers 0.1.11 (vf-eval) | `goose 1.53.0` | 12 | 0.0200 | 0/12 (0.0%, [0.00, 0.24]) | **0.4825** | **0/12** (0.0%, [0.00, 0.24]) | +0.4625 |
| `medarc/head-qa-v2` | verifiers 0.1.11 (vf-eval) | `goose 1.53.0` | 12 | 0.7500 | 9/12 (75.0%, [0.47, 0.91]) | **0.6667** | **8/12** (66.7%, [0.39, 0.86]) | -0.0833 |
| `hunterbown/fluxem-tools-env-v2` | verifiers 0.1.11 (vf-eval) | `goose 1.53.0` | 8 | 0.6156 | 0/8 (0.0%, [0.00, 0.32]) | **0.5472** | **0/8** (0.0%, [0.00, 0.32]) | -0.0684 |

## Telemetry (Tokens, Turns, MCP Calls, and Pure-GPU TPS)

| Benchmark | Arm | Turns | LLM Calls | Goose MCP Calls | Prompt Tokens | Completion Tokens | Mean / Peak Prefill (tok/s) | Mean / Peak Decode (tok/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `backtest_qa_v1` | `baseline` | 12 | 12 | 0 | 1,376 | 953 | 272.47 / 356.61 | 21.02 / 27.26 |
| `backtest_qa_v1` | `goose` | 12 | 29 | 5 | 20,045 | 380 | 359.51 / 595.01 | 34.34 / 47.83 |
| `aime25` | `baseline` | 6 | 6 | 0 | 701 | 4,096 | 167.06 / 322.02 | 10.27 / 17.09 |
| `aime25` | `goose` | 10 | 20 | 6 | 15,736 | 4,057 | 313.19 / 547.01 | 26.51 / 48.98 |
| `bitagent_dag_ops_v2` | `baseline` | 12 | 12 | 0 | 11,797 | 1,827 | 619.21 / 681.2 | 50.41 / 51.9 |
| `bitagent_dag_ops_v2` | `goose` | 12 | 24 | 0 | 32,412 | 954 | 574.01 / 697.9 | 19.44 / 23.37 |
| `head_qa_v2` | `baseline` | 12 | 12 | 0 | 1,464 | 84 | 375.28 / 424.85 | 42.7 / 47.34 |
| `head_qa_v2` | `goose` | 12 | 24 | 0 | 11,844 | 163 | 292.03 / 515.62 | 19.5 / 24.9 |
| `fluxem_tools_env_v2` | `baseline` | 22 | 22 | 0 | 22,034 | 1,483 | 195.21 / 602.88 | 49.85 / 51.7 |
| `fluxem_tools_env_v2` | `goose` | 25 | 50 | 0 | 68,634 | 1,075 | 492.1 / 689.47 | 20.5 / 28.51 |

## Notes & Auditability
- `fa1zvn/redline-v2` was pulled and tested but excluded from paired scoring due to an upstream `verifiers 0.1.11` `MultiTurnEnv.has_error(state=None)` `AttributeError` in the Hub package.
- Full scrubbed per-example rollouts and proxy telemetry are archived under `traces/*.jsonl.gz` (`<= 100 MB` in-repo).
