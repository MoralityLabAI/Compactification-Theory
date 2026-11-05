# SIML-01N neural curriculum pilot (2026-11-02)

**Exploratory CPU training, five new initialization seeds:** 20–24. Six curricula; each arm receives 320 AdamW optimizer updates, batch size 32, width-48 tied GRUCell and two recurrent steps per training example. Diagnostic measurements are acquired at identical intervals. Physical runtime and task-specific auxiliary training work vary with the chosen task.

| Curriculum | Mean held-out weighted accuracy | Unseen three-action composition accuracy |
|---|---:|---:|
| Random task selection | **0.5446** | 0.9167 |
| Guarded introspection | 0.5325 | 0.9417 |
| Guarded introspection, shuffled diagnostics | 0.5381 | **0.9500** |
| Fixed 5:9:6 curriculum | 0.5048 | 0.8750 |
| Greedy learning progress | 0.4425 | 0.5000 |
| Greedy introspection | 0.3519 | 0.4917 |

Earlier pilot seeds informed the architecture and guarded policy. The T1 final words are held out from the optimizer but share the eight microscopic states; T4 separates 678 train, 139 development and 144 test state-pairs. T0 reuses its eight-state domain. The three-action suite is an additional unseen horizon with familiar primitives.

**Observed failure:** Unconstrained greedy curricula concentrate on the difficult modular-shear task and lose ground on established competencies. Guarded task-family sampling prevents this starvation. The shuffled internal-diagnostic control slightly beats guarded introspection, so the current result does not demonstrate added value from hidden-state closure metrics. Random selection wins the weighted primary endpoint. All findings are exploratory.

The local source artifact is a PyTorch 2.10 CPU reference with a neural learner, task graph, six curriculum selectors, strict opt-in local 27B proposal-only interface, five-seed outcome receipt, offline source-hash/retrain verifier and tests. **49 unit tests pass** in normal and optimized Python, and 30 trained checkpoint fingerprints were replayed on the original CPU runtime. No 27B model was loaded, and no paid compute was used.

The companion `SIML-01_CPU_Reference_v02.zip` remains the earlier *tabular* stage. This new neural stage is a separate experiment; the GitHub training-source import is tracked as follow-up rather than implied by this results-only commit.
