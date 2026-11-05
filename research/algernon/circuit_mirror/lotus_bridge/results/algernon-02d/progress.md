# Algernon-02D progress: loop count as the only depth signal, fixed latent slots

Rendered 2026-11-06 06:25 +0800 from the run directory by `report_02c.py`. Protocol: [EXPERIMENT.md, H-A2d](../../../EXPERIMENT.md#h-a2d-loop-count-as-the-only-depth-signal-one-fixed-set-of-latent-slots-registered-2026-11-06-before-any-training). Chain status: `running D1 2026-11-06T04:53:27`.

Data: 20000 train / 512 validation `random_function` cases, depth 1-8, seed 20261105; train `2dbe58737c7a`, validation `c97512fbb461`. Validation accuracy is the model's final answer with its trained blocks and loops.

### Arm D1: per-loop supervision of the fixed slots (weight 0.05)

Status `paused_power_cap`; process starts 3; power-cap pauses 3; out-of-memory pauses 0.

| epoch | stage | train loss | val n | val acc | depth 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | minutes (last process) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | loop | 0.3747 | 64 | 0.656 | 0.57 | 0.67 | 0.75 | 0.56 | 0.50 | 0.67 | 0.88 | 0.67 | 1.1 |
| 1 | loop | 0.3050 | 64 | 0.609 | 0.71 | 0.83 | 0.25 | 0.78 | 0.62 | 0.56 | 0.50 | 0.67 | 5.2 |

### Arm D0: answer only (weight 0)

Not started.

### Behavioural depth curve

Not run yet.

### 02B instrument on the trained arms

Not run yet.
