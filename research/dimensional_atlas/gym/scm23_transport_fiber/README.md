# SCM-23: supervised causal transport within a fixed head fiber

**Status:** registered synthetic CPU assay completed; primary gate failed in one of three model seeds. Original SCM-20 trained checkpoints were unavailable here.

This follows merged SCM-22 PR #206, with the same tied four-visit F11 arithmetic model, 48-hidden CLS state, 37-dimensional linear readout-head fiber, three seeds 191/1191/2191, and shared TopK SAE. The frozen preregistration is [PROTOCOL.md](PROTOCOL.md), committed at af4058e before held-out evaluation.

The independent *compact* replay is [compact_replay.py](compact_replay.py), using the SCM-22 source already on main. Run on CPU:

\`\`\`bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python research/dimensional_atlas/gym/scm23_transport_fiber/compact_replay.py
\`\`\`

The full research archive also contains a granular 768-pair outcome receipt, a read-only independent auditor with 25 passing tests, and a separately gated SCM-20 checkpoint-host script. The checkpoint script has not been run against private SCM-20 weights.

A gradient-transport frame is fitted using only calibration recipient/donor target margins, then frozen. At held-out time every frame is driven by the same donor-recipient hidden-state difference; no test-target gradients are supplied to the patch. All rank-four interventions use identical per-pair L2 norms. The final LayerNorm means the fixed linear-head kernel is not a global nonlinear null.

See [RESULTS.md](RESULTS.md) for exact outcomes. The finite F11 algebra combinations recur across training and held-out generators, and samples within one model seed are dependent.