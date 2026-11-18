# CGC-02: trained protected causal dimension across recurrent depth

[Source-locked exploratory protocol](PROTOCOL.md) · [CGC-01 mathematical theorems](../THEORY.md) · [BPD protected causal compactification issue #245](https://github.com/MoralityLabAI/Compactification-Theory/issues/245)

CGC-02 trains a 12-state shared nonlinear recurrent transition, computes *actual* task-output Jacobians through its future shared-weight rollouts, and tests whether small protected physical subspaces selected on independent development data carry causal source changes across unseen recurrent depth.

The actual current protected readout is the **complete three-class softmax** of the physical state coordinates (0,4,zero). Every candidate write is projected to the conservative exact affine fiber \`delta[0]=delta[4]=0\`; source/code asserts all normalized probabilities are unchanged. Future protected-output drift is measured independently at each rollout horizon.

Each teacher gets two independent full-rank BPD physical factor fits \`V@U ≈ C\`, with gauge-sheared full matrix parity and rank-two physical **output SVD** comparisons. The dedicated [native BPD wrapper](native_bpd_parity.py) checks all-factor nine-pass masked execution and one full donor \`swap_acts\` at an immutable BPD commit. Arbitrary component singleton masks are **not** claimed gauge invariant.

Two frozen task-conditioned subspaces are built from protected future task Jacobians on **256 development-only examples**: a rank-two depth-4 eigenspace, and a rank-two weighted depth-2/4/6 eigenspace. The BPD-weight-SVD rank-two and teacher-weight-SVD rank-two baselines are independent of task Jacobians. Paired source interventions and further recurrence are tested at depths 2,4,6 and **untrained depth 8** on an independently generated anti-correlated heldout source family. Report every generated and clean-eligible pair, task-oracle accuracy, source-donor readout, finite future protected drift, and one-step reduced-state closure defect.

Controls include full physical BPD donor write, true physical teacher write, exact projected donor hidden state, zero edit, an independent random protected basis, 32 matched-dose random protected directions, and 16 **privileged full-two-task-Jacobian-and-dose matched** randomized directions at depths 4 and 8. Tangent-matched nulls are selected using the *evaluation teacher's Jacobian*, explicitly as a high-information control. A first-order task-space rank by itself is **not** an autonomous model-state quotient or a learned sparse BPD mechanism.

Previous architecture tuning used excluded seeds **901 and 902** only. The fresh roots \`2601,2607,2617,2621\`, selector depths 2/4/6, rank two, depth-eight holdout, two-root factor refits, pair-source family and numerical gates were registered in [PROTOCOL.md](PROTOCOL.md) before training any fresh root. Four roots are exploratory and cannot establish 95/95 statistical significance.

## Source-only and excluded engineering

\`\`\`bash
python -m unittest discover -s research/causal_gauge_compactification/cgc02 -p 'test_*.py' -v
python -m research.causal_gauge_compactification.cgc02.experiment --phase excluded --out /tmp/cgc02_excluded.json
\`\`\`

Native BPD parity additionally needs \`MoralityLabAI/BPD\` at pinned commit \`f22a883144ee7d2caa2bd5d9936dd9e083c7f139\` on \`PYTHONPATH\` and PyTorch installed:

\`\`\`bash
python -m research.causal_gauge_compactification.cgc02.native_bpd_parity
\`\`\`

The fresh prospective protocol uses:

\`\`\`bash
python -m research.causal_gauge_compactification.cgc02.experiment --phase fresh --out /tmp/cgc02_fresh.json
\`\`\`

Only hosted checks executed **after** the source hash freeze may establish a fresh-root receipt. The implementation is a small trained scalar source gym, not a pretrained TRM/Nanbeige benchmark, actual language privacy, verified Hessian certificate, or new physical state compression algorithm.
