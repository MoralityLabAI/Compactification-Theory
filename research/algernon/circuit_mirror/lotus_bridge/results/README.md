# LOTUS results

Reports from the Algernon LOTUS bridge, newest first. Every run was registered in
[EXPERIMENT.md](../../EXPERIMENT.md) before it ran. Large captures and checkpoints stay outside git, and their
SHA-256 hashes are recorded in each run's receipts.

| run | what | status |
| --- | --- | --- |
| [algernon-03](algernon-03/report.md) | Phase 4: 02D's D1 frozen, its slots wiped by noise after loop j (sigma\* = 16 x slot RMS). Correctors added after every loop: C (exact transition residual), I (the same information unsubtracted), N (decode only), plus plain recurrence P. The exact R1CS checker scores the whole decoded trace. | done 2026-11-07: R1-R3 not met. Primary accuracy P 0.494, C 0.487, I 0.505, N 0.500, and C - P = -0.007 [-0.028, +0.016]. Instrument valid. The registered correctors could not reach the noise (about 100x too small), so the constraint was not tested; 03b rescales them. |
| [algernon-02e](algernon-02e/report.md) | No training: 02D's D1 at d = 16..64 loops, and causal patching of its six slots (a donor's state after loop k replaces the recipient's). | done 2026-11-06: correct on all 432 prompts to 64 loops; patched answers follow the donor's state under the recipient's map in 141/144 pairs (0.979 [0.941, 0.993]), self-patches 144/144; registered decision `slots_carry_state` positive |
| [algernon-02d](algernon-02d/report.md) | No step count in the question; 6 fixed latent slots refined once per loop, k loops for depth k. Arm D1 (per-loop supervision of the slots) vs D0 (answer only). Scored with a 1-16 depth curve (loops = d) and the 02B instrument. | done 2026-11-06: D1 (per-loop supervision) answers every roster case at depths 1-16 (trained to 8) and passes the gate, `recurrence_specific_state`; reusable transition shown for D1. D0 `training_failed` (0.639), so per-loop supervision vs none is reported, not decided |
| [algernon-02c](algernon-02c/report.md) | Upstream's GPT-2 LOTUS recipe trained on the pointer task: stage-1 CoT, then arm A (published after-loop supervision) vs arm B (step k supervised at loop k). Scored with a 1-16 depth curve and the 02B instrument. | done 2026-11-05: latent states decodable within the trained depth (0.84-0.89) but not beyond; loop-to-step tying not shown (B - A = -0.099); both arms `no_iterative_state_signal` |
| [algernon-02b](algernon-02b/report.md) | Published `gsm-lotus-llama3b` checkpoint on the frozen 6x16 pointer roster: block-aligned capture, controls, Phase 4 gate. | done 2026-11-04: gate failed, `no_iterative_state_signal` |
