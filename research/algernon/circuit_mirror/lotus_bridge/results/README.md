# LOTUS results

Reports from the Algernon LOTUS bridge, newest first. Every run was registered in
[EXPERIMENT.md](../../EXPERIMENT.md) before it ran. Large captures and checkpoints stay outside git, and their
SHA-256 hashes are recorded in each run's receipts.

| run | what | status |
| --- | --- | --- |
| [algernon-02d](algernon-02d/progress.md) | No step count in the question; 6 fixed latent slots refined once per loop, k loops for depth k. Arm D1 (per-loop supervision of the slots) vs D0 (answer only). Scored with a 1-16 depth curve (loops = d) and the 02B instrument. | running; progress.md is re-rendered and pushed as each step finishes |
| [algernon-02c](algernon-02c/report.md) | Upstream's GPT-2 LOTUS recipe trained on the pointer task: stage-1 CoT, then arm A (published after-loop supervision) vs arm B (step k supervised at loop k). Scored with a 1-16 depth curve and the 02B instrument. | done 2026-11-05: latent states decodable within the trained depth (0.84-0.89) but not beyond; loop-to-step tying not shown (B - A = -0.099); both arms `no_iterative_state_signal` |
| [algernon-02b](algernon-02b/report.md) | Published `gsm-lotus-llama3b` checkpoint on the frozen 6x16 pointer roster: block-aligned capture, controls, Phase 4 gate. | done 2026-11-04: gate failed, `no_iterative_state_signal` |
