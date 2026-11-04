# LOTUS results

Reports from the Algernon LOTUS bridge, newest first. Every run was registered in
[EXPERIMENT.md](../../EXPERIMENT.md) before it ran. Large captures and checkpoints stay outside git, and their
SHA-256 hashes are recorded in each run's receipts.

| run | what | status |
| --- | --- | --- |
| [algernon-02c](algernon-02c/progress.md) | Upstream's GPT-2 LOTUS recipe trained on the pointer task: stage-1 CoT, then arm A (published after-loop supervision) vs arm B (step k supervised at loop k). Scored with a 1-16 depth curve and the 02B instrument. | running; progress.md is re-rendered and pushed as each step finishes |
| [algernon-02b](algernon-02b/report.md) | Published `gsm-lotus-llama3b` checkpoint on the frozen 6x16 pointer roster: block-aligned capture, controls, Phase 4 gate. | done 2026-11-04: gate failed, `no_iterative_state_signal` |
