# Bonsai 2 27B with Bonsai 2.5's MTP head

PrismML published an MTP (nextn) draft head only for Ternary Bonsai 2.5 27B (`-PTQ1_0-mtp.gguf`); the Bonsai 2 27B
GGUF repo has none. Both models share the qwen35 layout (851 tensors, same names and shapes; 96 small
`ssm_alpha/ssm_beta` tensors are BF16 in 2 and F32 in 2.5), and the 2.5 head is 15 `blk.64.*` tensors (228 MiB)
plus `qwen35.block_count 65` and `qwen35.nextn_predict_layers 1`.

- `mtp_graft.py` writes Bonsai 2's KV and tensors byte for byte plus the 2.5 head (`ggufhdr.py` reads GGUF
  headers; upstream gguf-py rejects the PTQ1_0 type id 143).
- `run_bench.sh` runs the first 13 MIT-13 pilot prompts (first one a warm-up) through llama-prism-b12178 with the
  pilot's server flags, once on the plain file and once on the graft with
  `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1`, via `bench_client.py` (same request as
  `mit13_dispatch.py`). `summarize.py` reports decode tok/s, draft acceptance and replies identical to the plain arm.

The head was trained against 2.5's hidden states, so acceptance on Bonsai 2 measures how far 2.5 drifted from 2.
