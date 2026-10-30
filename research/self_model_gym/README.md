# SelfModelGym: SELF-01 v0.1

An isolated, reproducible research gym for recursive self-reference, output repetition controls, and hidden-state recurrence. It studies model behavior and computation; it does not diagnose consciousness.

**Status:** the offline synthetic smoke runner works and has local tests. The optional Hugging Face backend is implemented but requires model-specific on-hardware validation. No Qwen inference results are included. Synthetic activations are generated from the same pseudo-random stream across matched arms, solely to test the pipeline.

## Offline reproduction

Run these commands from the Compactification-Theory repository root with Python 3.10+:

~~~sh
python -m unittest discover -s research/self_model_gym/tests -v
python -m research.self_model_gym.run --backend scripted --replicates 2 --seed 20261030 --save-traces --output runs/self-01-smoke
python -m research.self_model_gym.verify runs/self-01-smoke
~~~

The core environment uses Python's standard library. Output is excluded from Git via the existing runs/ ignore rule. Each run writes manifest.json, episodes.jsonl, and receipt.json. The manifest identifies the backend, protocol digest and selected model metadata. Each full episode has four turns, prompts, responses, behavioral scores and hidden-state scores. The receipt hashes both evidence files and source modules, checks expected episode counts and includes the paired primary contrast. Save traces to enable independent recalculation of activation metrics. Hashes are local checks, not signed remote attestations.

## Pinned real-model pilot

Install optional dependencies and provide an actual immutable model revision:

~~~sh
pip install -r research/self_model_gym/requirements-model.txt
# Install a CUDA-compatible bitsandbytes package for 4-bit loading.
python -m research.self_model_gym.run --backend hf \
  --model-id Qwen/Qwen3.5-4B --revision "$PINNED_40_HEX_HF_COMMIT" \
  --load-in-4bit --trace-layer -1 --max-new-tokens 32 \
  --replicates 2 --seed 20261030 --save-traces \
  --confirm-real-inference --output runs/self-01-qwen-pilot
python -m research.self_model_gym.verify runs/self-01-qwen-pilot
~~~

Weights are downloaded only when the real backend is explicitly selected and confirmed with a pinned revision. The adapter uses a text-only non-thinking chat template, temperature 0.7 and top-p 0.95. It makes an additional teacher-forced forward pass to capture generated-token representations at a specified decoder layer. This measures token-position activations, not verified cognitive attractor states. Model architecture compatibility, VRAM, and timing remain to be checked on the RTX 5080. Start with two paired replicates; the proposed full experiment is 100 replicates across five conditions (2,000 generated turns and associated trace passes).

## Core API

~~~python
from research.self_model_gym import SelfModelEnv

env = SelfModelEnv()
obs, info = env.reset(seed=7, options={"condition": "self_reference"})
while True:
    obs, reward, done, truncated, info = env.step("example response")
    if done or truncated:
        break
~~~

For activation measurements, pass a mapping with text, a rectangular numeric trace and trace_source="model_posthoc". Core validation is fail-before-mutation. Rewards are always zero to avoid optimizing the evaluation measure. The optional GymnasiumSelfModelEnv wrapper uses a Text action space and scores behavioral output. Install requirements-gym.txt separately.

## Conditions and next experiments

The five SELF-01 arms are neutral reflection, recursive self-reference, lexical repetition, meditation descriptions and a fictional machine monologue. In each arm the four-turn prompts are frozen in protocol.py; seed-matched arms are executed in shuffled order. The registered endpoint is the mean fraction of nonadjacent token-state pairs with mean-centered cosine similarity at least 0.90, compared to the lexical repetition arm. Behavioral word entropy and adjacent-word repetition are diagnostics.

See PROTOCOL.md for the exact operational hypothesis and falsification rules. SELF-02 (concealment and candor), SELF-03 (functional valence) and SELF-04 (compactified self-model) are planned separately.
