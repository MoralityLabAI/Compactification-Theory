"""ARG-02: recurrent-depth evaluation of a pretrained Qwen checkpoint on the ARG-01 gym.

Torch-free modules: splits, prompts, scoring, stats, flops, probe, analyze,
symbolic. Torch modules: recurrence, runner. The EvalMath oracle is imported,
never copied.
"""
