"""Synthetic plumbing test and optional pinned Hugging Face causal-model backend."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Sample:
    text: str
    trace: list[list[float]] | None
    trace_source: str
    generation_tokens: int


class ScriptedBackend:
    """Deterministic plumbing-only baseline, not model inference."""
    name = "scripted"

    def generate(self, messages: list[dict], *, seed: int, condition: str, turn: int) -> Sample:
        rng = random.Random(seed)
        # Same arbitrary pseudoactivations in every condition for paired seeds.
        trace = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(12)]
        return Sample(f"Synthetic response for {condition} at turn {turn}.", trace,
                      "synthetic", 12)


class HuggingFaceBackend:
    """One predeclared decoder layer, teacher-forced post-generation activation read.

    The trace is collected with a second forward pass after generation. Reported
    activations are per GENERATED token, and are not online recurrent states.
    """
    name = "hf"

    def __init__(self, model_id: str, revision: str, *, trace_layer: int = -1,
                 max_new_tokens: int = 32, load_in_4bit: bool = False):
        import re
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("pin an immutable 40-hex-character model commit revision")
        if not 4 <= max_new_tokens <= 64:
            raise ValueError("max_new_tokens must be in [4, 64]")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install requirements-model.txt for actual model inference") from exc
        import transformers
        self.library_versions = {"torch": torch.__version__,
                                 "transformers": transformers.__version__}
        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.trace_layer = trace_layer
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                       trust_remote_code=False)
        kwargs = {"revision": revision, "trust_remote_code": False,
                  "device_map": "auto", "torch_dtype": (torch.bfloat16 if torch.cuda.is_available()
                                                             else torch.float32)}
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise ValueError("4-bit quantization requires CUDA")
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("decoder layers unavailable: use a supported causal model")
        self.layers = layers
        self.actual_layer = trace_layer % len(layers)

    def generate(self, messages: list[dict], *, seed: int, condition: str, turn: int) -> Sample:
        torch = self.torch
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", enable_thinking=False)
        if tokens.shape[1] > 1024:
            raise ValueError("input exceeds SELF-01's 1024-token prompt cap")
        tokens = tokens.to(self.model.device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            sequence = self.model.generate(tokens, do_sample=True, temperature=0.7,
                                           top_p=0.95, max_new_tokens=self.max_new_tokens,
                                           pad_token_id=self.tokenizer.eos_token_id)
        prefix_len = tokens.shape[1]
        completion = sequence[0, prefix_len:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        captured = []

        def hook(_module, _args, output):
            values = output[0] if isinstance(output, tuple) else output
            captured.append(values[:, prefix_len:, :].detach().to("cpu", dtype=torch.float32))

        handle = self.layers[self.actual_layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                self.model(input_ids=sequence, use_cache=False)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("expected one complete teacher-forced layer trace")
        trace = captured[0][0].tolist()
        return Sample(text, trace, "model_posthoc", int(completion.shape[0]))
