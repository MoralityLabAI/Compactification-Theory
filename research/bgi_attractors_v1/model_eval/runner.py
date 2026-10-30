"""BGI-01B: real 135M-model forced-choice action pilot.

Requires torch, transformers and safetensors. This does not fine-tune weights:
state changes are explicit externally scripted policy-memory mutations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from world import SEED, load_cases, run_eval, summary

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
HERE = Path(__file__).resolve().parent


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class HFProposer:
    def __init__(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.transformers_version = transformers.__version__
        torch.set_num_threads(2)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REV, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REV, torch_dtype=torch.float32,
            trust_remote_code=False).cpu().eval()
        self.ids = {}
        for letter in "AB":
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"letter {letter} was not single-token: {ids}")
            self.ids[letter] = ids[0]
        if self.ids["A"] == self.ids["B"]:
            raise RuntimeError("choice tokens collided")
        self.calls = 0

    def __call__(self, messages: list[dict], request: dict) -> tuple[str, dict]:
        torch = self.torch
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        )
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("unexpected chat-template output")
        if input_ids.shape[1] > 512:
            raise RuntimeError("pilot case exceeded the frozen context budget")
        with torch.inference_mode():
            logits = self.model(input_ids).logits[0, -1].float()
            logit_a = float(logits[self.ids["A"]].item())
            logit_b = float(logits[self.ids["B"]].item())
        letter = "A" if logit_a >= logit_b else "B"
        self.calls += 1
        # A/B label-conditional discriminative scoring, rather than sampled
        # free-form tool calls. This is genuine forward inference of the model.
        return letter, {
            "inference_method": "next_token_binary_logits_v1",
            "input_tokens": int(input_ids.shape[1]),
            "logit_A": round(logit_a, 6),
            "logit_B": round(logit_b, 6),
            "margin_A_minus_B": round(logit_a - logit_b, 6),
            "chosen_token_id": self.ids[letter],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=HERE / "results" / "smollm2_135m_pilot_v1.json")
    args = ap.parse_args()

    fixture_meta, cases = load_cases()
    print("BGI-01B fixture SHA-256:", fixture_meta["sha256_fixture_bytes"], flush=True)
    agent = HFProposer()
    rows = run_eval(agent, cases, SEED)
    if agent.calls != 144 or len(rows) != 144:
        raise RuntimeError("incomplete paired pilot; no partial result can pass")
    payload = {
        "schema": "bgi01b-real-inference-pilot-v1",
        "kind": "real_model_forward_cpu; frozen synthetic tasks and synthetic control state",
        "model_id": MODEL_ID, "model_revision": MODEL_REV,
        "model_weights_updated": False,
        "policy_memory_updated_by": "scripted state fault and scripted repair; no learned moral adaptation",
        "choice_method": "binary next-token logits; no free-form generation",
        "control_limits": "restricted-class gate; misses public/revoked hazards; overblocks restricted/authorized",
        "fixture_metadata": fixture_meta,
        "versions": {"python": platform.python_version(),
                     "torch": agent.torch.__version__,
                     "transformers": agent.transformers_version},
        "arms": ["A_fixed", "B_mutable", "C_ungated", "D_hybrid"],
        "phases": ["baseline", "memory_attack", "recovery"],
        "seed": SEED,
        "n_agent_forward_calls": agent.calls,
        "scenarios": rows,
        "aggregate": summary(rows),
        "limitations": [
            "Single nonrandom generation setting and one fixed public model; no generalization claim",
            "Convenience fixtures authored by investigator; no independent human label audit",
            "All policies, detection, corruption and repair mechanisms are externally scripted",
            "A deterministic binary-action probe, not real tools or free-form agent behavior",
            "No gradient learning, self-editing weights, peer training or evidence of moral agency",
            "Synthetic world harm means an unauthorized packet release only",
        ],
    }
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    receipt = {"payload_sha256": digest, "payload": payload}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("BGI01B_RECEIPT_SHA256=" + digest, flush=True)
    print("BGI01B_SUMMARY_JSON=" + json.dumps(payload["aggregate"], sort_keys=True), flush=True)
    print("BGI01B_INFERENCE_CALLS=" + str(agent.calls), flush=True)
    print("BGI01B_RESULT_FILE=" + str(args.output), flush=True)


if __name__ == "__main__":
    main()
