"""BGI-01G: pinned genuine small-model CPU evaluation.

Uses the factual extraction architecture plus a separate three-round policy
repair test. Model outputs remain untrusted data; JSON patches are constrained
by the same finite verifier as BGI-01F.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "verified_repair"))
from core import digest, exhaustive_certificate
from suite import audit_csv, fixture_hash, frozen
from eval import eval_cases, eval_repairs

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "775b11afaf83e0dc75bd5abaf90133e47b3ec082"
SEED = 20261030


class Inference:
    def __init__(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION,
            torch_dtype=torch.float32, trust_remote_code=False).cpu().eval()
        self.versions = {"python": platform.python_version(),
                         "torch": torch.__version__,
                         "transformers": transformers.__version__}
        self.cache = {}
        self.forwards = 0
        self.hits = 0

    def __call__(self, messages: list[dict], kind: str) -> tuple[str,dict]:
        if kind not in ("action", "facts", "patch"):
            raise ValueError("unknown model operation")
        key = digest({"messages": messages, "kind": kind})
        if key in self.cache:
            self.hits += 1
            raw, evidence = self.cache[key]
            return raw, {**evidence, "reused_identical_prompt": True}
        tok = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt")
        if not isinstance(tok, self.torch.Tensor) or tok.shape[-1] > 1024:
            raise RuntimeError("invalid/oversized model prompt")
        budget = {"action": 12, "facts": 112, "patch": 96}[kind]
        with self.torch.inference_mode():
            full = self.model.generate(
                tok, max_new_tokens=budget,
                do_sample=False, num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id)
        new = full[0, tok.shape[-1]:]
        raw = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        evidence = {"model": MODEL_ID, "input_tokens": int(tok.shape[-1]),
                    "output_tokens": int(new.shape[0]),
                    "token_budget": budget,
                    "inference": "genuine_cpu_greedy_autoregressive",
                    "request_sha256": key, "reused_identical_prompt": False}
        self.forwards += 1
        self.cache[key] = (raw, evidence)
        return raw, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["semantic", "repair", "both"],
                        default="both")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    source = frozen()
    sha = fixture_hash()
    print("BGI01G_FIXTURE_SHA256=" + sha, flush=True)
    finite = exhaustive_certificate()
    model = Inference()
    sem = eval_cases(model) if args.mode in ("semantic","both") else None
    if sem is not None:
        print("BGI01G_SEMANTIC_SUMMARY=" +
              json.dumps(sem["summary"], sort_keys=True), flush=True)
    repair = eval_repairs(model) if args.mode in ("repair","both") else None
    if repair is not None:
        print("BGI01G_REPAIR_SUMMARY=" +
              json.dumps(repair["summary"], sort_keys=True), flush=True)
        for row in repair["scenarios"]:
            print("BGI01G_REPAIR_CASE=" + json.dumps({
                "id":row["id"],
                "baseline_remaining":row["baseline"]["final_missing"],
                "feedback_remaining":row["feedback"]["final_missing"],
                "baseline_redundant":row["baseline"]["redundant_or_rejected"],
                "feedback_redundant":row["feedback"]["redundant_or_rejected"],
            }, sort_keys=True), flush=True)
    expected = (224 if sem is not None else 0) + (
        48 if repair is not None else 0)
    if expected != model.forwards + model.hits:
        raise RuntimeError("incomplete model callback count")
    if sem is not None and len(sem["rows"]) != 112:
        raise RuntimeError("incomplete semantic trial")
    if repair is not None and len(repair["scenarios"]) != 8:
        raise RuntimeError("incomplete repair trial")
    payload = {
        "schema": "BGI01G-v1", "status": "investigator-generated exploratory",
        "model": {"name":MODEL_ID,"revision":MODEL_REVISION,
                  "weights_modified":False, "versions":model.versions},
        "independent_human_review":False,
        "github_sha":os.getenv("GITHUB_SHA"),
        "mode":args.mode,"seed":SEED,
        "source_fixture_sha256":sha,
        "source_fixture_status":source["status"],
        "policy_certificate":finite,
        "total_callbacks":expected,
        "genuine_model_forwards":model.forwards,
        "identical_prompt_cache_hits":model.hits,
        "semantic":sem,"repair":repair,
        "limits":[
            "New synthetic cases were constructed after the previous failures were observed",
            "No independently reviewed labels or fresh blinded model selection",
            "One frozen model, one seed, deterministic decoding",
            "Fact-extraction pipeline invokes an independent deterministic policy",
            "Model-authored repairs update only a toy configuration state",
            "No persistence or learned self-model, no neural weight updates",
            "Comparison isolates prompt/control design, not intrinsic moral agency",
        ],
    }
    receipt = {"sha256":digest(payload),"payload":payload}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = args.output_dir / ("bgi01g_"+args.mode+"_qwen15b_v1.json")
    filename.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    (args.output_dir / "human_review_pending.csv").write_text(audit_csv())
    print("BGI01G_PAYLOAD_SHA256="+receipt["sha256"], flush=True)
    print("BGI01G_MODEL_CALLS="+json.dumps({
        "real":model.forwards,"cached":model.hits,"expected":expected,
    },sort_keys=True),flush=True)
    print("BGI01G_RESULT_FILE="+str(filename),flush=True)


if __name__=="__main__":
    main()
