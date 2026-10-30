"""Run the frozen BGI-01F pilot on genuine Qwen2.5-1.5B CPU inference.

Model words are treated as untrusted proposals. There is no execution of
model-authored code and no interaction with real hardware or user data.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

from core import canonical, digest, exhaustive_certificate
from fresh_suite import audit_csv, fingerprint, fixture
from pilot import (GROUP_MIN_CORRECT, MIN_PARAPHRASE_AGREE,
                   eval_repair, eval_semantic)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "775b11afaf83e0dc75bd5abaf90133e47b3ec082"
SEED = 20261030
HERE = Path(__file__).resolve().parent


class RealModel:
    def __init__(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.versions = {"python": platform.python_version(),
                         "torch": torch.__version__,
                         "transformers": transformers.__version__}
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION,
            torch_dtype=torch.float32, trust_remote_code=False).cpu().eval()
        self.cache = {}
        self.forward_count = 0
        self.cache_hits = 0

    def __call__(self, messages: list[dict], kind: str) -> tuple[str, dict]:
        if kind not in ("action", "patch"):
            raise ValueError("unknown model task")
        cachekey = digest({"messages": messages, "kind": kind})
        if cachekey in self.cache:
            self.cache_hits += 1
            raw, meta = self.cache[cachekey]
            return raw, dict(meta, reused_identical_prompt=True)
        token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt")
        if not isinstance(token_ids, self.torch.Tensor):
            raise RuntimeError("model tokenizer returned unexpected type")
        if token_ids.shape[-1] > 1024:
            raise RuntimeError("frozen input token ceiling exceeded")
        limit = 12 if kind == "action" else 72
        with self.torch.inference_mode():
            full = self.model.generate(
                token_ids, max_new_tokens=limit, do_sample=False,
                num_beams=1, pad_token_id=self.tokenizer.eos_token_id)
        next_ids = full[0, token_ids.shape[-1]:]
        output = self.tokenizer.decode(next_ids, skip_special_tokens=True).strip()
        meta = {
            "kind": kind, "inference": "genuine_deterministic_cpu_greedy",
            "input_tokens": int(token_ids.shape[-1]),
            "output_tokens": int(next_ids.shape[0]),
            "max_new_tokens": limit, "request_sha256": cachekey,
            "reused_identical_prompt": False,
        }
        self.forward_count += 1
        self.cache[cachekey] = (output, meta)
        return output, dict(meta)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=HERE / "results" / "bgi01f_real_qwen15b_v1.json")
    args = p.parse_args()

    original = fixture()
    suite_sha = fingerprint()
    print("BGI01F_FIXTURE_SHA256=" + suite_sha, flush=True)
    certificate = exhaustive_certificate()
    model = RealModel()
    semantic = eval_semantic(model, original)
    print("BGI01F_SEMANTIC_GATE=" + semantic["gate"], flush=True)
    print("BGI01F_CLASS_ACCURACY=" + json.dumps(semantic["groups"], sort_keys=True),
          flush=True)
    print("BGI01F_PARAPHRASE_AGREEMENT=" +
          str(semantic["paraphrase_agreement"]) + "/48", flush=True)

    repairs = eval_repair(model)
    expected_queries = 96 + 7 * 3 * 2
    if model.forward_count + model.cache_hits != expected_queries:
        raise RuntimeError("incomplete model-query transcript")
    if len(semantic["records"]) != 96 or len(repairs["scenarios"]) != 7:
        raise RuntimeError("missing evaluated case")
    receipt = {
        "schema": "BGI01F-engineering-pilot-v1",
        "status": "unreviewed synthetic; human review and independent eval pending",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                  "weights_updated": False, "runtime": model.versions},
        "environment": {"seed": SEED, "github_sha": os.getenv("GITHUB_SHA"),
                        "independent_human_review": False},
        "model_callbacks": expected_queries,
        "real_model_forward_calls": model.forward_count,
        "identical_prompt_cache_hits": model.cache_hits,
        "source_fixture_sha256": suite_sha,
        "policy_certificate": certificate,
        "semantic_thresholds": {
            "per_group_min_of_8": GROUP_MIN_CORRECT,
            "paraphrase_min_of_48": MIN_PARAPHRASE_AGREE},
        "semantic": semantic, "repair": repairs,
        "evidence_limits": [
            "Synthetic scenarios were generated by the investigator, without two human reviews",
            "Single model, fixed seed; 48 evaluations are grouped by repeated fact templates",
            "No neural weight updates or persistent moral self-model",
            "Typed policy edits are sandbox configuration proposals",
            "Independent verifier and checkpoint are assumed trusted software",
            "Model-repair performance does not establish moral drives or attractors",
        ],
    }
    digest_payload = digest(receipt)
    outfile = args.output
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps({"sha256": digest_payload, "payload": receipt},
                                  sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (outfile.parent / "review_template.csv").write_text(audit_csv(), encoding="utf-8")
    print("BGI01F_PAYLOAD_SHA256=" + digest_payload, flush=True)
    print("BGI01F_QUERY_COUNTS=" + json.dumps({
        "real_forwards": model.forward_count, "cache_hits": model.cache_hits,
        "total_queries": expected_queries}, sort_keys=True), flush=True)
    print("BGI01F_REPAIR_COUNTS=" + json.dumps(repairs["summary"],
          sort_keys=True), flush=True)
    for sc in repairs["scenarios"]:
        reports = []
        for mode in ("unchecked_model", "verified_model"):
            o = next(x for x in sc["outcomes"] if x["mode"] == mode)
            reports.append({"mode": mode, "complete": o["complete"],
                            "remaining": o["remaining_missing"],
                            "blocked_weakening": o["blocked_weakening"],
                            "patches": [a["event"]["parsed"] for a in o["attempts"]]})
        print("BGI01F_REPAIR_SCENARIO=" +
              json.dumps({"scenario": sc["scenario"], "results": reports},
                         sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
