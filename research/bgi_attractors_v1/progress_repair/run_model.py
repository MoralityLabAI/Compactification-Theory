"""BGI-01H pinned real Qwen progress selection and fallback comparison."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "verified_repair"))
sys.path.insert(0, str(HERE.parent / "factored_admission"))

from core import digest, exhaustive_certificate
from controller import evaluate, exhaustive_progress_certificate

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "775b11afaf83e0dc75bd5abaf90133e47b3ec082"
SEED = 20261030


class QwenSelector:
    def __init__(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(2)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.versions = {"python":platform.python_version(),
                         "torch":torch.__version__,
                         "transformers":transformers.__version__}
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION,
            torch_dtype=torch.float32,
            trust_remote_code=False).cpu().eval()
        self.cache = {}
        self.forwards = 0
        self.cache_hits = 0

    def __call__(self,messages:list[dict],kind:str):
        if kind!="selection":raise ValueError("unknown operation")
        key=digest({"messages":messages,"kind":kind})
        if key in self.cache:
            self.cache_hits+=1
            output,meta=self.cache[key]
            return output,{**meta,"cache_hit":True}
        ids=self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt")
        if not isinstance(ids,self.torch.Tensor) or ids.shape[1]>1024:
            raise RuntimeError("invalid or oversized prompt")
        with self.torch.inference_mode():
            completion=self.model.generate(
                ids,max_new_tokens=80,do_sample=False,num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id)
        tokens=completion[0,ids.shape[1]:]
        output=self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
        meta={"type":"real_cpu_autoregressive_greedy",
              "input_tokens":int(ids.shape[1]),
              "output_tokens":int(tokens.shape[0]),
              "max_new_tokens":80,
              "request_sha256":key,
              "cache_hit":False}
        self.cache[key]=(output,meta)
        self.forwards+=1
        return output,meta


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,
                        default=HERE / "results"/"bgi01h_qwen15b_progress_v1.json")
    args=parser.parse_args()
    oracle=exhaustive_certificate()
    progress=exhaustive_progress_certificate()
    print("BGI01H_FINITE_PROGRESS="+json.dumps(progress,sort_keys=True),flush=True)
    model=QwenSelector()
    result=evaluate(model)
    expected=60
    if result["callback_count"]!=expected or model.forwards+model.cache_hits!=expected:
        raise RuntimeError("missing planned inference callback")
    payload={
        "schema":"BGI01H-progress-selector-v1",
        "status":"exploratory post-BGI01G, independent human audit pending",
        "model":{"id":MODEL_ID,"revision":MODEL_REVISION,
                 "weights_modified":False, "versions":model.versions},
        "github_sha":os.getenv("GITHUB_SHA"),
        "real_model_forwards":model.forwards,
        "identical_prompt_cache_hits":model.cache_hits,
        "total_model_callbacks":expected,
        "seed":SEED,
        "core_policy_exhaustive_certificate":oracle,
        "progress_exhaustive_certificate":progress,
        "result":result,
        "limits":[
            "Same repair fault patterns as BGI01G, inspected before experimental design",
            "Only eight cases are inside the prespecified three-round recovery scope",
            "Two four-/five-flag cases intentionally exceed that scope",
            "A model-chosen guard is distinct from externally substituted fallback repair",
            "One Qwen1.5B model, deterministic run, no independently human reviewed cases",
            "Trusted external witness and reference policy are engineering assumptions",
            "No neural weight updates, persistent policy learning, or moral self-model",
        ],
    }
    receipt={"sha256":digest(payload),"payload":payload}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("BGI01H_RECEIPT_SHA256="+receipt["sha256"],flush=True)
    print("BGI01H_INFERENCE="+json.dumps({
        "real_forwards":model.forwards,"cache_hits":model.cache_hits,
        "expected":expected},sort_keys=True),flush=True)
    print("BGI01H_SUMMARY="+json.dumps(result["summary"],sort_keys=True),flush=True)
    for case in result["scenarios"]:
        print("BGI01H_SCENARIO="+json.dumps({
            "id":case["id"],
            "model_only_remaining":case["model_only"]["potential_final"],
            "fallback_remaining":case["model_plus_fallback"]["potential_final"],
            "fallback_substitutions":case["model_plus_fallback"]["external_substitutions"],
            "fallback_model_progress":case["model_plus_fallback"]["model_progress_steps"],
        },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
