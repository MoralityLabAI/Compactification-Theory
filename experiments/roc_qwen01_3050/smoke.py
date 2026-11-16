"""Real-checkpoint NF4 smoke; imports audit before the model library."""
from preflight import audit, ROOT, RESULTS, CEILING, save, sha
torch, hardware = audit()
import gc
import importlib.util
import json
from pathlib import Path
import time
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

MODEL = "Qwen/Qwen3.5-0.8B-Base"


def download():
    manifest_path = ROOT / "model_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = old.get("revision") or HfApi().model_info(MODEL).sha
    # Never select an alternate checkpoint. Disk cache contains source safetensors;
    # only the text module is instantiated by Qwen3_5ForCausalLM.
    from download import checkpoint
    path = checkpoint(MODEL, revision)
    files = {str(p.relative_to(path)): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(path.rglob("*")) if p.is_file()}
    manifest = dict(checkpoint=MODEL, revision=revision, files=files,
                    backend="transformers/bitsandbytes NF4 double-quant FP16 compute",
                    model_class="Qwen3_5ForCausalLM", vision_instantiated=False,
                    snapshot_path=str(path))
    save(manifest_path, manifest)
    return path, manifest


def load():
    path, manifest = download()
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config = config.get_text_config()
    config._attn_implementation = "eager"
    config.use_cache = False
    start = time.perf_counter()
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        path, config=config, local_files_only=True, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="eager", output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16))
    model.eval().requires_grad_(False)
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
        raise RuntimeError(f"Incomplete checkpoint load: {loading_info}")
    if any("visual" in n or "vision" in n for n, _ in model.named_modules()):
        raise RuntimeError("Unexpected vision module loaded")
    if not model.is_loaded_in_4bit:
        raise RuntimeError("NF4 loading was not applied")
    if torch.cuda.max_memory_allocated() >= CEILING:
        raise RuntimeError("Loading exceeded memory budget")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer, manifest, time.perf_counter() - start


def main():
    start = time.perf_counter()
    model, tokenizer, manifest, load_seconds = load()
    prompt = "Rule: copy. Remember A. Wait 1. Wait 2. Wait 3. Release. Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    saved = {}
    def hook(_module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        saved["hidden"] = h[0, -1].float().cpu().numpy().copy()
    handle = model.model.layers[11].register_forward_hook(hook)
    with torch.inference_mode():
        out = model(**inputs, use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].float().cpu()
    handle.remove()
    candidates = {x: tokenizer.encode(x, add_special_tokens=False) for x in (" A", " B", "A", "B")}
    if not all(len(candidates[x]) == 1 for x in (" A", " B")):
        raise RuntimeError(f"Not single-token answers: {candidates}")
    ids = [candidates[x][0] for x in (" A", " B")]
    baseline = (logits[ids[0]] - logits[ids[1]]).item()
    del out, logits
    gc.collect()
    delta = torch.zeros(1024, dtype=torch.float16, device="cuda")
    delta[0] = 0.5
    actual = {}
    def intervene(_module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        changed = h.clone()
        changed[0, -1] += delta
        actual["l2"] = (changed[0, -1].float() - h[0, -1].float()).norm().item()
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = model.model.layers[11].register_forward_hook(intervene)
    with torch.inference_mode():
        out = model(**inputs, use_cache=False, logits_to_keep=1)
        changed_logits = out.logits[0, -1].float().cpu()
    handle.remove()
    changed_margin = (changed_logits[ids[0]] - changed_logits[ids[1]]).item()
    result = dict(status="passed", checkpoint=manifest["checkpoint"], revision=manifest["revision"],
                  tokens=inputs["input_ids"].shape[1], candidate_token_ids=candidates,
                  residual_layer_one_based=12, selected_token_position=-1,
                  hidden_shape=list(saved["hidden"].shape), baseline_margin=baseline,
                  intervened_margin=changed_margin, actual_displacement_l2=actual["l2"],
                  requested_radius=0.5, successful_forward_passes=2,
                  effect_type="conditional_residual_state_effect", use_cache=False,
                  kernel_availability={p: importlib.util.find_spec(p) is not None
                                       for p in ("fla", "causal_conv1d", "triton")},
                  allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  device_name=torch.cuda.get_device_name(0), load_seconds=load_seconds,
                  duration_seconds=time.perf_counter()-start)
    if not torch.isfinite(changed_logits).all():
        raise RuntimeError("Nonfinite output")
    save(RESULTS / "smoke.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        save(RESULTS / "smoke_failure.json", dict(error=repr(e), hardware=hardware))
        raise
