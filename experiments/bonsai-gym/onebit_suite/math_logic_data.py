"""Build deterministic math/logic splits for the onebit-suite adapters.

Output: C:\\Users\\patri\\Documents\\Codex\\onebit-suite\\data\\{math,logic}\\*.jsonl + manifest.json
Usage: python math_logic_data.py [--check]   (--check rebuilds in memory and compares to disk hashes)
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import truststore
truststore.inject_into_ssl()

sys.path.insert(0, str(Path(__file__).parent))
import math_logic_score as S  # noqa: E402

OUT = Path(r"C:\Users\patri\Documents\Codex\onebit-suite\data")
TOK = r"C:\Users\patri\models\Qwen3.5-4B\tokenizer.json"
SEED = 20261125
MAX_TOK = 2000
INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
SKIP = {"arc_agi", "arc_agi_2", "buggy_tables"}


def tok():
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(TOK)
    return lambda s: len(t.encode(s).ids)


def row(id_, family, source, split, prompt, gold, target=None):
    r = {"id": id_, "family": family, "source": source, "split": split,
         "messages": [{"role": "user", "content": prompt}]}
    if target is not None:
        r["target"] = target
    r["gold"] = gold
    return r


def gsm_target(ans: str):
    body, final = ans.split("####")
    final = final.strip().replace(",", "")
    body = re.sub(r"<<.*?>>", "", body).strip()
    return f"{body}\nThe final answer is $\\boxed{{{final}}}$.", final


def build():
    from datasets import load_dataset
    ntok = tok()
    dropped = collections.Counter()
    files = {}

    # ---- math ----
    m = load_dataset("PrimeIntellect/INTELLECT-3-RL", "math", split="train")
    cand, seen = [], set()
    for i, r in enumerate(m):
        v = r["avg@8_qwen3_4b_instruct_2507"]
        if v is None or not (0.25 <= v <= 1.0):
            continue
        q = r["question"]
        if q in seen:
            dropped["i3math_dup"] += 1
            continue
        seen.add(q)
        cand.append((i, q, str(r["answer"])))
    random.Random(SEED).shuffle(cand)

    def take_math(n, split):
        out = []
        while cand and len(out) < n:
            i, q, a = cand.pop(0)
            p = INSTRUCTION + q
            if ntok(p) > MAX_TOK:
                dropped["i3math_long"] += 1
                continue
            if not S.math_score(a, "\\boxed{" + a + "}")["score"]:
                dropped["i3math_gold_unparseable"] += 1
                continue
            out.append(row(f"i3math-{i}", "math", "i3-math", split, p, {"answer": a}))
        return out

    math_eval = take_math(60, "eval")
    math_pool = take_math(400, "rft_pool")

    g = load_dataset("openai/gsm8k", "main")
    gt = list(enumerate(g["test"]))
    random.Random(SEED + 1).shuffle(gt)
    gsm_eval = []
    for i, r in gt:
        if len(gsm_eval) == 60:
            break
        p = INSTRUCTION + r["question"]
        _, fin = gsm_target(r["answer"])
        if ntok(p) > MAX_TOK or not S.math_score(fin, "\\boxed{" + fin + "}")["score"]:
            dropped["gsm_test_skip"] += 1
            continue
        gsm_eval.append(row(f"gsm8k-test-{i}", "math", "gsm8k", "eval", p, {"answer": fin}))

    gtr = list(enumerate(g["train"]))
    random.Random(SEED + 2).shuffle(gtr)
    gold_train, val = [], []
    for i, r in gtr:
        if len(val) == 24:
            break
        p = INSTRUCTION + r["question"]
        tgt, fin = gsm_target(r["answer"])
        if ntok(p + tgt) > MAX_TOK or not S.math_score(fin, tgt)["score"]:
            dropped["gsm_train_skip"] += 1
            continue
        split, dst = ("gold_train", gold_train) if len(gold_train) < 400 else ("val", val)
        dst.append(row(f"gsm8k-train-{i}", "math", "gsm8k", split, p, {"answer": fin}, tgt))

    files["math/eval.jsonl"] = math_eval + gsm_eval
    files["math/rft_pool.jsonl"] = math_pool
    files["math/gold_train.jsonl"] = gold_train
    files["math/val.jsonl"] = val

    # ---- logic ----
    _, vclasses = S._load_logic()
    L = load_dataset("PrimeIntellect/INTELLECT-3-RL", "logic", split="train")
    by_task, seen = collections.defaultdict(list), set()
    for i, r in enumerate(L):
        info = json.loads(r["info"])
        t = info["task_name"]
        v = r.get("avg@16_qwen3_4b_instruct_2507") or 0
        if t in SKIP or t not in vclasses or not (0.25 <= v <= 1.0):
            continue
        if r["question"] in seen:
            dropped["logic_dup"] += 1
            continue
        seen.add(r["question"])
        gd = info["game_data_str"] or info["game_data"]
        if not isinstance(gd, str):
            gd = json.dumps(gd)
        if ntok(r["question"]) > MAX_TOK:
            dropped["logic_long"] += 1
            continue
        by_task[t].append(row(f"i3logic-{i}", "logic", "i3-logic", "", r["question"], {"task_name": t, "game_data": gd}))
    rng = random.Random(SEED + 3)
    for t in sorted(by_task):
        rng.shuffle(by_task[t])

    def take_logic(n, split):
        out = []
        while len(out) < n and any(by_task.values()):
            for t in sorted(by_task):
                if by_task[t] and len(out) < n:
                    r = by_task[t].pop(0)
                    r["split"] = split
                    out.append(r)
        return out

    files["logic/eval.jsonl"] = take_logic(100, "eval")
    files["logic/rft_pool.jsonl"] = take_logic(600, "rft_pool")

    blobs = {k: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in v).encode("utf-8") for k, v in files.items()}
    manifest = {"seed": SEED, "max_prompt_tokens": MAX_TOK, "dropped": dict(dropped), "files": {}}
    for k, v in files.items():
        lens = [ntok(r["messages"][0]["content"]) for r in v]
        manifest["files"][k] = {
            "sha256": hashlib.sha256(blobs[k]).hexdigest(), "n": len(v),
            "by_source": dict(collections.Counter(r["source"] for r in v)),
            "by_task_name": dict(sorted(collections.Counter(r["gold"]["task_name"] for r in v if r["family"] == "logic").items())),
            "prompt_tokens_max": max(lens), "prompt_tokens_mean": round(sum(lens) / len(lens), 1)}
    ids = [r["id"] for v in files.values() for r in v]
    assert len(ids) == len(set(ids)), "split overlap"
    return blobs, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    blobs, manifest = build()
    if a.check:
        old = json.load(open(OUT / "manifest.json", encoding="utf-8"))
        same = all(old["files"][k]["sha256"] == manifest["files"][k]["sha256"] for k in manifest["files"])
        print("REPRODUCIBLE" if same else "MISMATCH")
        sys.exit(0 if same else 1)
    for k, b in blobs.items():
        p = OUT / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
