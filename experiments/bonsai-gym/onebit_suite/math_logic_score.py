"""Scorers for the i3-math / gsm8k / i3-logic adapter families.

score(row, reply_text) -> {"score": 0|1, ...}
Math follows envs/i3_math/i3_math/verify.py (math-verify on the last \\boxed{}); logic follows
envs/i3_logic/i3_logic/taskset.py (task_name -> verifier class on Data.from_json_str(game_data)).
CLI: --rows X.jsonl --replies R.jsonl [--out per-item.jsonl]
"""
import argparse
import collections
import json
import sys
import threading
import types
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

ENVS = Path(r"C:\Users\patri\Documents\Codex\onebit-suite\envs")
MATH_VERIFY_TIMEOUT = 5
import logging
logging.getLogger("math_verify").setLevel(logging.ERROR)
BOX = chr(92) + "boxed{"
_logic = None


def _load_logic():
    """Import i3_logic submodules from the env dir. The package __init__ pulls in `verifiers`
    (the runtime), which the verifiers themselves do not need, so register a bare package."""
    global _logic
    if _logic is None:
        pkg = ENVS / "i3_logic" / "i3_logic"
        sys.path.insert(0, str(ENVS / "i3_logic"))
        m = types.ModuleType("i3_logic")
        m.__path__ = [str(pkg)]
        sys.modules["i3_logic"] = m
        from i3_logic.base.data import Data
        from i3_logic.task2verifier import verifier_classes
        _logic = (Data, verifier_classes)
    return _logic


def logic_parse_answer(text: str) -> str:
    if "<think>" in text and "</think>" not in text:
        return ""
    return text.split("</think>")[-1].strip()


def extract_boxed(text: str) -> str:
    start = text.rfind("\\boxed{")
    if start == -1:
        return ""
    i, depth = start + len("\\boxed{"), 1
    while i < len(text) and depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[start + len("\\boxed{"): i - 1] if depth == 0 else ""


def _verify_pair(gold, answer, timeout):
    from math_verify import parse, verify
    return verify(parse(BOX + gold + "}", parsing_timeout=timeout),
                  parse(BOX + answer + "}", parsing_timeout=timeout),
                  timeout_seconds=timeout)


def _verify_with_timeout(gold, answer, timeout):
    """math-verify's own timeout forks a process per call on Windows (very slow), so there it
    runs unbounded inside a watchdog thread and a call over `timeout` seconds scores 0 (as
    the env's verify.py does on a timeout exception). Elsewhere it uses math-verify's own."""
    if sys.platform != "win32":
        return _verify_pair(gold, answer, timeout)
    box = {}

    def run():
        try:
            box["ok"] = _verify_pair(gold, answer, None)
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(2 * timeout)
    if t.is_alive():
        raise TimeoutError("math-verify timeout")
    if "err" in box:
        raise box["err"]
    return box["ok"]


def math_score(gold: str, pred: str, timeout: int = MATH_VERIFY_TIMEOUT) -> dict:
    if "<think>" in pred and "</think>" not in pred:
        return {"score": 0, "reason": "unfinished_think"}
    pred = pred.split("</think>")[-1]
    answer = extract_boxed(pred)
    if not answer:
        return {"score": 0, "reason": "no_boxed"}
    try:
        ok = _verify_with_timeout(gold, answer, timeout)
    except Exception as e:
        return {"score": 0, "reason": "verify_error:" + type(e).__name__, "pred": answer}
    return {"score": int(bool(ok)), "pred": answer}


def logic_score(task_name: str, game_data: str, reply: str) -> dict:
    Data, verifier_classes = _load_logic()
    cls = verifier_classes.get(task_name)
    if cls is None:
        raise ValueError(f"Verifier class not found for task: {task_name}")
    try:
        ok = cls().verify(Data.from_json_str(game_data), logic_parse_answer(reply))
    except Exception as e:
        return {"score": 0, "reason": "verify_error:" + type(e).__name__}
    return {"score": int(bool(ok))}


def score(row: dict, reply_text: str) -> dict:
    reply_text = reply_text or ""
    if row["family"] == "math":
        return math_score(str(row["gold"]["answer"]), reply_text)
    g = row["gold"]
    return logic_score(g["task_name"], g["game_data"], reply_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
    replies = {}
    for l in open(a.replies, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            replies[d["id"]] = d["reply"]
    res, by_src, by_task = [], collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        s = score(r, replies.get(r["id"], ""))
        s["id"] = r["id"]
        res.append(s)
        by_src[r["source"]].append(s["score"])
        if r["family"] == "logic":
            by_task[r["gold"]["task_name"]].append(s["score"])
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    print(f"n={len(res)} mean={mean([s['score'] for s in res]):.4f} missing_replies={sum(r['id'] not in replies for r in rows)}")
    for k, v in sorted(by_src.items()):
        print(f"  source {k}: n={len(v)} mean={mean(v):.4f}")
    for k, v in sorted(by_task.items()):
        print(f"  task {k}: n={len(v)} mean={mean(v):.4f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            for s in res:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
