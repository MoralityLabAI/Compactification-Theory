"""Verify explicit Lean companion certificates against a pinned full set.mm."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

DB_SHA = "9c4bcad276eab5f8b17b231a4543cfe185c19c95cfb7940780dd1de265610c12"
PY_SHA = "a1586636b9f3b8378932e129a1a8299dab8ad39098eca22480583947d1f644b8"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def admit(source, manifest):
    tokens = re.sub(r"\$\([\s\S]*?\$\)", "", source).split()
    stack = [[]]
    labels = set()
    proved = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if token == "$" + "{":
            stack.append([])
            continue
        if token == "$}":
            require(len(stack) > 1, "scope underflow")
            stack.pop()
            continue
        require(re.fullmatch(r"ml-[A-Za-z0-9.-]+", token) and token not in labels, "invalid label")
        labels.add(token)
        kind = tokens[i]
        i += 1
        require(kind in ("$e", "$p"), "new axioms, syntax, variables and includes forbidden")
        end = tokens.index("$." if kind == "$e" else "$=", i)
        statement = tokens[i:end]
        require(statement and statement[0] == "|-" and not any("$" in t for t in statement), "statement")
        i = end + 1
        if kind == "$e":
            require(len(stack) > 1, "unscoped premise")
            stack[-1].append(" ".join(statement))
            continue
        end = tokens.index("$.", i)
        proof = tokens[i:end]
        require(proof and all(re.fullmatch(r"[A-Za-z0-9_.-]+", t) for t in proof), "explicit proof labels")
        i = end + 1
        proved[token] = (" ".join(statement), sum(stack, []), len(proof))
    require(len(stack) == 1, "unclosed scope")
    expected = {p["metamath_label"]: (p["statement"], p["hypotheses"], p["proof_steps"])
                for p in manifest["ports"]}
    require(len(expected) == len(manifest["ports"]) and proved == expected, "manifest statement drift")
    return proved


def run(command, path, positive, c=False):
    proc = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False,
                          stdin=subprocess.DEVNULL)
    log = proc.stdout + proc.stderr
    path.write_text(log, encoding="utf-8")
    success = "All proofs in the database were verified" if c else "No errors were found."
    explicit = "?Error" in log if c else "MMError:" in log or "MMKeyError:" in log
    contradiction = explicit and success in log
    if positive:
        require(proc.returncode == 0 and success in log and not explicit
                and "were not proved" not in log and "incomplete proof" not in log.lower()
                and "Source was not read" not in log, f"positive verification failed: {path}\n{log[-3000:]}")
    else:
        require(proc.returncode in (0, 1) and explicit, f"negative proof not rejected: {path}\n{log[-3000:]}")
    return {"returncode": proc.returncode, "explicit_error": explicit, "success_summary": success in log,
            "contradictory_summary": contradiction, "log_sha256": digest(log.encode())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("database", type=Path)
    p.add_argument("python_verifier", type=Path)
    p.add_argument("c_verifier", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--negative", action="store_true")
    a = p.parse_args()
    directory = Path(__file__).resolve().parent
    repo = directory.parent.parent
    ext = (directory / "ports.mm").read_bytes()
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["schema"] == "lean-setmm-port-v1", "manifest schema")
    proved = admit(ext.decode("ascii"), manifest)
    for port in manifest["ports"]:
        src = repo / port["source"]
        require(blob(src.read_bytes()) == port["source_blob"], f"Lean source drift: {src}")
        require(port["lean_declaration"].split(".")[-1] in src.read_text(), f"missing source name: {src}")
    raw = a.database.read_bytes()
    require(digest(raw) == DB_SHA and digest(a.python_verifier.read_bytes()) == PY_SHA, "upstream hash drift")
    a.out.mkdir(parents=True, exist_ok=True)
    combined = a.out.resolve() / "combined.mm"
    combined.write_bytes(raw + b"\n" + ext)

    def both(db, prefix, positive):
        return {"python": run([sys.executable, str(a.python_verifier.resolve()), str(db), "-v", "1"],
                              a.out / (prefix + "-python.log"), positive),
                "c": run([str(a.c_verifier.resolve()), f'read "{db}"', "verify proof *", "exit"],
                         a.out / (prefix + "-c.log"), positive, c=True)}

    receipt = {"schema": "lean-setmm-port-receipt-v1", "database_sha256": DB_SHA,
               "python_verifier_sha256": PY_SHA, "c_verifier_sha256": digest(a.c_verifier.read_bytes()),
               "extension_sha256": digest(ext), "combined_sha256": digest(combined.read_bytes()),
               "proofs": len(proved), "proof_steps": sum(x[2] for x in proved.values()),
               "positive": both(combined, "positive", True),
               "scope": "Metamath proof validity; human statement correspondence, no automatic Lean translation"}
    if a.negative:
        bad_ext, n = re.subn(rb"[A-Za-z0-9_.-]+(?=\s+\$\.\s*(?:\$\}\s*)?$)", b"wph", ext)
        require(n == 1 and bad_ext != ext, "mutation failed")
        bad = a.out.resolve() / "mutant.mm"
        bad.write_bytes(raw + b"\n" + bad_ext)
        receipt["negative"] = both(bad, "negative", False)
        bad.unlink()
    combined.unlink()
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
