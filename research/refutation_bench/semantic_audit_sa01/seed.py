"""Public SA-01 fixture export and native replay. Gold is public; this is a development seed."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
# (family, shared source, two (declaration, advertised proposition, verdict,
#  certificate body, expected additional axioms))
FAMILIES = [
    ("dependency", """axiom externalLemma : ∀ n : Nat, n + 0 = n
theorem intermediate : ∀ n : Nat, n + 0 = n := externalLemma
theorem submitted : ∀ n : Nat, n + 0 = n := intermediate
theorem clean : ∀ n : Nat, n + 0 = n := Nat.add_zero
""", [
        ("submitted", "∀ n : Nat, n + 0 = n", "extra_axiom", None, ["externalLemma"]),
        ("clean", "∀ n : Nat, n + 0 = n", "supported", "clean", []),
    ]),
    ("encoding", """def encoded (n : Nat) : Prop := n = n
def intended (n : Nat) : Prop := n = 0
theorem submitted : ∀ n : Nat, encoded n := fun n => rfl
""", [
        ("submitted", "∀ n : Nat, intended n", "false_advertisement",
         "by\n  intro h\n  exact (by decide : (1 : Nat) ≠ 0) (h 1)", []),
        ("submitted", "∀ n : Nat, encoded n", "supported", "submitted", []),
    ]),
    ("vacuity", """def eligible (n : Nat) : Prop := n = 0 ∧ n = 1
theorem submitted : ∀ n : Nat, eligible n → n = 7 := by
  intro n h
  have hz : (0 : Nat) = 1 := h.1.symm.trans h.2
  exact False.elim ((by decide : (0 : Nat) ≠ 1) hz)
""", [
        ("submitted", "∃ n : Nat, eligible n", "false_advertisement",
         "by\n  intro h\n  cases h with\n  | intro n hn =>\n    exact (by decide : (0 : Nat) ≠ 1) (hn.1.symm.trans hn.2)", []),
        ("submitted", "∀ n : Nat, eligible n → n = 7", "supported", "submitted", []),
    ]),
    ("domain", """theorem submitted : ∀ n : Nat, n ≤ 1 → n < 2 :=
  fun n h => Nat.lt_succ_of_le h
""", [
        ("submitted", "∀ n : Nat, n < 2", "false_advertisement",
         "by\n  intro h\n  exact (by decide : ¬ ((2 : Nat) < 2)) (h 2)", []),
        ("submitted", "∀ n : Nat, n ≤ 1 → n < 2", "supported", "submitted", []),
    ]),
    ("premise", """theorem submitted : ∀ p : Prop, p → p := fun p h => h
""", [
        ("submitted", "∀ p : Prop, p", "false_advertisement", "by\n  intro h\n  exact h False", []),
        ("submitted", "∀ p : Prop, p → p", "supported", "submitted", []),
    ]),
    ("direction", """theorem submitted : ∀ p q : Prop, (p → q) → p → q :=
  fun p q f hp => f hp
""", [
        ("submitted", "∀ p q : Prop, (p → q) → q → p", "false_advertisement",
         "by\n  intro h\n  exact h False True (fun f => False.elim f) True.intro", []),
        ("submitted", "∀ p q : Prop, (p → q) → p → q", "supported", "submitted", []),
    ]),
    ("definition", """def score (n : Nat) : Nat := n + 1
def independentMeasure (n : Nat) : Nat := n
theorem submitted : ∀ n : Nat, score n = n + 1 := fun n => rfl
""", [
        ("submitted", "∀ n : Nat, independentMeasure n = n + 1", "false_advertisement",
         "by\n  intro h\n  exact (by decide : (0 : Nat) ≠ 1) (h 0)", []),
        ("submitted", "∀ n : Nat, score n = n + 1", "supported", "submitted", []),
    ]),
    ("witness", """def eligible (n : Nat) : Prop := n = 0 ∨ n = 1
theorem submitted : ∃ n : Nat, eligible n := ⟨0, Or.inl rfl⟩
""", [
        ("submitted", "∃ n : Nat, eligible n ∧ n = 2", "false_advertisement",
         "by\n  intro h\n  cases h with\n  | intro n hn =>\n    cases hn.1 with\n    | inl h0 => exact (by decide : (0 : Nat) ≠ 2) (h0.symm.trans hn.2)\n    | inr h1 => exact (by decide : (1 : Nat) ≠ 2) (h1.symm.trans hn.2)", []),
        ("submitted", "∃ n : Nat, eligible n", "supported", "submitted", []),
    ]),
]

def records():
    for family, src, arms in FAMILIES:
        for i, (decl, target, label, proof, extras) in enumerate(arms):
            yield ({"id": "SA01-" + digest(src + "\\n" + decl + "\\n" + target)[:16], "family": family, "source": "import Lean\n\n" + src,
                    "declaration": decl, "advertised": target, "allowed_axioms": sorted(ALLOWED)},
                   {"verdict": label, "certificate": proof, "extra_axioms": extras})

def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def verify_schema(rs):
    assert len(rs) == 16
    assert len({x["id"] for x, _ in rs}) == len(rs)
    for i in range(0, len(rs), 2):
        a, b = rs[i:i+2]
        assert a[0]["source"] == b[0]["source"]
        assert a[0]["family"] == b[0]["family"]
        assert {a[1]["verdict"], b[1]["verdict"]} in (
            {"extra_axiom", "supported"}, {"false_advertisement", "supported"})
    for x, y in rs:
        assert not set(x) & set(y)
        assert not re.search(r"\b(sorry|admit|native_decide|sorryAx)\b", x["source"] + (y["certificate"] or ""))
        assert (y["certificate"] is None) == (y["verdict"] == "extra_axiom")

def read_axioms(output, decl):
    pattern = rf"'{re.escape(decl)}' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])"
    matches = list(re.finditer(pattern, output))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one axiom result for {decl}: {output}")
    return {x.strip() for x in (matches[0].group(2) or "").split(",") if x.strip()}

def run_lean(lean, path, decl):
    try:
        p = subprocess.run([lean, str(path)], capture_output=True, text=True, timeout=60)
        output = p.stdout + p.stderr
        path.with_suffix(".log").write_text(output, encoding="utf-8")
        if p.returncode or re.search(r"error:|sorryAx|uses 'sorry'", output):
            raise ValueError(f"Lean failed ({p.returncode}): {output}")
        return read_axioms(output, decl)
    except subprocess.TimeoutExpired as e:
        path.with_suffix(".log").write_text(str(e), encoding="utf-8")
        raise

def replay(rs, out, lean):
    receipt = []
    for x, y in rs:
        key, decl = x["id"], x["declaration"]
        row = {"id": key, "source_sha256": digest(x["source"]), "expected_verdict": y["verdict"]}
        try:
            path = out / (key + ".lean")
            path.write_text(x["source"] + f"\n#print axioms {decl}\n", encoding="utf-8")
            axioms = run_lean(lean, path, decl)
            row["source_axioms"] = sorted(axioms)
            if axioms - ALLOWED != set(y["extra_axioms"]):
                raise ValueError("selected declaration axiom-policy mismatch")
            if y["certificate"] is not None:
                target = x["advertised"]
                if y["verdict"] == "false_advertisement":
                    target = f"¬ ({target})"
                proof = y["certificate"]
                cert = x["source"] + f"\ntheorem audit_certificate : {target} := {proof}\n#print axioms audit_certificate\n"
                cp = out / (key + "-certificate.lean")
                cp.write_text(cert, encoding="utf-8")
                ca = run_lean(lean, cp, "audit_certificate")
                row["certificate_axioms"] = sorted(ca)
                row["certificate_sha256"] = digest(cert)
                if ca - ALLOWED:
                    raise ValueError("gold certificate uses disallowed axioms")
            row["passed"] = True
        except Exception as exc:
            row["passed"] = False
            row["error"] = str(exc)
        receipt.append(row)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": sum(x["passed"] for x in receipt), "total": len(receipt)}))
    return all(x["passed"] for x in receipt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lean", default="lean")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rs = list(records())
    verify_schema(rs)
    (args.out / "inputs.jsonl").write_text("".join(json.dumps({k: v for k, v in x.items() if k != "family"}, ensure_ascii=False) + "\n" for x, _ in sorted(rs, key=lambda pair: pair[0]["id"])), encoding="utf-8")
    if not args.export_only and not replay(rs, args.out, args.lean):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
