"""SA-02 finite audit acquisition: deterministic sources, observed evidence, native replay."""
import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

SPLITS = {
    "train": (32, 9201, ("narrow", "expand"), (2, 5)),
    "cal": (12, 9202, ("narrow", "expand"), (2, 5)),
    "test_depth": (20, 9203, ("narrow", "expand"), (7, 12)),
    "test_mechanism": (16, 9204, ("existential", "vacuity"), (2, 5)),
}
BUDGETS = (4, 8, 16, 32, 64, 128)
POLICIES = ("random", "lexical", "proof_first", "semantic_first", "alternating")
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
FULL = (1 << 16) - 1

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def truth(expr):
    op, g, s = expr
    return (g & s) == g if op == "subset" else (g & s) != 0

def statement(expr):
    op, g, s = expr
    left = f"(({g} : Nat) &&& ({s} : Nat))"
    return f"{left} = {g}" if op == "subset" else f"{left} ≠ 0"

def refs(node):
    kind = node["kind"]
    if kind in ("lemma", "proof", "axiom"):
        return [node["type"], *node.get("deps", [])]
    if kind == "alias":
        return [node["ref"]]
    if kind == "formula":
        return [node["g"], node["s"]]
    return []

def generate_family(split, idx, rng, mechanism, depth_range):
    nodes = {}
    def add(kind, **data):
        name = "n" + format(rng.getrandbits(64), "016x")
        assert name not in nodes
        nodes[name] = {"kind": kind, **data}
        return name

    g = rng.randrange(1, FULL)
    s = g | rng.randrange(FULL + 1)
    if s == FULL:
        s = g
    missing = FULL ^ s
    if mechanism == "narrow":
        bit = g & -g
        source = ("subset", g, s)
        bad = ("subset", g, s ^ bit)
    elif mechanism == "expand":
        source = ("subset", g, s)
        bad = ("subset", g | (missing & -missing), s)
    elif mechanism == "existential":
        source = ("nonempty", g, s)
        bad = ("nonempty", g, FULL ^ g)
    else:
        source = ("subset", 0, s)
        bad = ("nonempty", 0, s)
    assert truth(source) and not truth(bad)

    def aliases(name, sort, depth):
        for _ in range(depth):
            name = add("alias", sort=sort, ref=name)
        return name

    def formula(expr):
        op, a, z = expr
        a = aliases(add("mask", value=a), "Nat", rng.randint(*depth_range))
        z = aliases(add("mask", value=z), "Nat", rng.randint(*depth_range))
        p = add("formula", op=op, g=a, s=z)
        return aliases(p, "Prop", rng.randint(*depth_range))

    claim = formula(source)
    good_target = formula(source)
    bad_target = formula(bad)
    def leaf():
        return add("lemma", type=claim)
    def proof(*deps):
        deps = list(deps)
        rng.shuffle(deps)
        return add("proof", type=claim, deps=deps)

    # Shared clean DAG with distinct leaves; all nodes have opaque names.
    leaves = [leaf() for _ in range(4)]
    pool = list(leaves)
    for _ in range(rng.randint(*depth_range) * 2):
        pool.append(proof(*rng.sample(pool, 2)))
    def chain(start):
        for _ in range(rng.randint(*depth_range)):
            start = proof(start, rng.choice(pool)) if rng.random() < .8 else proof(start)
        return start

    forbidden = add("axiom", type=claim)
    dead = add("axiom", type=claim)
    dead_root = chain(dead)
    clean = chain(proof(rng.choice(pool), rng.choice(leaves)))
    tainted = chain(forbidden)
    # Extra outer branches keep the forbidden leaf off the selected root.
    tainted = proof(tainted, rng.choice(pool))
    ns = "F" + str(SPLITS[split][1]) + "_" + str(idx)
    source_lines = ["namespace " + ns]
    for name, node in nodes.items():
        k = node["kind"]
        if k == "mask":
            line = f"def {name} : Nat := {node['value']}"
        elif k == "alias":
            line = f"def {name} : {node['sort']} := {node['ref']}"
        elif k == "formula":
            a, z = node["g"], node["s"]
            term = f"({a} &&& {z}) = {a}" if node["op"] == "subset" else f"({a} &&& {z}) ≠ 0"
            line = f"def {name} : Prop := {term}"
        elif k == "axiom":
            line = f"axiom {name} : {claim}"
        elif k == "lemma":
            line = f"theorem {name} : {claim} := by change {statement(source)}; decide"
        else:
            ds = node["deps"]
            term = ds[0] if len(ds) == 1 else f"And.left (And.intro {ds[0]} {ds[1]})"
            line = f"theorem {name} : {claim} := {term}"
        source_lines.append(line)
    raw = "\n".join(source_lines + ["end " + ns]) + "\n"
    expected = {clean: [], tainted: [forbidden]}
    native = list(source_lines)
    native += [f"#print axioms {clean}", f"#print axioms {tainted}"]
    cert_good, cert_bad = "certGood", "certBad"
    native += [f"theorem {cert_good} : {good_target} := {clean}",
               f"#print axioms {cert_good}",
               f"theorem {cert_bad} : ¬ {bad_target} := by change ¬ ({statement(bad)}); decide",
               f"#print axioms {cert_bad}"]
    expected[cert_good] = []
    expected[cert_bad] = []
    if bad[0] == "subset":
        w = ((bad[1] & ~bad[2]) & -(bad[1] & ~bad[2])).bit_length() - 1
        native += [f"theorem certBit : Nat.testBit {bad[1]} {w} = true ∧ Nat.testBit {bad[2]} {w} = false := by decide",
                   "#print axioms certBit"]
        expected["certBit"] = []
    native.append("end " + ns)
    arms = []
    for root, extra in ((clean, False), (tainted, True)):
        for target, false in ((good_target, False), (bad_target, True)):
            arms.append({"id": sha(ns + root + target)[:20], "root": root, "target": target,
                         "gold": {"axiom": "extra" if extra else "clean",
                                  "semantic": "false" if false else "match",
                                  "verdict": "defect" if extra or false else "supported"}})
    rng.shuffle(arms)
    return {"family": ns, "split": split, "mechanism": mechanism, "graph": nodes,
            "arms": arms, "source": raw, "native": "\n".join(native) + "\n",
            "expected": {ns + "." + k: [ns + "." + a for a in v] for k, v in expected.items()},
            "forbidden": forbidden, "dead_root": dead_root}

def families():
    for split, (n, seed, mechanisms, depths) in SPLITS.items():
        rng = random.Random(seed)
        for idx in range(n):
            yield generate_family(split, idx, rng, mechanisms[idx % len(mechanisms)], depths)

class Session:
    def __init__(self, graph, root, target, budget):
        self._graph = graph
        self.root, self.target, self.budget = root, target, budget
        self.observed = {}
        self.discovered = dict.fromkeys((root, target))
    def read(self, name):
        if name not in self.discovered or name in self.observed or len(self.observed) >= self.budget:
            raise ValueError("invalid or over-budget read")
        # Return a copy of the individual acquired record.
        item = json.loads(json.dumps(self._graph[name]))
        self.observed[name] = item
        for ref in refs(item):
            self.discovered.setdefault(ref, None)
        return item

def proof_evidence(root, seen):
    todo = [(root, [root])]
    visited = set()
    missing = []
    while todo:
        name, path = todo.pop(0)
        if name in visited:
            continue
        visited.add(name)
        if name not in seen:
            missing.append(name)
            continue
        n = seen[name]
        if n["kind"] == "axiom":
            return "extra", {"path": path}, []
        if n["kind"] not in ("proof", "lemma"):
            raise ValueError("unexpected proof node")
        todo.extend((x, path + [x]) for x in n.get("deps", []))
    if missing:
        return "open", None, missing
    return "clean", {"closure": sorted(visited)}, []

def semantic_evidence(root, target, seen):
    trace = set()
    missing = []
    def read(name):
        if name not in seen:
            if name not in missing:
                missing.append(name)
            return None
        trace.add(name)
        return seen[name]
    def resolve(name):
        n = read(name)
        if n is None:
            return None
        if n["kind"] == "alias":
            return resolve(n["ref"])
        if n["kind"] == "mask":
            return n["value"]
        if n["kind"] == "formula":
            a, b = resolve(n["g"]), resolve(n["s"])
            return None if a is None or b is None else (n["op"], a, b)
        raise ValueError("unexpected expression")
    p = read(root)
    source = None if p is None else resolve(p["type"])
    wanted = resolve(target)
    # A false advertised statement alone establishes an advertised defect.
    if wanted is not None and not truth(wanted):
        op, a, b = wanted
        witness = ((a & ~b) & -(a & ~b)).bit_length() - 1 if op == "subset" else None
        return "false", {"trace": sorted(trace), "target": wanted, "bit": witness}, []
    if source is not None and wanted == source:
        return "match", {"trace": sorted(trace), "target": wanted}, []
    return "open", None, missing

def evidence(root, target, seen):
    a, ac, ap = proof_evidence(root, seen)
    s, sc, sp = semantic_evidence(root, target, seen)
    verdict = "defect" if a == "extra" or s == "false" else (
        "supported" if a == "clean" and s == "match" else "abstain")
    return {"axiom": a, "semantic": s, "verdict": verdict,
            "axiom_certificate": ac, "semantic_certificate": sc}, ap, sp

def check_certificate(graph, arm, acquired, result):
    """Independent checker: verify supplied path/closure and normalized-expression evidence."""
    seen = set(acquired)
    def get(n):
        if n not in seen:
            raise ValueError("certificate needs unobserved node")
        return graph[n]
    a, s = result["axiom"], result["semantic"]
    ac, sc = result["axiom_certificate"], result["semantic_certificate"]
    if a == "extra":
        path = ac["path"]
        assert path and path[0] == arm["root"]
        for x, y in zip(path, path[1:]):
            assert y in get(x).get("deps", [])
        assert get(path[-1])["kind"] == "axiom"
    elif a == "clean":
        closure = set(ac["closure"])
        reached, frontier = set(), [arm["root"]]
        while frontier:
            x = frontier.pop()
            if x in reached:
                continue
            reached.add(x)
            node = get(x)
            assert node["kind"] in ("lemma", "proof")
            frontier.extend(node.get("deps", []))
        assert closure == reached
    else:
        assert a == "open" and ac is None
    def normalize(x):
        n = get(x)
        while n["kind"] == "alias":
            n = get(n["ref"])
        if n["kind"] == "mask":
            return n["value"]
        assert n["kind"] == "formula"
        return (n["op"], normalize(n["g"]), normalize(n["s"]))
    if s != "open":
        assert set(sc["trace"]) <= seen
        target = normalize(arm["target"])
        assert tuple(sc["target"]) == target
        op, g, t = target
        if s == "match":
            assert target == normalize(get(arm["root"])["type"])
        elif s == "false":
            if op == "subset":
                bit = sc["bit"]
                assert 0 <= bit < 16 and g & (1 << bit) and not t & (1 << bit)
            else:
                assert op == "nonempty" and not g & t and sc["bit"] is None
        else:
            raise ValueError("bad semantic status")
    else:
        assert sc is None
    expected = "defect" if a == "extra" or s == "false" else (
        "supported" if a == "clean" and s == "match" else "abstain")
    assert result["verdict"] == expected
    return True

def run_policy(graph, arm, budget, policy):
    session = Session(graph, arm["root"], arm["target"], budget)
    rng = random.Random(int(arm["id"], 16) + budget)
    for i in range(budget):
        result, ap, sp = evidence(session.root, session.target, session.observed)
        if result["axiom"] != "open" and result["semantic"] != "open":
            break
        frontier = [x for x in session.discovered if x not in session.observed]
        candidates = list(dict.fromkeys(ap + sp))
        if not candidates:
            break
        assert set(candidates) <= set(frontier)
        if policy == "random":
            chosen = rng.choice(frontier)
        elif policy == "lexical":
            chosen = min(frontier)
        elif policy == "proof_first":
            chosen = (ap or sp)[0]
        elif policy == "semantic_first":
            chosen = (sp or ap)[0]
        else:
            chosen = ((ap or sp) if i % 2 == 0 else (sp or ap))[0]
        session.read(chosen)
    result, _, _ = evidence(session.root, session.target, session.observed)
    check_certificate(graph, arm, session.observed, result)
    result["reads"] = len(session.observed)
    result["acquired"] = list(session.observed)
    return result

def validate(fs):
    assert len(fs) == 80
    assert len({sha(f["source"]) for f in fs}) == 80
    assert len({a["id"] for f in fs for a in f["arms"]}) == 320
    for f in fs:
        graph = f["graph"]
        assert all(x in graph for n in graph.values() for x in refs(n))
        assert len(f["arms"]) == 4
        assert {(a["gold"]["axiom"], a["gold"]["semantic"]) for a in f["arms"]} == {
            ("clean", "match"), ("clean", "false"), ("extra", "match"), ("extra", "false")}
        assert not re.search(r"\b(sorry|admit|native_decide|sorryAx)\b", f["native"])
        for arm in f["arms"]:
            result, _, _ = evidence(arm["root"], arm["target"], graph)
            assert all(result[k] == v for k, v in arm["gold"].items())
            check_certificate(graph, arm, graph, result)
            s = Session(graph, arm["root"], arm["target"], 1)
            for name in (f["dead_root"],):
                try:
                    s.read(name)
                except ValueError:
                    pass
                else:
                    raise AssertionError("undiscovered read allowed")
            s.read(arm["root"])
            try:
                s.read(arm["target"])
            except ValueError:
                pass
            else:
                raise AssertionError("budget exceeded")

def baseline(fs, out):
    rows = []
    for split in SPLITS:
        subset = [f for f in fs if f["split"] == split]
        for budget in BUDGETS:
            for policy in POLICIES:
                counts = Counter()
                total = 4 * len(subset)
                for f in subset:
                    joint = True
                    for arm in f["arms"]:
                        p = run_policy(f["graph"], arm, budget, policy)
                        gold = arm["gold"]
                        for key in ("axiom", "semantic"):
                            assert p[key] in ("open", gold[key]), (arm["id"], key, p)
                            counts[key] += p[key] != "open"
                        correct = p["verdict"] == gold["verdict"]
                        joint &= correct
                        counts["correct"] += correct
                        counts["abstain"] += p["verdict"] == "abstain"
                        counts["false_rejection"] += p["verdict"] == "defect" and gold["verdict"] == "supported"
                        counts["incorrect_support"] += p["verdict"] == "supported" and gold["verdict"] == "defect"
                        counts["reads"] += p["reads"]
                        if budget == 128:
                            assert correct, (f["family"], arm["id"], policy, p)
                    counts["joint"] += joint
                rows.append({"split": split, "budget": budget, "policy": policy,
                             "n": total, "families": len(subset), **counts,
                             "mean_reads": counts["reads"] / total})
    (out / "baseline.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps([x for x in rows if x["budget"] in (16, 32, 64) and x["split"].startswith("test")], indent=2))

def native(fs, out, lean):
    for split in SPLITS:
        subset = [f for f in fs if f["split"] == split]
        expected = {k: v for f in subset for k, v in f["expected"].items()}
        path = out / (split + ".lean")
        path.write_text("import Lean\n\n" + "\n".join(f["native"] for f in subset))
        (out / (split + "-expected.json")).write_text(json.dumps(expected, indent=2))
        proc = subprocess.run([lean, str(path)], capture_output=True, text=True, timeout=240)
        output = proc.stdout + proc.stderr
        (out / (split + ".log")).write_text(output)
        assert proc.returncode == 0 and not re.search(r"error:|sorryAx|uses 'sorry'", output), output[-8000:]
        found = {}
        for match in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])", output):
            name = match[1]
            assert name not in found
            found[name] = {x.strip() for x in (match[3] or "").split(",") if x.strip()}
        assert set(found) == set(expected), (set(expected) - set(found), set(found) - set(expected))
        for name, axioms in expected.items():
            assert found[name] - ALLOWED == set(axioms), (name, found[name], axioms)
        print(f"{split}: {len(subset)} native source families, {len(expected)} axiom checks")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lean", default="lean")
    ap.add_argument("--skip-native", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fs = list(families())
    validate(fs)
    for split in SPLITS:
        subset = [f for f in fs if f["split"] == split]
        inp = [{"id": a["id"], "root": a["root"], "target": a["target"]}
               for f in subset for a in f["arms"]]
        (args.out / (split + "-input.jsonl")).write_text("".join(json.dumps(x) + "\n" for x in sorted(inp, key=lambda x: x["id"])))
        (args.out / (split + "-environment.json")).write_text(json.dumps(subset))
    baseline(fs, args.out)
    if not args.skip_native:
        native(fs, args.out, args.lean)

if __name__ == "__main__":
    main()
