"""ARG-02 torch-free tests: split identities, public-only prompts, grading, statistics."""
from __future__ import annotations

from dataclasses import replace
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from spiral_ln import algebraic_reasoning_oracle as oracle

from research.algernon.algebraic_reasoning_gym import external
from research.algernon.algebraic_reasoning_gym.arg02 import flops, prompts, scoring, splits, stats
from research.algernon.algebraic_reasoning_gym.arg02.settings import load_config, load_lock

OPS = oracle.OPS
OP_IDS = {"I": 101, "N": 102, "S": 103, "R": 104}


def record_for(program, *, hi=-0.1, lo=-5.0):
    """Synthetic answer record whose joint argmax is `program`."""
    a, b, c = (OPS.index(x) for x in program)
    lp1 = [hi if i == a else lo for i in range(4)]
    lp2 = [[hi if (i == a and j == b) else lo for j in range(4)] for i in range(4)]
    lp3 = [[[hi if (i == a and j == b and k == c) else lo for k in range(4)] for j in range(4)] for i in range(4)]
    g2 = [OP_IDS[program[1]]] * 4
    g3 = [[OP_IDS[program[2]]] * 4 for _ in range(4)]
    return {"lp1": lp1, "lp2": lp2, "lp3": lp3, "g1": OP_IDS[program[0]], "g2": g2, "g3": g3}


class SplitTests(unittest.TestCase):
    def test_audit_disjoint_and_sized(self):
        a = splits.audit()
        self.assertEqual(a["counts"], {"test": 192, "test_positive": 168, "calibration": 36, "dev": 96, "shots": 4})
        self.assertTrue(all(v == 0 for v in a["signature_overlaps"].values()), a["signature_overlaps"])
        self.assertEqual(a["dev_families"], sorted(oracle.CAL_FAMILIES))
        self.assertEqual(a["test_duplicate_public_signatures"], 6)  # the six collision pairs

    def test_test_rows_are_arg01_public_export(self):
        self.assertEqual(list(external.public_records()), [splits.public_record(c) for c in splits.test_cases()])

    def test_shots_solve_their_demos(self):
        for c in splits.shot_cases():
            row = splits.shot_record(c)
            ok, faults = scoring.public_check(row, row["program"])
            self.assertTrue(ok, faults)
        with self.assertRaises(ValueError):
            splits.shot_record(splits.test_cases()[0])

    def test_dev_cases_are_valid_and_deterministic(self):
        dev = splits.dev_cases()
        self.assertEqual(len(dev), 96)
        for c in dev[:12]:
            self.assertEqual(scoring.public_check(splits.public_record(c), c.source_graph)[0], True)
        self.assertEqual(splits.roster_digest(dev), splits.roster_digest(splits.dev_cases.__wrapped__()))


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.case = splits.test_cases()[5]
        self.row = splits.public_record(self.case)

    def test_prompt_ignores_hidden_fields_and_query(self):
        hidden = replace(self.case, source_graph=None, future=(1,) * len(self.case.future))
        self.assertEqual(prompts.render(self.row), prompts.render(splits.public_record(hidden)))
        moved = dict(self.row, query=[1] + [7] * 12)
        self.assertEqual(prompts.render(self.row), prompts.render(moved))
        self.assertNotIn(self.case.identifier, prompts.render(self.row))

    def test_non_public_fields_rejected(self):
        for key in ("future", "source_graph", "family", "split", "program"):
            with self.assertRaises(ValueError):
                prompts.render(dict(self.row, **{key: 1}))

    def test_tail_and_variants(self):
        text = prompts.render(self.row)
        self.assertTrue(text.endswith(prompts.TAIL))
        swap = prompts.render(self.row, variant="swap")
        d0, d1 = self.row["demos"]
        self.assertIn("Input 1: " + ", ".join(map(str, d1["input"])), swap)
        self.assertIn("Input 2: " + ", ".join(map(str, d0["input"])), swap)
        self.assertIn("Input 1: " + " ".join(map(str, d0["input"])), prompts.render(self.row, variant="space"))

    def test_collision_pairs_render_identically(self):
        col = [c for c in splits.test_cases() if c.split == "collision"]
        for i in range(0, len(col), 2):
            self.assertEqual(prompts.render(splits.public_record(col[i])), prompts.render(splits.public_record(col[i + 1])))

    def test_feedback_lines(self):
        line = prompts.feedback_line((("I", "R", "N"), [3, None]))
        self.assertEqual(line, "Rejected: I R N (example 1 differs at x^3, example 2 ok)")
        swapped = prompts.feedback_line((("I", "R", "N"), [3, None]), swapped=True)
        self.assertEqual(swapped, "Rejected: I R N (example 1 ok, example 2 differs at x^3)")
        text = prompts.render(self.row, feedback=[(("I", "R", "N"), [3, None])])
        self.assertTrue(text.endswith("Rejected: I R N (example 1 differs at x^3, example 2 ok)" + prompts.TAIL))

    def test_shots_render_with_answers(self):
        shots = [splits.shot_record(c) for c in splits.shot_cases()]
        text = prompts.render(self.row, shots=shots)
        for s in shots:
            self.assertIn("Program: " + " ".join(s["program"]) + "\n", text)
        self.assertEqual(text.count(prompts.TAIL), 5)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.cases = [c for c in splits.test_cases() if c.split in splits.POSITIVE_SPLITS][:20]

    def test_source_program_ranked_first_is_accepted(self):
        for case in self.cases:
            g = scoring.score_record(case, record_for(case.source_graph), OP_IDS)
            self.assertEqual(g["top1"], list(case.source_graph))
            self.assertEqual(g["rank_of_source"], 1)
            self.assertEqual(g["at1"]["accepted"], 1)
            self.assertEqual(g["at1"]["queries"], 1)
            self.assertEqual(g["greedy_valid"], 1)
            self.assertEqual(g["greedy"]["accepted"], 1)

    def test_greedy_leaving_grammar_is_invalid(self):
        rec = record_for(("I", "I", "I"))
        rec["g1"] = 7
        self.assertIsNone(scoring.greedy_program(rec, OP_IDS))
        rec = record_for(("I", "I", "I"))
        rec["g3"][0][0] = 9
        self.assertIsNone(scoring.greedy_program(rec, OP_IDS))

    def test_ranking_ties_and_budget(self):
        flat = [0.0] * 64
        self.assertEqual(scoring.ranking(flat), list(oracle.PROGRAMS))
        g = scoring.grade_ranking(self.cases[0], scoring.ranking(flat))
        self.assertLessEqual(g["at12"]["queries"], 12)
        self.assertEqual(g["at12"]["trace"], [list(p) for p in oracle.PROGRAMS[:g["at12"]["queries"]]])

    def test_ensemble_mean(self):
        a, b = record_for(("I", "N", "S")), record_for(("R", "R", "R"))
        s = scoring.ensemble_scores([a, b])
        ja, jb = scoring.joint_scores(a), scoring.joint_scores(b)
        self.assertEqual(s, [(x + y) / 2 for x, y in zip(ja, jb)])

    def test_public_check_matches_oracle(self):
        rng = random.Random(5)
        for case in self.cases:
            for _ in range(4):
                p = rng.choice(oracle.PROGRAMS)
                ok, faults = scoring.public_check(splits.public_record(case), p)
                ref = oracle.check_candidate(case, p)
                self.assertEqual(ok, ref["accepted"])
                self.assertEqual(faults, ref["first_faults"])

    def test_controller_regrade(self):
        case = self.cases[0]
        wrong = [p for p in oracle.PROGRAMS if not oracle.check_candidate(case, p)["accepted"]][:3]
        g = scoring.score_controller(case, wrong + [case.source_graph])
        self.assertEqual((g["accepted"], g["queries"]), (1, 4))
        with self.assertRaises(ValueError):
            scoring.score_controller(case, [wrong[0], wrong[0]])

    def test_non_finite_rejected(self):
        rec = record_for(("I", "I", "I"))
        rec["lp1"][0] = float("nan")
        with self.assertRaises(ValueError):
            scoring.joint_scores(rec)


class StatsTests(unittest.TestCase):
    def test_mcnemar(self):
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(stats.mcnemar_exact(5, 0), 2 / 32)
        self.assertAlmostEqual(stats.mcnemar_exact(3, 3), 1.0)

    def test_paired_and_bootstrap(self):
        a = [1] * 30 + [0] * 10
        b = [0] * 20 + [1] * 10 + [0] * 10
        cl = [i % 8 for i in range(40)]
        r = stats.paired(a, b, cl, boot=2000)
        self.assertEqual((r["a_only"], r["b_only"], r["difference"]), (20, 0, 20))
        self.assertGreater(r["cluster_ci95"][0], 0)
        same = stats.paired(a, a, cl, boot=200)
        self.assertEqual(same["cluster_ci95"], [0.0, 0.0])

    def test_wilson_and_quantile(self):
        lo, hi = stats.wilson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertEqual(stats.quantile([1, 2, 3, 4], 0.5), 2.5)


class FlopTests(unittest.TestCase):
    def test_qwen35_08b_dense_macs_match_size(self):
        tc = load_lock()["model"]["text_config"]
        per_token = sum(flops.layer_macs(tc, i, 1) for i in range(tc["num_hidden_layers"])) - \
            tc["num_attention_heads"] * tc["head_dim"] * 2 * 6
        self.assertGreater(per_token, 0.40e9)
        self.assertLess(per_token, 0.60e9)

    def test_paths_and_ensemble_sizes(self):
        tc = load_lock()["model"]["text_config"]
        self.assertEqual(len(flops.path(tc, (12, 16), 4)), 36)
        self.assertEqual(flops.path(tc, (12, 16), 1), list(range(24)))
        self.assertEqual(flops.ensemble_sizes(24, (12, 16), [1, 2, 4, 8, 16], 4), {1: 1, 2: 1, 4: 2, 8: 2, 16: 4})
        cfg = load_config()
        self.assertEqual(len(cfg["ensemble_variants"]), 4)
        self.assertTrue(all(len(b) == 2 for b in cfg["block_candidates"]))

    def test_lock_pins_revisions(self):
        lock = load_lock()
        for k in ("model", "ff_control"):
            self.assertRegex(lock[k]["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(all(len(v) == 64 for v in lock[k]["files_sha256"].values()))
        self.assertFalse(lock["recurrence"]["native_recurrence"])


class ProbeTests(unittest.TestCase):
    def test_probe_on_separable_data(self):
        import numpy as np
        from research.algernon.algebraic_reasoning_gym.arg02 import probe
        rng = np.random.default_rng(0)
        centres = rng.normal(size=(4, 20)) * 3

        def make(n):
            y = [OPS[i % 4] for i in range(n)]
            x = np.stack([centres[OPS.index(v)] + rng.normal(size=20) for v in y])
            return x, y
        tx, ty = make(80)
        sx, sy = make(24)
        ex, ey = make(40)
        r = probe.evaluate(tx, ty, sx, sy, {"novel": (ex, ey)}, list(OPS), permutations=20)
        self.assertGreater(r["tests"]["novel"]["accuracy"], 0.9)
        self.assertLess(r["tests"]["novel"]["permutation_p"], 0.1)


class SymbolicTests(unittest.TestCase):
    def test_reuses_arg01_arms(self):
        from research.algernon.algebraic_reasoning_gym.arg02 import symbolic
        cases = [c for c in splits.test_cases() if c.split == "signed"][:3]
        out, meta = symbolic.run(cases)
        for c in cases:
            self.assertEqual(out["exhaustive_64"][c.identifier]["accepted"], 1)
            self.assertLessEqual(out["certified_mass12"][c.identifier]["queries"], 12)
        self.assertGreater(meta["class_compile_ns"], 0)


class ExportTests(unittest.TestCase):
    def test_export_writes_public_rows_only(self):
        from research.algernon.algebraic_reasoning_gym.arg02 import analyze
        with TemporaryDirectory() as tmp:
            analyze.main(["export", "--run-dir", tmp, "--sample", "1"])
            for name in ("test.jsonl", "development.jsonl"):
                for line in (Path(tmp) / "public" / name).read_text().splitlines():
                    self.assertEqual(set(json.loads(line)), prompts.PUBLIC_FIELDS)
            shots = json.loads((Path(tmp) / "public" / "shots.json").read_text())
            self.assertEqual({tuple(sorted(s)) for s in shots}, {tuple(sorted(prompts.SHOT_FIELDS))})


if __name__ == "__main__":
    unittest.main()
