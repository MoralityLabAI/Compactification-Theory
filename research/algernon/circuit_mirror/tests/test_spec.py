import itertools
import unittest
from research.algernon.circuit_mirror.spec import (
    P, PointerTask, build, case, emit_circom, witness_for, witness_for_trace,
)


class TestR1CS(unittest.TestCase):
    def test_exhaustive_small_machines(self):
        count = 0
        for f in itertools.product(range(3), repeat=3):
            for initial in range(3):
                for steps in (0, 1, 2, 3, 4, 8, 12):
                    task = PointerTask(f, initial, steps)
                    c = build(task)
                    w, public = witness_for(task, c)
                    self.assertTrue(c.check(w, public)["valid"])
                    self.assertEqual(w[c.indices[f"final[{task.trace()[-1]}]"]], 1)
                    count += 1
        self.assertEqual(count, 567)

    def test_corrupted_product_and_state_fail(self):
        task = PointerTask((1, 2, 0, 0), 0, 4)
        c = build(task)
        w, public = witness_for(task, c)
        for name in ("prod[0][0][1]", "state[2][1]", "final[0]", "edge[0][1]", "start[0]"):
            damaged = w[:]
            damaged[c.indices[name]] ^= 1
            self.assertFalse(c.check(damaged, public)["valid"], name)

    def test_bad_claimed_trace_rejected_even_if_final_correct(self):
        t = PointerTask((1,2,0), 0, 3)
        c = build(t)
        # The false trace starts correctly and ends at the oracle's final state,
        # but violates intermediate transitions. Final-answer-only checking misses it.
        w,public=witness_for_trace(t,c,[0,2,2,0])
        self.assertFalse(c.check(w,public)["valid"])

    def test_invalid_public_mismatch(self):
        t = PointerTask((0, 0, 0), 1, 2)
        c = build(t); w, public = witness_for(t, c)
        bad = dict(public); bad["start[1]"] = 0
        self.assertEqual(c.check(w, bad)["error"], "public input mismatch")
        self.assertFalse(c.check(w[:-1], public)["valid"])

    def test_invalid_graph_and_signal(self):
        with self.assertRaises(ValueError): PointerTask((0, 9), 0, 1)
        with self.assertRaises(ValueError): PointerTask((0, 1), 2, 1)
        with self.assertRaises(ValueError): PointerTask((0, 1), 0, -1)
        with self.assertRaises(ValueError): case("unknown", 1, 1)

    def test_circom_json_semantics(self):
        t=case("permutation", 34, 4, n=4, steps=3)
        c=build(t)
        j=c.export_circom_constraints()
        self.assertEqual(len(j["constraints"]), len(c.constraints))
        self.assertTrue(all(len(triple)==3 for triple in j["constraints"]))
        self.assertEqual(c.check(*witness_for(t,c))["violations"], 0)
        source=emit_circom(4,3)
        self.assertIn("component main {public [edge, start]}", source)
        self.assertIn("product[t][i][j] <==", source)
        self.assertTrue(P>2**200)


if __name__=='__main__': unittest.main()
