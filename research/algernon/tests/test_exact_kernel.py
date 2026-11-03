import unittest

from research.algernon.exact_kernel import (
    SPLIT_FAMILIES, Task, compile_task, make_case, oracle_bfs,
    run_compiled, run_suite,
)


class ExactKernelTests(unittest.TestCase):
    def test_boundary_and_direction(self):
        spec = Task(n=4, source=0, goal=3, edges=((0, 1), (1, 2), (2, 3)))
        r = run_compiled(spec, compile_task(spec))
        self.assertTrue(r["reachable"])
        self.assertEqual(r["visited_mask"], 15)
        rev = Task(n=4, source=3, goal=0, edges=spec.edges)
        self.assertFalse(run_compiled(rev, compile_task(rev))["reachable"])

    def test_repeated_edges_and_cycles(self):
        task = Task(n=3, source=1, goal=0,
                    edges=((0, 1), (0, 1), (1, 2), (2, 0)))
        result = run_compiled(task, compile_task(task))
        self.assertTrue(result["reachable"])
        self.assertEqual(result["visited_mask"], 7)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Task(17, 0, 0, ())
        with self.assertRaises(ValueError):
            Task(3, 0, 0, ((2, 3),))
        with self.assertRaises(ValueError):
            run_compiled(Task(2, 0, 1, ()), (0, 4))

    def test_random_oracle_parity(self):
        for family in {f for families in SPLIT_FAMILIES.values() for f in families}:
            for i in range(200):
                task = make_case(family, 209317, i)
                a = run_compiled(task, compile_task(task))
                expected, mask = oracle_bfs(task)
                self.assertEqual((a["reachable"], a["visited_mask"]),
                                 (expected, mask))

    def test_family_disjoint_and_determinism(self):
        families = list(SPLIT_FAMILIES.values())
        for i, first in enumerate(families):
            for second in families[i + 1:]:
                self.assertTrue(set(first).isdisjoint(second))
        a = run_suite(seed=7, cases_per_family=3)
        b = run_suite(seed=7, cases_per_family=3)
        self.assertEqual(a, b)
        self.assertEqual(a["n"], 15)
        self.assertEqual(a["reference_mismatches"], 0)
        self.assertEqual(len({r["id"] for r in a["records"]}), 15)


if __name__ == "__main__":
    unittest.main()
