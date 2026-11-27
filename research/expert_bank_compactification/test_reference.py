import itertools
import unittest
from fractions import Fraction as F
from reference import mask, quotient, full_trace_classes, exact_static, normalized_tv, fixture


class TestEBC(unittest.TestCase):
    def test_frontier(self):
        f = fixture()
        self.assertEqual([f['capacity_frontier'][str(k)]['classes'] for k in (1,2,3)], [4,2,1])
        self.assertEqual(len(set(f['future_trace_classes'])), 4)
        self.assertEqual(f['static_counterexample'], (7, (0,1,2)))

    def test_independent_exhaustion(self):
        # Independent exhaustive labelled maps, including all unary dynamics.
        checks = 0
        for nexts in itertools.product(range(3), repeat=3):
            succ = [(s,) for s in nexts]
            for demand in itertools.product((1,2,4), repeat=3):
                for cap in (1,2,3):
                    best = 3
                    for labels in itertools.product(range(3), repeat=3):
                        support = [0]*3
                        for s, c in enumerate(labels):
                            support[c] |= demand[s]
                        if max(x.bit_count() for x in support) > cap:
                            continue
                        if any(labels[i] == labels[j] and labels[nexts[i]] != labels[nexts[j]]
                               for i,j in itertools.combinations(range(3),2)):
                            continue
                        best = min(best, len(set(labels)))
                    got = quotient(demand, succ, cap)
                    self.assertEqual(got['classes'], best)
                    for i,c in enumerate(got['labels']):
                        self.assertEqual(got['supports'][c] & demand[i], demand[i])
                        self.assertEqual(got['successors'][c], (got['labels'][nexts[i]],))
                    if cap == 1:
                        self.assertEqual(best, len(set(full_trace_classes(demand, succ))))
                    checks += 1
        self.assertEqual(checks, 2187)

    def test_pairwise_is_insufficient(self):
        self.assertEqual(quotient([1,2,4], [(0,), (1,), (2,)], 2)['classes'], 2)

    def test_tv(self):
        checks = 0
        for c,a,b in itertools.product(range(6), repeat=3):
            if min(c+a,c+b) == 0:
                continue
            p = (F(c,c+a), F(a,c+a), F(0))
            q = (F(c,c+b), F(0), F(b,c+b))
            self.assertEqual(normalized_tv(c,a,b), sum(abs(x-y) for x,y in zip(p,q))/2)
            checks += 1
        self.assertEqual(checks, 205)

    def test_invalid(self):
        for d,s,k in [([3], [(0,)], 1), ([1], [(1,)], 1), ([0], [(0,)], 1)]:
            with self.assertRaises(ValueError):
                quotient(d,s,k)
        with self.assertRaises(ValueError):
            normalized_tv(0,0,1)


if __name__ == '__main__':
    unittest.main()
