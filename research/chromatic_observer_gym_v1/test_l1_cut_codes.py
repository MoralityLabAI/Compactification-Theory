"""Independent exhaustive replay and exact-rational COE-02 witnesses."""
import itertools
import unittest
from fractions import Fraction as F

try:
    from .l1_cut_codes import (
        metric_matrix, graph_edges, minimum_l1_observer, verify_observer,
        canonical_cuts, crossing, equilateral, dual_width_lower_bound,
        balanced_fractional_certificate)
except ImportError:
    from l1_cut_codes import (
        metric_matrix, graph_edges, minimum_l1_observer, verify_observer,
        canonical_cuts, crossing, equilateral, dual_width_lower_bound,
        balanced_fractional_certificate)


def brute_words(d, edges, L, margin, max_width=4):
    """Independent word-enumerator without cut-BFS, n <= 3."""
    n = len(d)
    for k in range(max_width + 1):
        for tail in itertools.product(range(1 << k), repeat=n - 1):
            words = (0,) + tail
            if all((words[u] ^ words[v]).bit_count() >= margin for u, v in edges):
                if all((words[u] ^ words[v]).bit_count() <= F(L) * d[u][v]
                       for u in range(n) for v in range(u + 1, n)):
                    return k
    return None


class TestCutObserver(unittest.TestCase):
    def test_exhaustive_all_three_vertex_graphs(self):
        n = 3
        pairs = tuple(itertools.combinations(range(n), 2))
        d = equilateral(n)
        for mask in range(1 << len(pairs)):
            edges = tuple(e for i, e in enumerate(pairs) if mask & (1 << i))
            for L in (F(0), F(1), F(2)):
                for margin in (1, 2):
                    with self.subTest(mask=mask, L=L, margin=margin):
                        result = minimum_l1_observer(d, edges, L, margin)
                        expected = brute_words(d, edges, L, margin)
                        self.assertEqual(None if result is None else result.width, expected)
                        if result is not None:
                            self.assertTrue(verify_observer(d, edges, L, margin, result))

    def test_equilateral_chromatic_vs_l1_obstructions(self):
        n = 5
        clique = tuple(itertools.combinations(range(n), 2))
        cycle = tuple((i, (i + 1) % n) for i in range(n))
        d = equilateral(n)
        self.assertIsNone(minimum_l1_observer(d, clique, 1))
        self.assertIsNone(minimum_l1_observer(d, cycle, 1))
        self.assertEqual(minimum_l1_observer(d, clique, 2).width, 4)
        self.assertEqual(minimum_l1_observer(d, clique, 3).width, 3)
        self.assertEqual(minimum_l1_observer(d, cycle, 2).width, 2)
        self.assertEqual(minimum_l1_observer(equilateral(4),
                             tuple(itertools.combinations(range(4), 2)), 2).width, 2)
        self.assertEqual(minimum_l1_observer(d, [], 0).width, 0)
        self.assertEqual(minimum_l1_observer(equilateral(2), [(0,1)], F(1,2)), None)

    def test_explicit_pareto_witness(self):
        d = equilateral(5)
        edges = tuple(itertools.combinations(range(5), 2))
        best = minimum_l1_observer(d, edges, 2)
        self.assertEqual(best.width, 4)
        self.assertEqual(best.words, (0, 1, 2, 4, 8))
        self.assertEqual(max(best.loads), 2)
        self.assertTrue(verify_observer(d, edges, 2, 1, best))

    def test_line_metric_bottleneck_compatible(self):
        x = [F(0), F(3,5), F(6,5)]
        d = tuple(tuple(abs(a-b) for b in x) for a in x)
        self.assertIsNone(minimum_l1_observer(d, [(0,2)], F(49,30)))
        result = minimum_l1_observer(d, [(0,2)], F(5,3))
        self.assertIsNotNone(result)
        self.assertEqual(result.width, 1)
        self.assertTrue(verify_observer(d, [(0,2)], F(5,3), 1, result))

    def test_graph_margin_and_pair_budget(self):
        d = equilateral(3)
        e = [(0,1), (1,2)]
        self.assertIsNone(minimum_l1_observer(d, e, 1, margin=2))
        result = minimum_l1_observer(d, e, 2, margin=2)
        self.assertEqual(result.width, 2)
        self.assertTrue(verify_observer(d, e, 2, 2, result))

    def test_dual_certificates(self):
        d = equilateral(3)
        edges = tuple(itertools.combinations(range(3), 2))
        self.assertEqual(dual_width_lower_bound(d, edges, 2, 1,
                         {e:F(1,2) for e in edges}, {}), 2)
        self.assertEqual(dual_width_lower_bound(equilateral(2), [(0,1)], 3, 3,
                         {(0,1):1}, {}), 3)
        with self.assertRaises(ValueError):
            dual_width_lower_bound(equilateral(2), [(0,1)], 3, 3,
                                   {(0,1):2}, {})

    def test_balanced_fractional_lp_gap(self):
        for n in range(2,10):
            with self.subTest(n=n):
                cuts, weight, value = balanced_fractional_certificate(n)
                self.assertEqual(value, F(n*(n-1)//2, (n*n)//4))
                self.assertEqual(sum(weight*crossing(c,0,1) for c in cuts),1)
        cuts, weight, v = balanced_fractional_certificate(5)
        self.assertEqual((len(cuts),weight,v), (10,F(1,6),F(5,3)))
        self.assertEqual(F(4,1)/v, F(12,5))

    def test_errors_and_resource_guard(self):
        with self.assertRaises(ValueError):
            metric_matrix(((0,1,3),(1,0,1),(3,1,0)))
        with self.assertRaises(ValueError):
            graph_edges(3,[(1,1)])
        with self.assertRaises(ValueError):
            minimum_l1_observer(equilateral(2), [(0,1)], -1)
        with self.assertRaises(RuntimeError):
            minimum_l1_observer(equilateral(5), tuple(itertools.combinations(range(5),2)),
                                2, max_states=1)
        self.assertEqual(canonical_cuts(1), ())


if __name__ == '__main__':
    unittest.main()

class TestActivationAdapter(unittest.TestCase):
    def test_fixed_prototype_report(self):
        import json
        from pathlib import Path
        try:
            from .activation_case import analyse
        except ImportError:
            from activation_case import analyse
        path = Path(__file__).with_name('frozen_k5_onehot.json')
        case = json.loads(path.read_text(encoding='utf8'))
        result = analyse(case)
        self.assertEqual(result['linf']['width'],3)
        self.assertEqual(result['l1']['width'],4)
        self.assertEqual(result['provenance'],case['provenance'])
        self.assertEqual(result['distance'][0][1], '1')
        case['lipschitz']='3'
        self.assertEqual(analyse(case)['l1']['width'],3)
        case['lipschitz']='1'
        self.assertIsNone(analyse(case)['l1'])
        self.assertEqual(analyse(case)['linf']['width'],3)

    def test_rejects_wrong_provenance_shape(self):
        try:
            from .activation_case import analyse
        except ImportError:
            from activation_case import analyse
        base={'format':'coe02-activation-v1','lipschitz':'2',
              'states':[{'id':'a','activation':['1']},{'id':'b','activation':['1']}],
              'evaluator_edges':[['a','b']]}
        with self.assertRaises(ValueError):
            analyse(base)
        base['states'][1]['activation']=['2']
        base['evaluator_edges']=[['a','unknown']]
        with self.assertRaises(ValueError):
            analyse(base)
