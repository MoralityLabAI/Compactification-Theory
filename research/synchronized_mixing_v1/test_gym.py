from fractions import Fraction
from itertools import permutations
import unittest
import gym as dm


class DualMixingTests(unittest.TestCase):
    def test_uniform_stationarity(self):
        gs = tuple(dm.perfect_matchings(tuple(range(4))))
        ps = tuple(permutations(range(4)))
        for kernel, domain in ((dm.switch_kernel(gs), gs), (dm.sweep_kernel(ps), ps)):
            uniform = {x: Fraction(1, len(domain)) for x in domain}
            self.assertEqual(dm.push(uniform, kernel), uniform)

    def test_exact_shared_quantile_marginals(self):
        a = {0: Fraction(1, 4), 1: Fraction(3, 4)}
        b = {'a': Fraction(1, 2), 'b': Fraction(1, 2)}
        coupling = dm.shared_quantile(a, b)
        for x in a:
            self.assertEqual(sum(v for (i, _), v in coupling.items() if i == x), a[x])
        for y in b:
            self.assertEqual(sum(v for (_, j), v in coupling.items() if j == y), b[y])

    def test_coupling_bound_exact(self):
        for share in (Fraction(0), Fraction(1, 4), Fraction(1, 2)):
            result = dm.evaluate(share, horizon=4, block=3)
            self.assertEqual(result['orbit_lock_tv'], '2/3')
            for row in result['rows']:
                self.assertLessEqual(Fraction(row['coupled_tv']), Fraction(row['telescoping_bound']))
                self.assertLessEqual(Fraction(row['coupled_tv']), Fraction(row['block_bound']))

    def test_product_joint_tv_sandwich(self):
        example = dm.evaluate(Fraction(1, 4), horizon=4)
        for row in example['rows']:
            a = Fraction(row['graph_tv'])
            b = Fraction(row['permutation_tv'])
            joint = Fraction(row['independent_tv'])
            self.assertLessEqual(max(a, b), joint)
            self.assertLessEqual(joint, a + b - a * b)


if __name__ == '__main__':
    unittest.main()
