from fractions import Fraction as F
from itertools import combinations, product
import unittest
from spectral import ultrametric, dirac_spectrum, commutator_norm, recovery_certificates


class SpectralTests(unittest.TestCase):
    def test_ultrametric_inequality(self):
        d = ultrametric(tuple(product((0, 1), repeat=3)))
        for i, j, k in product(range(len(d)), repeat=3):
            self.assertLessEqual(d[i][j], max(d[i][k], d[k][j]))

    def test_exact_metric_recovery(self):
        for width in range(1, 5):
            d = ultrametric(tuple(product((0, 1), repeat=width)))
            certs = recovery_certificates(d)
            self.assertEqual(len(certs), len(d) * (len(d) - 1) // 2)
            self.assertTrue(all(achieved == d[i][j] and norm <= 1 for i, j, achieved, norm in certs))

    def test_dirac_spectral_symmetry_and_dimension(self):
        d = ultrametric(((0, 0), (0, 1), (1, 0)))
        spectrum = dirac_spectrum(d)
        self.assertEqual(len(spectrum), len(d) * (len(d) - 1))
        self.assertEqual(spectrum, tuple(sorted(-x for x in spectrum)))

    def test_observable_commutator(self):
        d = ultrametric(((0,), (1,)))
        self.assertEqual(commutator_norm(d, (F(0), F(1))), F(2))
        self.assertEqual(commutator_norm(d, (F(7), F(7))), F(0))

    def test_sentence_order_changes_spectrum(self):
        p = ((0, 0, 0), (0, 0, 1), (0, 1, 1))
        a = ultrametric(p)
        b = ultrametric(p, order=(2, 1, 0))
        self.assertNotEqual(dirac_spectrum(a), dirac_spectrum(b))
        self.assertEqual([a[i][j] for i, j in combinations(range(3), 2)], [F(1, 8), F(1, 4), F(1, 4)])
        self.assertEqual([b[i][j] for i, j in combinations(range(3), 2)], [F(1, 2), F(1, 2), F(1, 4)])

    def test_spectrum_forgets_marked_point_names(self):
        p = ((0, 0, 0), (0, 0, 1), (0, 1, 1))
        a = ultrametric(p)
        b = ultrametric((p[0], p[2], p[1]))
        self.assertEqual(dirac_spectrum(a), dirac_spectrum(b))
        self.assertNotEqual(a[0][1], b[0][1])

    def test_scale_changes_spectrum(self):
        p = ((0, 0), (0, 1), (1, 1))
        a = ultrametric(p)
        b = ultrametric(p, weights=(F(1), F(1, 2)))
        self.assertEqual(dirac_spectrum(b), tuple(x / 2 for x in dirac_spectrum(a)))

    def test_invalid_inputs(self):
        for p in ((), ((0,),), ((0,), (0,)), ((0,), (1, 0)), ((0,), (2,))):
            with self.assertRaises(ValueError):
                ultrametric(p)
        with self.assertRaises(ValueError):
            ultrametric(((0, 0), (1, 1)), order=(0, 0))
        with self.assertRaises(ValueError):
            ultrametric(((0, 0), (1, 1)), weights=(F(1), F(2)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
