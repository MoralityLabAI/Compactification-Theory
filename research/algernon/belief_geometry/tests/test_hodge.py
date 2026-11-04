import unittest
from math import comb
try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'NumPy needed for Hodge numeric checks')
class HodgeTests(unittest.TestCase):
    def test_boundary_and_exact_spectrum(self):
        from research.algernon.belief_geometry.hodge import cube,theoretical_spectrum
        for n in range(2,6):
            edges,faces,B,C=cube(n)
            self.assertTrue(np.array_equal(B@C,0*B@C))
            self.assertEqual(len(edges),n*(1<<(n-1)))
            self.assertEqual(len(faces),comb(n,2)*(1<<(n-2)))
            eigen=np.linalg.eigvalsh(B.T@B+C@C.T)
            for value,count in theoretical_spectrum(n):
                self.assertEqual(int(np.count_nonzero(np.isclose(eigen,value,atol=1e-8))),count)
            self.assertAlmostEqual(float(eigen.min()),2,delta=1e-9)

    def test_sharp_global_bound(self):
        from research.algernon.belief_geometry.hodge import cube,sharp_example,certificate
        for n in range(2,6):
            g=sharp_example(n)
            z=certificate(n,g)
            self.assertTrue(z['holds'])
            self.assertGreater(z['gradient_residual_l2'],.5)
            self.assertAlmostEqual(z['gradient_residual_l2'],z['sharp_upper_bound'],delta=1e-8)

    def test_random_nonflat_fields(self):
        from research.algernon.belief_geometry.hodge import cube,certificate
        rng=np.random.default_rng(20261104)
        for n in range(2,6):
            edges,faces,B,C=cube(n)
            for _ in range(30):
                g=rng.normal(size=len(edges))
                z=certificate(n,g)
                self.assertTrue(z['holds'])

    def test_flat_gradients(self):
        from research.algernon.belief_geometry.hodge import cube,certificate
        for n in range(2,6):
            edges,faces,B,C=cube(n)
            phi=np.arange(B.shape[0],dtype=float)
            z=certificate(n,B.T@phi)
            self.assertAlmostEqual(z['face_curl_l2'],0,delta=1e-9)
            self.assertAlmostEqual(z['gradient_residual_l2'],0,delta=1e-9)

    def test_invalid(self):
        from research.algernon.belief_geometry.hodge import certificate,cube
        with self.assertRaises(ValueError):cube(1)
        with self.assertRaises(ValueError):certificate(3,[1,2])


if __name__=='__main__':
    unittest.main()
