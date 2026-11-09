"""Exact-interval infinite-resolvent certificate verification and smoke tests."""
import copy
import json
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import hamiltonian
from infinite_resolvent import (
    machin_pi_interval, inverse_pi_interval, h_pi_half, resolvent_witness,
    check_witness, analytic_full_projection_bound, verify_payload,
    RESIDUAL_SCALE, PI_SCALE, VECTOR_SCALE)

HERE = Path(__file__).resolve().parents[1]


class InfiniteResolventTests(unittest.TestCase):
    def test_exact_pi_interval_via_machin(self):
        lo, hi=machin_pi_interval()
        self.assertGreater(lo, 3*PI_SCALE)
        self.assertLess(hi, 4*PI_SCALE)
        self.assertLessEqual(hi-lo, 3)
        il, ih=inverse_pi_interval()
        self.assertLessEqual(il*lo, PI_SCALE**2)
        self.assertGreaterEqual(ih*hi, PI_SCALE**2)

    def test_pi_half_formula_matches_original(self):
        for m in (2,6,10,14,30):
            np.testing.assert_allclose(h_pi_half(m),hamiltonian(np.pi/2,m),atol=3e-14)

    def test_pi_half_is_hermitian(self):
        for m in (2,6,10,14,30):
            h=h_pi_half(m)
            np.testing.assert_allclose(h,h.conj().T,atol=1e-14)

    def test_bounded_perturbation_finite_sections(self):
        for m in (2,6,10,14,30,90):
            h=h_pi_half(m)
            n=np.arange(-m//2,m//2)
            v=h-np.diag(np.pi*(n+.5))
            self.assertLessEqual(np.linalg.norm(v,2),np.pi+1e-12)

    def test_machin_yield_is_independent_of_witness_solver(self):
        u=resolvent_witness(6)
        self.assertEqual(len(u),6)
        self.assertTrue(all(isinstance(v,int) for row in u for pair in row for v in pair))
        c=check_witness(6,u,cutoff=128)
        self.assertGreater(c['error_upper'],0)
        self.assertLess(c['error_upper'],0.07)

    def test_bound_decreases_with_truncation(self):
        bounds=[analytic_full_projection_bound(m,np.pi/2,2j)['certified_upper']
                for m in (2,6,10,14,22,30)]
        self.assertTrue(all(a>=b for a,b in zip(bounds,bounds[1:])))
        self.assertLess(bounds[-1],bounds[0])

    def test_infinite_residual_bounds_finite_reference(self):
        # A finite reference is only a sanity test; the proof uses an infinite tail.
        big_m=80
        h=h_pi_half(big_m)
        f=np.zeros((big_m,2),complex)
        f[big_m//2-1,0]=1
        f[big_m//2,1]=1
        u_big=np.linalg.solve(h-2j*np.eye(big_m),f)
        for m in (2,6,10,14):
            witness=resolvent_witness(m)
            c=check_witness(m,witness,cutoff=128)
            u=np.asarray(witness,dtype=float)
            u=(u[:,:,0]+1j*u[:,:,1])/VECTOR_SCALE
            embed=np.zeros_like(u_big)
            embed[(big_m-m)//2:(big_m+m)//2,:]=u
            observed=np.linalg.norm(u_big-embed,2)
            self.assertLess(observed,c['error_upper'])

    def test_mutated_receipt_is_detected(self):
        case={'schema':'qft-lattice-infinite-resolvent-v1',
              'mode':'k=pi/2, z=iy, core n=-1,0',
              'cases':[{'M':6,'witness_vector_integer_pairs':resolvent_witness(6)}]}
        case['cases'][0]['certificate']=check_witness(6,case['cases'][0]['witness_vector_integer_pairs'],cutoff=128)
        self.assertTrue(verify_payload(case))
        tampered=copy.deepcopy(case)
        tampered['cases'][0]['witness_vector_integer_pairs'][0][0][0]+=12345
        with self.assertRaises(AssertionError): verify_payload(tampered)

    def test_registered_receipts_validate(self):
        p=HERE/'results'/'infinite_resolvent_certificate.json'
        payload=json.loads(p.read_text())
        self.assertTrue(verify_payload(payload))
        bounds=[x['certificate']['error_upper'] for x in payload['cases']]
        self.assertTrue(all(a>b for a,b in zip(bounds,bounds[1:])))
        self.assertLess(bounds[-1],0.004)
        self.assertEqual(payload['cases'][0]['certificate']['resolution_scale'],str(RESIDUAL_SCALE))

if __name__=='__main__': unittest.main()
