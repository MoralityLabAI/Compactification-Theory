"""Deterministic CPU smoke/unit gates for free-Dirac replication."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import (hamiltonian, axial_charge, flavors, w_kernel,
                   w_integral, theta, two_flavor_energy, commutator,
                   projection_boundary_identity, resolvent_comparison,
                   correlation_matrix, entropy_from_correlation,
                   b_function, b_quadrature)


class Tests(unittest.TestCase):
    def test_theta_endpoints(self):
        self.assertAlmostEqual(float(theta(0)),np.pi/2)
        self.assertAlmostEqual(float(theta(.5)),np.pi/4)
        self.assertAlmostEqual(float(theta(1)),0)
    def test_theta_integral_matches_printed_w(self):
        for d in range(-10,11):
            self.assertLess(abs(w_integral(d)-w_kernel(d)),1e-11)
    def test_w_4zero(self):
        self.assertLess(abs(w_kernel(4)),1e-25)
    def test_even_only(self):
        with self.assertRaises(ValueError): flavors(3)
    def test_hermitian(self):
        for M in (2,6,10,14):
            for k in (0.,.7,1.4,np.pi):
                H=hamiltonian(k,M)
                self.assertLess(np.linalg.norm(H-H.conj().T),1e-12)
                Q=axial_charge(k,M)
                self.assertLess(np.linalg.norm(Q-Q.conj().T),1e-10)
    def test_two_flavor_analytic_dispersion(self):
        for k in np.linspace(-np.pi,np.pi,23):
            eig=np.linalg.eigvalsh(hamiltonian(k,2))
            np.testing.assert_allclose(eig,[-two_flavor_energy(k),two_flavor_energy(k)],atol=2e-13)
    def test_only_zero_at_k_zero_for_m2(self):
        for k in np.linspace(.05,np.pi,80):
            self.assertGreater(min(abs(np.linalg.eigvalsh(hamiltonian(k,2)))),1e-4)
        self.assertLess(max(abs(np.linalg.eigvalsh(hamiltonian(0,2)))),1e-12)
    def test_mod_four_zero_mode_rule(self):
        for M in (2,6,10,14):
            self.assertLess(np.min(np.abs(np.linalg.eigvalsh(hamiltonian(0.,M)))),1e-10)
        for M in (4,8,12):
            self.assertGreater(np.min(np.abs(np.linalg.eigvalsh(hamiltonian(0.,M)))),1e-6)
    def test_no_extra_zero_on_grid(self):
        for M in (6,10,14):
            for k in np.linspace(.05,np.pi,31):
                self.assertGreater(np.min(np.abs(np.linalg.eigvalsh(hamiltonian(k,M)))),1e-4)
    def test_low_dispersion_converges(self):
        errors=[]
        for M in (2,6,10,14):
            eig=np.linalg.eigvalsh(hamiltonian(.7,M))
            errors.append(max(abs(eig[eig>0][0]-.7),abs(eig[eig<0][-1]+.7)))
        self.assertLess(errors[-1],errors[0])
    def test_axial_low_energy_restoration(self):
        norms=[]
        for M in (2,6,10,14):
            H=hamiltonian(.7,M); ev,U=np.linalg.eigh(H)
            P=U[:,np.argsort(abs(ev))[:2]]
            norms.append(np.linalg.norm(P.conj().T@commutator(H,axial_charge(.7,M))@P))
        self.assertLess(norms[-1],norms[0]/1000)
    def test_entropy_k_tail_converges(self):
        x=entropy_from_correlation(correlation_matrix(64,2,10,1024))
        y=entropy_from_correlation(correlation_matrix(64,2,10,2048))
        self.assertLess(abs(y-x),1e-6)
    def test_vector_number_commutes(self):
        for M in (2,6):
            h=hamiltonian(.7,M)
            self.assertLess(np.linalg.norm(commutator(h,np.eye(M))),1e-12)
    def test_boundary_projection_identity(self):
        for M in (2,6,10,14):
            residual,_,_=projection_boundary_identity(.7,M,32)
            self.assertLess(residual,1e-9)
    def test_resolvent_schur_identity_and_bound(self):
        for M in (2,6,10,14):
            r=resolvent_comparison(.7,M,32)
            self.assertLess(r['schur_residual'],1e-10)
            self.assertLessEqual(r['error'],r['bound']+1e-12)
    def test_fresnel_b_matches_direct_integral(self):
        z=np.array([0.,1.,2.,3.,20.,50.,100.,200.])
        np.testing.assert_allclose(b_function(z),b_quadrature(z),atol=1e-11)
    def test_correlation_physical_and_entropy(self):
        C=correlation_matrix(32,2,6,qmax=256)
        eigen=np.linalg.eigvalsh(C)
        self.assertGreater(eigen[0],-1e-7)
        self.assertLess(eigen[-1],1+1e-7)
        self.assertGreater(entropy_from_correlation(C),0)


if __name__=='__main__': unittest.main()
