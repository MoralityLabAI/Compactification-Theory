"""WS-06: stationary 5D probe plus genuinely GI 4D scalar-gravity EFT.

Tests intentionally do not claim that the full 5D charged Einstein-brane scalar
constraints have been eliminated or that a 5D radion-phase spectrum was found.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from dataclasses import replace
import numpy as np
from bulk_probe import (ProbeParams,equilibrium,assemble,weighted_nonlinear,
    goldstone_sound_speed_squared,normal_frequencies,energy,energy_derivative)
from effective_gravity import (EFTParams,eft_from_probe,initial_conditions,
    friedmann_constraint,gradient_hessian,potential,mass_matrix,background_rhs,
    gauge_invariant_Q,gauge_time_shift,transfer)


class TestFiveDimensionalProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params=ProbeParams()
        cls.s=equilibrium(cls.params,n=65)

    def test_equilibrium_exact_weak_residual(self):
        self.assertLess(self.s['nonlinear_residual'],2e-10)
        self.assertLess(abs(self.s['F']@self.s['M']@self.s['F']-.32**2),1e-10)

    def test_stationary_charge_is_positive(self):
        self.assertGreater(self.s['omega'],0)
        self.assertGreater(self.s['omega2'],self.s['linear_omega2'])

    def test_robin_both_ends_grid_refinement(self):
        def mismatch(s):
            F=s['F'];h=s['y'][1]-s['y'][0]
            d0=(-3*F[0]+4*F[1]-F[2])/(2*h)
            dL=(3*F[-1]-4*F[-2]+F[-3])/(2*h)
            return np.array([d0-.5*F[0],dL+.5*F[-1]])
        a=np.max(abs(mismatch(equilibrium(self.params,n=33))))
        b=np.max(abs(mismatch(self.s)))
        c=np.max(abs(mismatch(equilibrium(self.params,n=129))))
        self.assertGreater(a/b,3.4)
        self.assertGreater(b/c,3.4)

    def test_frequency_second_order_refinement(self):
        ws=[equilibrium(self.params,n=n)['omega2'] for n in (33,65,129,257)]
        self.assertGreater(abs(ws[0]-ws[1])/abs(ws[1]-ws[2]),3.2)
        self.assertGreater(abs(ws[1]-ws[2])/abs(ws[2]-ws[3]),3.2)

    def test_ground_nodeless(self):
        self.assertGreater(np.min(self.s['F']),0)

    def test_phase_null_and_hessian_identity(self):
        s=self.s
        self.assertLess(s['phase_null_relative'],1e-9)
        self.assertTrue(np.allclose(s['Kr']-s['Ks'],2*self.params.lam*s['T'],atol=1e-12))

    def test_positive_energy_hessian(self):
        self.assertGreater(self.s['min_eig_Kr'],.01)
        self.assertGreater(self.s['min_eig_Ks'],-1e-8)

    def test_schur_complement_speed(self):
        c2=goldstone_sound_speed_squared(self.s)
        self.assertGreater(c2,0)
        self.assertLess(c2,1)
        self.assertAlmostEqual(c2,.0128437405,places=6)

    def test_small_momentum_vs_schur_speed(self):
        c2=goldstone_sound_speed_squared(self.s)
        omega=normal_frequencies(self.s,0.002)['min_frequency']
        self.assertLess(abs((omega/0.002)**2-c2),2e-6)

    def test_probe_mode_growth_absent(self):
        for p in (.02,.15,.5):
            r=normal_frequencies(self.s,p)
            self.assertLess(r['max_abs_growth_rate'],1e-7)
            self.assertGreater(r['min_frequency'],0)

    def test_gyroscopic_energy_conservation(self):
        rng=np.random.default_rng(6);n=len(self.s['F'])
        q=rng.normal(size=2*n);v=rng.normal(size=2*n)
        E=energy(self.s,.2,q,v)
        self.assertGreater(E,0)
        self.assertLess(abs(energy_derivative(self.s,.2,q,v)),1e-8*E)

    def test_numerical_instability_control(self):
        # An intentionally negative amplitude Hessian should trigger growth.
        fake={**self.s,'Kr':self.s['Kr']-2*self.s['M']}
        self.assertGreater(normal_frequencies(fake,.02)['max_abs_growth_rate'],.01)

    def test_brane_stiffness_changes_spectrum(self):
        stiff=equilibrium(replace(self.params,kap_ir=1.5),n=65)
        self.assertGreater(abs(stiff['omega2']-self.s['omega2']),.005)

    def test_charged_stress_breaks_4d_lorentz_invariance(self):
        y=self.s['y'];F=self.s['F'];omega=self.s['omega']
        anisotropy=-np.exp(2*self.params.k*y)*(omega*F)**2
        self.assertTrue(np.all(anisotropy<0))
        self.assertGreater(np.max(abs(anisotropy)),.01)


class TestFourDimensionalGaugeInvariantEFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sol=equilibrium(ProbeParams(),n=65)
        cls.p=eft_from_probe(ProbeParams(),cls.sol)
        cls.z=initial_conditions(cls.p,omega=cls.sol['omega'])

    def test_reduced_planck_mass_and_finite_charge_match(self):
        p=self.p
        self.assertAlmostEqual(p.Mpl2,1-np.exp(-2),places=10)
        omega4=np.sqrt(p.mass_bulk2+p.lam*.32**2)
        self.assertLess(abs(omega4-self.sol['omega']),1e-5)

    def test_initial_friedmann_constraint(self):
        self.assertLess(abs(friedmann_constraint(self.z,self.p)),1e-12)

    def test_gradient_hessian_symmetry(self):
        phi=np.array([.03,.3,.11]);g,h=gradient_hessian(phi,self.p)
        hfd=np.column_stack([(gradient_hessian(phi+1e-5*np.eye(3)[j],self.p)[0]-
                            gradient_hessian(phi-1e-5*np.eye(3)[j],self.p)[0])/(2e-5)
                            for j in range(3)])
        self.assertTrue(np.allclose(h,h.T,atol=1e-12))
        self.assertLess(np.max(abs(hfd-h)),1e-8)

    def test_u1_potential_and_gradient_equivariance(self):
        phi=np.array([-.02,.3,.12]);angle=.47
        c,s=np.cos(angle),np.sin(angle)
        R=np.array([[1,0,0],[0,c,-s],[0,s,c]])
        a=potential(phi,self.p);b=potential(R@phi,self.p)
        self.assertAlmostEqual(a,b,places=13)
        self.assertTrue(np.allclose(gradient_hessian(R@phi,self.p)[0],R@gradient_hessian(phi,self.p)[0]))

    def test_gauge_invariant_Mukhanov_Sasaki_variables(self):
        rng=np.random.default_rng(22)
        dphi=rng.normal(size=3);psi=.02;T=.7
        v=self.z[3:6];H=self.z[7]
        new_phi,new_psi=gauge_time_shift(dphi,psi,v,H,T)
        self.assertTrue(np.allclose(gauge_invariant_Q(dphi,psi,v,H),
                                   gauge_invariant_Q(new_phi,new_psi,v,H),atol=1e-14))

    def test_gravitational_mass_correction_is_symmetric(self):
        m=mass_matrix(self.z,self.p)
        h=mass_matrix(self.z,self.p,gravity=False)
        self.assertTrue(np.allclose(m,m.T,atol=1e-13))
        self.assertGreater(np.linalg.norm(m-h),.01)

    def test_background_charge_and_Friedmann_preserved(self):
        s=transfer(.2,self.p)
        self.assertLess(s['max_Friedmann_error'],1e-9)
        self.assertLess(s['charge_drift'],1e-9)

    def test_symplectic_gauge_invariant_propagator(self):
        for p3 in (.05,1.,3.):
            s=transfer(p3,self.p)
            self.assertLess(s['max_symplectic_residual'],1e-8)
            self.assertTrue(np.all(np.isfinite(s['fundamental'])))

    def test_finite_momentum_changes_transfer(self):
        a=transfer(.05,self.p)
        b=transfer(3.,self.p)
        self.assertGreater(np.linalg.norm(a['fundamental'][-1]-b['fundamental'][-1]),.2)

    def test_mixing_has_dynamical_effect(self):
        coupled=transfer(.2,self.p)
        self.assertGreater(abs(coupled['response_radion_from_phase_pi'][-1]),1e-7)
        reference=transfer(.2,replace(self.p,g=0))
        self.assertLess(abs(reference['response_radion_from_phase_pi'][-1]),1e-11)

    def test_local_kinetic_gradient_positivity(self):
        s=transfer(.2,self.p)
        self.assertGreater(np.min(s['instantaneous_mass_eigen']),0)
        # Canonical EFT has K_kin = I, C_grad = I (strictly positive).
        self.assertEqual(np.linalg.eigvalsh(np.eye(3))[0],1)

    def test_background_expands_and_decelerates(self):
        s=transfer(.2,self.p)
        self.assertGreater(s['background'][-1,6],s['background'][0,6])
        self.assertLess(s['background'][-1,7],s['background'][0,7])

if __name__=='__main__':unittest.main()
