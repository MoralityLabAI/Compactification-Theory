import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.special import lambertw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from branes import ChargedBrane
from radion import RadionEFT
from gw import GoldbergerWiseProbe
from helium3 import hamiltonian, periodic_derivative, periodic_velocity, summary

class TestFiveDimensionalBrane(unittest.TestCase):
    def setUp(self):
        self.b=ChargedBrane()

    def test_shift_charge_and_dilution(self):
        ns=np.linspace(0,5,37)
        rho,p,s,theta_dot,q,cs=self.b.fluid(ns)
        self.assertLess(np.max(np.abs(q/self.b.Q-1)),2e-14)
        self.assertTrue(np.allclose(s/self.b.s0,np.exp(-4*ns)))
        self.assertTrue(np.allclose(cs,1/3))
        self.assertGreater(np.min(rho+p),0)

    def test_friedmann_and_raychaudhuri(self):
        ns=np.linspace(0,4,50)
        rho,p,*_=self.b.fluid(ns)
        H=self.b.hubble(ns)
        self.assertLess(np.max(abs(H**2 - self.b.k**2*((1+rho/self.b.sigma)**2-1))),2e-15)
        expected=-3*self.b.k**2/self.b.sigma*(1+rho/self.b.sigma)*(rho+p)
        self.assertLess(np.max(abs(self.b.hubble_dot(ns)-expected)),2e-15)
        delta=(H**2-rho/(3*self.b.M4squared))
        self.assertTrue(np.all(delta>=-3e-15))
        self.assertLess(np.max(abs(delta-self.b.k**2*(rho/self.b.sigma)**2)),2e-15)

    def test_all_bulk_einstein_components(self):
        residual=self.b.einstein_bulk_residuals(np.linspace(0,3,21),np.linspace(0,1.5,71))
        self.assertLess(max(residual.values()),1e-10, str(residual))

    def test_uv_israel_junctions(self):
        result=self.b.junction(np.linspace(0,3,29),1.5)
        self.assertLess(result['uv_spatial_residual_max'],1e-12)
        self.assertLess(result['uv_temporal_residual_max'],1e-12)

    def test_second_brane_matter_constraint(self):
        result=self.b.junction(np.array([0.0,1.0,3.0]),1.5)
        self.assertTrue(np.all(result['ir_required_density']<0))
        self.assertGreater(result['ir_tuned_vacuum_mismatch'],2.0)
        self.assertAlmostEqual(result['ir_required_density'][0],-2.6779117431913755,places=9)

    def test_vacuum_limit_and_gaussian_patch(self):
        tiny=ChargedBrane(V0=0,s0=1e-12)
        yy=np.linspace(0,1.5,31)
        bulk=tiny.bulk(np.array([0.0]),yy)
        self.assertLess(np.max(abs(bulk['D'][0]-np.exp(-yy))),1e-11)
        self.assertLess(np.max(abs(bulk['lapse'][0]-np.exp(-yy))),1e-11)
        with self.assertRaises(ValueError):
            self.b.bulk(np.array([0.0]),np.array([4.0]))

    def test_ir_local_conservation(self):
        ns=np.linspace(0,3,53); L=1.5
        junction=self.b.junction(ns,L)
        bulk=self.b.bulk(ns,np.array([L]))
        D=bulk['D'][:,0]; lapse=bulk['lapse'][:,0]
        rho,p,s,*_=self.b.fluid(ns)
        # Analytic N-derivative of rho_IR = -rho exp(kL)/D.
        drhoIR_dN=12*s*np.exp(self.b.k*L)/D*(1+rho*np.sinh(self.b.k*L)/(self.b.sigma*D))
        residual=drhoIR_dN+3*lapse/D*(junction['ir_required_density']+junction['ir_required_pressure'])
        self.assertLess(np.max(abs(residual)),1e-11)

    def test_time_history(self):
        h=self.b.history(Nmax=3,n=101)
        self.assertTrue(np.all(np.diff(h['t'])>0))
        self.assertLess(h['continuity_max_abs'],1e-14)
        self.assertLess(h['charge_relative_drift'],1e-14)

class TestRadionEffectiveTheory(unittest.TestCase):
    def setUp(self):
        self.r=RadionEFT()

    def test_conformal_quartic_cancellation(self):
        for r in (-1,0,2):
            A=np.exp(self.r.beta*r/self.r.M4)
            for X in (0.001,0.5,12):
                alpha=1.7
                left=A**4*(-self.r.V0+alpha*(X/A**2)**2)
                right=-self.r.V0*A**4+alpha*X**2
                self.assertAlmostEqual(left,right,places=11)

    def test_lambert_radion_minimum(self):
        rmin=self.r.equilibrium()
        self.assertLess(abs(self.r.Veff_derivative(rmin)),1e-13)
        self.assertGreater(self.r.Veff_second(rmin),0)
        self.assertLess(rmin,0)
        self.assertGreater(self.r.Veff(rmin),0)
        self.assertLess(self.r.Veff(rmin),self.r.Veff(0))

    def test_positive_vacuum_sensitivity(self):
        # Envelope theorem: dVeff(r_min)/dV0 = exp(4 beta r_min/M4) > 0.
        eps=1e-6
        low=RadionEFT(V0=self.r.V0-eps)
        high=RadionEFT(V0=self.r.V0+eps)
        deriv=(high.Veff(high.equilibrium())-low.Veff(low.equilibrium()))/(2*eps)
        predicted=np.exp(4*self.r.beta*self.r.equilibrium()/self.r.M4)
        self.assertLess(abs(deriv-predicted),1e-10)
        self.assertGreater(predicted,0)

    def test_frw_continuity_and_attractor(self):
        result=self.r.evolve(Nmax=8,steps=1601)
        self.assertLess(result['friedmann_residual_max'],1e-13)
        self.assertLess(result['continuity_residual_max'],1e-13)
        self.assertLess(result['raychaudhuri_fd_max_abs'],2e-5)
        self.assertLess(result['radion_eom_fd_max_abs'],2e-5)
        self.assertLess(abs(result['r'][-1]-result['equilibrium_r']),2e-4)
        self.assertLess(abs(result['H'][-1]-result['asymptotic_H_expected']),1e-4)
        self.assertTrue(np.all(result['effective_radion_mass2']>0))
        self.assertTrue(np.all(result['w_total']>=-1-1e-12))

    def test_conformal_fluid_has_radiation_sound_speed(self):
        result=self.r.evolve(Nmax=3,steps=71)
        N=result['N']
        ratio=result['fluid_kinetic_density']/result['fluid_kinetic_density'][0]
        self.assertTrue(np.allclose(ratio,np.exp(-4*N)))
        self.assertTrue(np.all(result['vacuum']>0))

class TestGoldbergerWiseProbe(unittest.TestCase):
    def test_warped_stabilization_at_electroweak_and_dark_energy_scales(self):
        for target in (35,69):
            gw=GoldbergerWiseProbe(z_star=target,epsilon=0.1)
            self.assertAlmostEqual(gw.stabilized_separation(),target,places=12)
            self.assertAlmostEqual(gw.normalized_potential(target),0,places=16)
            self.assertGreater(gw.normalized_potential(target-0.1),0)
            self.assertGreater(gw.normalized_potential(target+0.1),0)
            self.assertGreater(gw.curvature_at_minimum(),0)

class TestHeliumThreeWeylTexture(unittest.TestCase):
    def test_discrete_derivative_is_skew_hermitian(self):
        D=periodic_derivative(41,2*np.pi)
        self.assertTrue(np.allclose(D+D.T,0,atol=1e-14))

    def test_uniform_weyl_cone_on_lattice(self):
        n=65;L=2*np.pi;px=0.2;pz=0.3;cpar=1.5;v0=0.8
        H=hamiltonian(np.full(n,v0),L,p_x=px,p_z=pz,c_parallel=cpar)
        y=L*np.arange(n)/n
        theta=np.exp(2j*np.pi*y/L)/np.sqrt(n)
        D=periodic_derivative(n,L)
        discrete_p=np.sin(2*np.pi/n)/(L/n)
        e2=(v0*discrete_p)**2+(px*v0)**2+(pz*cpar)**2
        # Acting twice on either pseudospin times a momentum eigenstate.
        spin=np.array([1,0],complex)
        psi=np.kron(spin,theta)
        self.assertLess(np.linalg.norm((H@H)@psi-e2*psi),1e-12)

    def test_texture_hermiticity_particle_hole_and_shift(self):
        result=summary(n=65,depth=1.0)
        self.assertLess(result['hermiticity_residual'],1e-13)
        self.assertLess(result['particle_hole_spectrum_residual'],1e-12)
        # mass/longitudinal term fixes exact minimum; next excited mode shifts
        self.assertAlmostEqual(result['gap_min'],0.45,places=10)
        self.assertLess(result['lowest_positive_eigenvalues'][1],
                        0.85*result['uniform_lowest_positive_eigenvalues'][1])

    def test_flat_texture_control(self):
        y,v=periodic_velocity(n=33,depth=0)
        self.assertTrue(np.allclose(v,1))
        result=summary(n=33,depth=0)
        self.assertTrue(np.allclose(result['lowest_positive_eigenvalues'],
                                    result['uniform_lowest_positive_eigenvalues']))

if __name__=='__main__':
    unittest.main()
