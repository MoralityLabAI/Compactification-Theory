"""WS-04: twenty-two regression gates for the radion and charged bulk."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT.parent/'warped_superfluid_v3'),str(ROOT)]
from ws03 import System
from probes import probe_spectrum
from radion import RadionModel
from bulk_complex import BulkChargedSystem


class GaugeInvariantRadionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=RadionModel()

    def test_01_BPS_background_is_monotone(self):
        z=np.linspace(0,1,101)
        a,H,cy,ratio,Hp=self.r.background(z)
        self.assertTrue(np.all(H<0))
        self.assertTrue(np.all(cy>0))
        self.assertTrue(np.all(Hp<=H*H))

    def test_02_correct_boundary_signs(self):
        g0,gL=self.r.gvalues()
        self.assertGreater(g0,0)
        self.assertLess(gL,0)
        a0=self.r.background(0.)[0];aL=self.r.background(1.)[0]
        self.assertAlmostEqual(g0,a0*4.)
        self.assertAlmostEqual(gL,-aL*4.)

    def test_03_gauge_invariants_under_scalar_diffeomorphism(self):
        z=np.linspace(0,1,127)
        a,H,cy,_,_=self.r.background(z)
        S=.02*np.sin(z);xi=.03*np.sin(3*z)
        xiy=.09*np.cos(3*z)*a;Sy=.02*np.cos(z)*a
        psi=np.cos(z);delta=np.sin(z);phi=.1*z
        invariant_psi=psi+H*S
        invariant_Delta=delta+cy*S
        invariant_Phi=phi+H*S+Sy
        psi2=psi-H*xi;delta2=delta-cy*xi;phi2=phi-H*xi-xiy
        self.assertTrue(np.allclose(psi2+H*(S+xi),invariant_psi,atol=1e-13))
        self.assertTrue(np.allclose(delta2+cy*(S+xi),invariant_Delta,atol=1e-13))
        self.assertTrue(np.allclose(phi2+H*(S+xi)+Sy+xiy,invariant_Phi,atol=1e-13))

    def test_04_lowest_radion_eigenmode(self):
        m2=brentq(self.r.boundary_residual,0,.02,xtol=2e-13)
        self.assertAlmostEqual(m2,.0011806541054,delta=1e-8)
        sol=self.r.integrate(m2)
        self.assertLess(abs(sol['residual']),2e-10)
        self.assertAlmostEqual(sol['g_uv']*sol['pi'][0]+m2*sol['psi'][0],0,delta=1e-12)

    def test_05_reconstruct_coupled_scalar_metric(self):
        m2=brentq(self.r.boundary_residual,0,.02)
        r=self.r.gauge_invariant_reconstruction(m2)
        cy=self.r.background(r['z'])[2]
        self.assertLess(np.max(np.abs(r['Phi']+2*r['psi'])),1e-14)
        self.assertLess(np.max(np.abs(cy*r['Delta']+3*self.r.integrate(m2)['pi'])),1e-12)

    def test_06_boundary_residual_and_first_3_modes(self):
        out=self.r.modes(count=3,m2max=15,samples=100)
        roots=out['positive_m2']
        self.assertEqual(len(roots),3)
        self.assertTrue(np.all(np.diff(roots)>0))
        self.assertAlmostEqual(roots[1],2.4311097342,delta=3e-8)
        self.assertAlmostEqual(roots[2],8.1326717809,delta=3e-8)
        self.assertEqual(out['negative_m2_scan'],[])

    def test_07_stability_criterion_and_tachyon_counterexample(self):
        good=self.r.gvalues()
        self.assertGreater(good[0],0)
        self.assertLess(good[1],0)
        bad=RadionModel(stiffness_uv=8,stiffness_ir=-3)
        self.assertGreater(bad.gvalues()[1],0)
        tachyon=brentq(bad.boundary_residual,-.1,-.00001,xtol=1e-12)
        self.assertAlmostEqual(tachyon,-.00162559616,delta=1e-8)

    def test_08_brane_stiffness_controls_radion_mass(self):
        low=RadionModel(stiffness_uv=1,stiffness_ir=1)
        mlo=brentq(low.boundary_residual,0,.01)
        mhi=brentq(self.r.boundary_residual,0,.01)
        self.assertGreater(mhi,mlo)

    def test_09_probe_misses_light_radion(self):
        rad=brentq(self.r.boundary_residual,0,.01)
        probe=probe_spectrum(System(s0=0),nodes=129,count=1)[0]
        self.assertGreater(probe/rad,100)

    def test_10_parameters_and_massless_boundary_rejected(self):
        with self.assertRaises(ValueError): RadionModel(u=0)
        with self.assertRaises(ValueError): RadionModel(stiffness_ir=0)


class ChargedBulkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=BulkChargedSystem()

    def test_11_U1_symmetric_bulk_potential(self):
        m=self.m;r=.04
        for phase in np.linspace(0,2*np.pi,9):
            self.assertAlmostEqual(m.bulk_potential(r*np.cos(phase),r*np.sin(phase)),m.bulk_potential(r,0.))

    def test_12_5D_charged_initial_constraints(self):
        initial=self.m.initial_slice(n=97)
        self.assertGreater(initial['H2_uv'],0)
        self.assertLess(initial['constraint_00_exact'],2e-13)
        self.assertLess(initial['constraint_0y_exact'],2e-13)

    def test_13_all_ten_junctions(self):
        initial=self.m.initial_slice(n=97)
        self.assertLess(max(initial['junction_exact'].values()),2e-10)
        self.assertLess(abs(initial['x'][0]-self.m.bulk_amp),1e-14)
        self.assertLess(abs(initial['x'][-1]-self.m.bulk_amp),1e-14)

    def test_14_nonzero_global_charge_and_closed_endpoints(self):
        initial=self.m.initial_slice(n=97)
        fields=np.stack([initial[key] for key in ('A','B','chi','x','y','At','Bt','ct','xt','yt')])
        charge=self.m.global_charge(fields,initial['grid'])
        self.assertGreater(charge,1e-6)
        self.assertLess(max(abs(a) for a in self.m.boundary_flux(fields)),1e-14)

    def test_15_negative_frequency_reverses_charge(self):
        m=BulkChargedSystem(bulk_omega=-self.m.bulk_omega)
        i1=self.m.initial_slice(n=65)
        i2=m.initial_slice(n=65)
        self.assertAlmostEqual(i1['H2_uv'],i2['H2_uv'],delta=1e-12)
        f1=np.stack([i1[k] for k in ('A','B','chi','x','y','At','Bt','ct','xt','yt')])
        f2=np.stack([i2[k] for k in ('A','B','chi','x','y','At','Bt','ct','xt','yt')])
        self.assertAlmostEqual(self.m.global_charge(f1,i1['grid']),-m.global_charge(f2,i2['grid']),delta=1e-12)

    def test_16_zero_charge_and_zero_amplitude_static_limit(self):
        m=BulkChargedSystem(bulk_amp=0,bulk_omega=0)
        i=m.initial_slice(n=65)
        self.assertEqual(i['H2_uv'],0)
        self.assertAlmostEqual(i['constraint_00_exact'],0,delta=1e-13)
        self.assertLess(max(i['junction_exact'].values()),1e-7)

    def test_17_phase_flux_exact_for_rotated_boundary_values(self):
        m=self.m;i=m.initial_slice(n=65)
        v=np.stack([i[k] for k in ('A','B','chi','x','y','At','Bt','ct','xt','yt')])
        ang=np.linspace(0,1,len(i['grid']))
        x=v[3].copy();y=v[4].copy()
        v[3]=np.cos(ang)*x-np.sin(ang)*y
        v[4]=np.sin(ang)*x+np.cos(ang)*y
        self.assertLess(max(abs(a) for a in m.boundary_flux(v)),1e-14)

    def test_18_bulk_constraint_refines(self):
        low=self.m.evolve(n=65,time_end=.18,samples=7)
        high=self.m.evolve(n=129,time_end=.18,samples=7)
        self.assertGreater(low['max_constraint00']/high['max_constraint00'],2.8)
        self.assertGreater(low['max_constraint0y']/high['max_constraint0y'],2.8)

    def test_19_global_charge_drift_refines(self):
        low=self.m.evolve(n=65,time_end=.2,samples=9)
        high=self.m.evolve(n=129,time_end=.2,samples=9)
        self.assertGreater(low['max_charge_drift']/high['max_charge_drift'],3.)
        self.assertLess(high['max_charge_drift'],1e-5)

    def test_20_both_ends_zero_phase_flux_after_evolution(self):
        high=self.m.evolve(n=97,time_end=.18,samples=7)
        self.assertLess(max(d['normal_current_flux'] for d in high['diagnostics']),1e-13)

    def test_21_junction_derivative_refines(self):
        low=self.m.evolve(n=65,time_end=.18,samples=7)
        high=self.m.evolve(n=129,time_end=.18,samples=7)
        e0=max(d['junction_error'] for d in low['diagnostics'])
        e1=max(d['junction_error'] for d in high['diagnostics'])
        self.assertGreater(e0/e1,3)

    def test_22_charged_bulk_requires_time_dependent_geometry(self):
        m=self.m;i=m.initial_slice(n=97)
        F=i['x'];gtt=-np.exp(-2*i['B'])
        anisotropy=gtt*m.bulk_omega**2*F**2
        self.assertLess(float(np.max(anisotropy)),0)
        self.assertGreater(float(np.max(np.abs(anisotropy))),1e-5)


if __name__=='__main__':
    unittest.main()
