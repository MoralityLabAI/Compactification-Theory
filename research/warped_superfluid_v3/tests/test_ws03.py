"""Independent tests for WS-03 5D Einstein, charged UV phase, two junctions."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ws03 import System
from probes import probe_spectrum

class WS03Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=System(s0=.001)

    def test_01_exact_GW_superpotential_bulk(self):
        m=self.model;z=np.linspace(0,m.proper_L,111)
        c=m.c_background(z);cp=m.u*c;w_p=m.W(c)/(3*m.M3)
        w_pp=cp**2/(3*m.M3)
        self.assertLess(max(abs(w_pp-cp**2/(3*m.M3))),1e-13)
        self.assertLess(max(abs(6*m.M3*w_p**2-.5*cp**2+m.V(c))),1e-13)
        self.assertLess(max(abs(m.u**2*c-4*w_p*cp-m.dV(c))),1e-13)

    def test_02_background_two_brane_Israel(self):
        m=System(s0=0)
        yL,evaluate=m.background()
        grid=np.array([0.,yL]);A,Ap,c,cp=evaluate(grid)
        bc=m.boundary_slopes(A,A,c)
        self.assertLess(max(abs(Ap-bc['A'])),1e-14)
        self.assertLess(max(abs(Ap-bc['B'])),1e-14)
        self.assertLess(max(abs(cp-bc['chi'])),1e-14)
        self.assertGreater(yL,m.proper_L)

    def test_03_positive_charge_matter(self):
        m=self.model
        self.assertEqual(tuple(float(v) for v in m.fluid(0.)),(.003,.001))
        N=np.linspace(-.2,.5,21)
        rho,p=m.fluid(N)
        self.assertLess(max(abs(rho/(3*p)-1)),1e-14)
        self.assertLess(max(abs(p*np.exp(4*N)/m.s0-1)),1e-14)
        self.assertGreater(min(rho+p),0)

    def test_04_initial_hamiltonian_and_momentum(self):
        a=self.model.initial_slice(n=97)
        self.assertGreater(a['q'],0)
        self.assertLess(a['hamiltonian_max_analytic'],3e-12)
        self.assertLess(a['momentum_max_analytic'],3e-12)

    def test_05_all_six_junctions_on_charged_slice(self):
        a=self.model.initial_slice(n=97)
        self.assertLess(max(abs(v) for v in a['jump_errors'].values()),1e-10)
        self.assertGreater(a['radion_length_initial'],.99)

    def test_06_correct_uv_Hubble_and_momentum_profile(self):
        a=self.model.initial_slice(n=97)
        H=np.sqrt(a['q'])
        self.assertLess(max(abs(a['Adot']-H*np.exp(a['B']-a['A']))),1e-14)
        self.assertLess(abs(a['Adot'][0]-H),1e-14)

    def test_07_low_energy_4d_gravitational_reduction(self):
        m=self.model
        z=np.linspace(0,m.proper_L,3001)
        M4sq=2*m.M3*np.trapezoid(np.exp(-2*m.warp(z)),z)
        q=m.initial_slice(n=97)['q']
        measured=q/(3*m.s0)
        expected=1/(3*M4sq)
        self.assertLess(abs(measured/expected-1),.001)

    def test_08_conformal_coordinate_proper_length(self):
        m=System(s0=0)
        yL,bps=m.background()
        y=np.linspace(0,yL,801)
        A,Ap,c,cp=bps(y)
        self.assertLess(abs(np.trapezoid(np.exp(A),y)-m.proper_L),1e-5)

    def test_09_fourth_order_boundary_derivative(self):
        h=.02; y=np.arange(21)*h;v=1+2*y+3*y**2-4*y**3+5*y**4
        left=2.;right=2+6*y[-1]-12*y[-1]**2+20*y[-1]**3
        self.assertLess(System.one_sided_junction_error(v,h,left,right),1e-9)

    def test_10_evolution_constraint_refinement(self):
        m=self.model
        coarse=m.evolve(n=65,time_end=.24,samples=7)
        fine=m.evolve(n=129,time_end=.24,samples=7)
        self.assertGreater(coarse['max_00_constraint_interior']/fine['max_00_constraint_interior'],2.5)
        self.assertGreater(coarse['max_0y_constraint_interior']/fine['max_0y_constraint_interior'],2.5)

    def test_11_israel_numeric_refinement(self):
        m=self.model
        coarse=m.evolve(n=65,time_end=.15,samples=5)
        fine=m.evolve(n=129,time_end=.15,samples=5)
        err1=max(t['junction_one_sided_max'] for t in coarse['diagnostics'])
        err2=max(t['junction_one_sided_max'] for t in fine['diagnostics'])
        self.assertLess(err2,err1/3)

    def test_12_zero_charge_static_mesh_limit(self):
        m=System(s0=0)
        c=m.evolve(n=65,time_end=.24,samples=7)
        f=m.evolve(n=129,time_end=.24,samples=7)
        self.assertGreater(c['max_00_constraint_interior']/f['max_00_constraint_interior'],2.0)
        self.assertLess(abs(f['diagnostics'][-1]['H_UV']),.002)

    def test_13_conserved_phase_charge_in_5d_evolution(self):
        q=self.model.evolve(n=65,time_end=.24,samples=9)
        self.assertLess(max(d['charge_relative_error'] for d in q['diagnostics']),1e-12)
        self.assertLess(q['diagnostics'][-1]['rho_UV'],q['diagnostics'][0]['rho_UV'])

    def test_14_perturbed_initial_slice_keeps_all_constraints(self):
        for bump in [-.001,.001]:
            a=self.model.initial_slice(n=65,scalar_bump=bump)
            self.assertLess(a['hamiltonian_max_analytic'],3e-12)
            self.assertLess(a['momentum_max_analytic'],3e-12)
            self.assertLess(max(abs(v) for v in a['jump_errors'].values()),1e-10)

    def test_15_coupled_radion_linear_response_convergence(self):
        runs={}
        for n in [65,129]:
            r1=self.model.evolve(n=n,time_end=.4,samples=9,scalar_bump=-.001)
            r2=self.model.evolve(n=n,time_end=.4,samples=9,scalar_bump=.001)
            d=(r2['diagnostics'][-1]['radion_proper_length']-r1['diagnostics'][-1]['radion_proper_length'])/.002
            runs[n]=d
        self.assertGreater(runs[129],0)
        self.assertLess(abs(runs[129]-runs[65]),2e-5)

    def test_16_positive_fixed_metric_probe_spectrum(self):
        vals=probe_spectrum(self.model,nodes=129,count=4)
        self.assertGreater(vals[0],0)
        self.assertTrue(np.all(np.diff(vals)>0))

    def test_17_probe_spectral_mesh_convergence(self):
        m=self.model
        a=probe_spectrum(m,nodes=65,count=1)[0]
        b=probe_spectrum(m,nodes=129,count=1)[0]
        c=probe_spectrum(m,nodes=257,count=1)[0]
        self.assertGreater(abs(a-b)/abs(b-c),3.5)

    def test_18_probe_stiffness_sensitivity(self):
        a=probe_spectrum(System(stiffness_uv=2,stiffness_ir=2),nodes=85,count=1)[0]
        b=probe_spectrum(System(stiffness_uv=8,stiffness_ir=8),nodes=85,count=1)[0]
        self.assertGreater(b,a)

    def test_19_BF_local_mass_above_AdS_bound(self):
        m=self.model
        self.assertGreater(m.d2V(m.chi_uv),-4*m.k**2)

    def test_20_vacuum_tension_sensitivity_no_self_tuning(self):
        m=System(s0=0,vacuum_detune_uv=.001)
        initial=m.initial_slice(n=97)
        z=np.linspace(0,m.proper_L,3001)
        M4sq=2*m.M3*np.trapezoid(np.exp(-2*m.warp(z)),z)
        predicted=1/(3*M4sq)
        measured=initial['q']/m.vacuum_detune_uv
        self.assertGreater(measured,0)
        self.assertLess(abs(measured/predicted-1),.001)
        self.assertLess(max(abs(v) for v in initial['jump_errors'].values()),1e-10)

if __name__=='__main__': unittest.main()
