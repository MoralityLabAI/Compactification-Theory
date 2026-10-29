"""WS-05 regression gates: homogeneous Goldstone theory + 5D tangent dynamics."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT.parent/'warped_superfluid_v3'),
              str(ROOT.parent/'warped_superfluid_v4'),str(ROOT)]
from bulk_complex import BulkChargedSystem
from goldstone import HomogeneousCondensate
from tangent import (
    base_initial_state,initial_tangent,jvp,constraint_jvp,junction_jvp,
    charge_derivative,geometry_derivative,geometry_observables,
    phase_rotation_tangent,evolve_tangent,finite_difference_response,
)


class GoldstoneControl(unittest.TestCase):
    def setUp(self):
        self.c=HomogeneousCondensate(F=.5)

    def test_01_equilibrium_and_radial_curvature(self):
        c=self.c
        self.assertAlmostEqual(c.equilibrium_frequency**2,c.mass2+c.coupling*c.F**2)
        self.assertAlmostEqual(c.radial_curvature,2*c.coupling*c.F**2)

    def test_02_goldstone_zero_and_gapped_mode(self):
        low,high=self.c.omega_squared_branches(np.array([0.]))
        self.assertAlmostEqual(low[0],0)
        self.assertAlmostEqual(high[0],self.c.gap_squared)

    def test_03_quadratic_determinant(self):
        ps=np.array([0,1e-4,.02,.1,.5,2.])
        low,high=self.c.omega_squared_branches(ps)
        self.assertLess(np.max(abs(self.c.determinant(ps,low))),1e-11)
        self.assertLess(np.max(abs(self.c.determinant(ps,high))),1e-11)
        self.assertTrue(np.all(low>=0))
        self.assertTrue(np.all(high>=low))

    def test_04_hydrodynamic_sound_from_charge_susceptibility(self):
        c=self.c
        omega=c.equilibrium_frequency
        omega2=omega*omega
        susceptibility=(3*omega2-c.mass2)/c.coupling
        n=omega*(omega2-c.mass2)/c.coupling
        cs2=n/(omega*susceptibility)
        self.assertAlmostEqual(cs2,c.goldstone_sound_squared)
        tiny=1e-5
        low,_=c.omega_squared_branches(np.array([tiny]))
        self.assertAlmostEqual(low[0]/tiny**2,cs2,delta=1e-9)

    def test_05_gap_and_no_tachyon_over_grid(self):
        p=np.linspace(0,5,400)
        low,high=self.c.omega_squared_branches(p)
        self.assertTrue(np.all(np.diff(low)>0))
        self.assertTrue(np.all(np.diff(high)>0))
        self.assertTrue(np.all(np.isfinite(low)))

    def test_06_ws04_state_is_not_local_homogeneous_equilibrium(self):
        m=BulkChargedSystem()
        c=HomogeneousCondensate(F=m.bulk_amp,mass2=m.bulk_mass2,coupling=m.bulk_lambda)
        self.assertAlmostEqual(c.equilibrium_frequency,0.707191628909732)
        self.assertLess(c.local_frequency_gap(m.bulk_omega,0),-.3)

    def test_07_invalid_homogeneous_inputs(self):
        with self.assertRaises(ValueError): HomogeneousCondensate(F=0)
        with self.assertRaises(ValueError): HomogeneousCondensate(coupling=-.1)
        with self.assertRaises(ValueError): self.c.omega_squared_branches([-1])


class CoupledLinearResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=BulkChargedSystem()
        cls.grid,cls.state,cls.tangent=initial_tangent(cls.m,n=49)
        cls.dy=cls.grid[1]-cls.grid[0]

    def test_08_jvp_matches_nonlinear_fd(self):
        s=self.state;p=self.tangent;h=1e-5
        direct=jvp(self.m,0,s,p,self.dy)
        fd=(self.m.rhs(0,(s+h*p).ravel(),len(self.grid),self.dy).reshape(s.shape)
           -self.m.rhs(0,(s-h*p).ravel(),len(self.grid),self.dy).reshape(s.shape))/(2*h)
        self.assertLess(np.max(abs(direct-fd)),1e-6)

    def test_09_jvp_linearity(self):
        s=self.state;p=self.tangent;r=phase_rotation_tangent(s)
        left=jvp(self.m,0,s,2*p-3*r,self.dy)
        right=2*jvp(self.m,0,s,p,self.dy)-3*jvp(self.m,0,s,r,self.dy)
        self.assertLess(np.max(abs(left-right)),1e-12)

    def test_10_global_u1_exact_zero_mode_of_jacobian(self):
        s=self.state
        v=phase_rotation_tangent(s)
        lhs=jvp(self.m,0,s,v,self.dy)
        rhs=phase_rotation_tangent(self.m.rhs(0,s.ravel(),s.shape[-1],self.dy).reshape(s.shape))
        self.assertLess(np.max(abs(lhs-rhs)),1e-11)

    def test_11_u1_rotation_leaves_physical_observables_unchanged(self):
        s=self.state;v=phase_rotation_tangent(s)
        self.assertLess(np.max(abs(geometry_derivative(s,v,self.grid))),1e-12)
        dc00,dc0y=constraint_jvp(self.m,s,v,self.dy)
        self.assertLess(max(np.max(abs(dc00)),np.max(abs(dc0y))),1e-11)

    def test_12_u1_brane_robin_covariance(self):
        s=self.state;rot=phase_rotation_tangent(s)
        bc=self.m.boundary_slopes(*s[:5]);dbc=junction_jvp(self.m,s,rot)
        self.assertLess(np.max(abs(dbc['x']+bc['y'])),1e-12)
        self.assertLess(np.max(abs(dbc['y']-bc['x'])),1e-12)
        for q in ['A','B','chi']:
            self.assertLess(np.max(abs(dbc[q])),1e-12)

    def test_13_charge_variation_matches_nonlinear_fd(self):
        s=self.state;p=self.tangent
        eps=3e-5
        qp=self.m.global_charge(s+eps*p,self.grid)
        qm=self.m.global_charge(s-eps*p,self.grid)
        self.assertAlmostEqual((qp-qm)/(2*eps),charge_derivative(s,p,self.grid),delta=5e-10)

    def test_14_initial_family_satisfies_linear_junctions(self):
        s=self.state;p=self.tangent;dy=self.dy
        dbc=junction_jvp(self.m,s,p)
        for row,key in enumerate(['A','B','chi','x','y']):
            vals=p[row]
            dleft=(-25*vals[0]+48*vals[1]-36*vals[2]+16*vals[3]-3*vals[4])/(12*dy)
            dright=(25*vals[-1]-48*vals[-2]+36*vals[-3]-16*vals[-4]+3*vals[-5])/(12*dy)
            # One-sided spatial derivatives have finite grid truncation error.
            self.assertLess(abs(dleft-dbc[key][0]),3e-5, key)
            self.assertLess(abs(dright-dbc[key][1]),3e-5, key)

    def test_15_domega_parameter_is_initial_data_only(self):
        _,base,domega=initial_tangent(self.m,n=49,parameter='bulk_omega',step=1e-4)
        self.assertLess(np.max(abs(base-self.state)),1e-13)
        self.assertGreater(np.max(abs(domega[9])),.01)
        self.assertLess(np.max(abs(domega[4])),1e-12)

    def test_16_tangent_matches_independent_nonlinear_trajectories(self):
        r=evolve_tangent(self.m,n=49,time_end=.08,samples=4)
        check=finite_difference_response(self.m,r,epsilon=5e-4)
        self.assertLess(np.max(check['max_relative_error']),2e-5)
        self.assertGreater(abs(r['linear_observables'][-1,0]),.1)
        self.assertGreater(abs(r['linear_observables'][-1,2]),1e-3)

    def test_17_response_centred_difference_quadratic(self):
        r=evolve_tangent(self.m,n=33,time_end=.06,samples=4)
        x=finite_difference_response(self.m,r,epsilon=6e-3)
        y=finite_difference_response(self.m,r,epsilon=3e-3)
        a=np.max(x['absolute_error'][:,[0,2]])
        b=np.max(y['absolute_error'][:,[0,2]])
        self.assertGreater(a/b,2.7)

    def test_18_linearized_constraints_refine(self):
        lo=evolve_tangent(self.m,n=49,time_end=.06,samples=3)
        hi=evolve_tangent(self.m,n=97,time_end=.06,samples=3)
        c0=max(lo['linear_constraint_errors'][:,0])
        c1=max(hi['linear_constraint_errors'][:,0])
        self.assertGreater(c0/c1,2.5)

    def test_19_goldstone_covariance_at_evolved_state(self):
        out=evolve_tangent(self.m,n=33,time_end=.055,samples=3)
        s=out['states'][-1];dy=out['grid'][1]-out['grid'][0]
        v=phase_rotation_tangent(s)
        lhs=jvp(self.m,float(out['t'][-1]),s,v,dy)
        rhs=phase_rotation_tangent(self.m.rhs(float(out['t'][-1]),s.ravel(),s.shape[-1],dy).reshape(s.shape))
        self.assertLess(np.max(abs(lhs-rhs)),1e-10)

    def test_20_reject_inconsistent_parameter_changes(self):
        with self.assertRaises(ValueError):
            initial_tangent(self.m,n=33,parameter='bulk_lambda')
        with self.assertRaises(ValueError):
            initial_tangent(self.m,n=33,parameter='bulk_amp',step=.1)
        with self.assertRaises(ValueError):
            jvp(self.m,0,self.state,self.tangent[:-1],self.dy)


if __name__=='__main__':
    unittest.main()
