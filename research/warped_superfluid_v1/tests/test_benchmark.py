"""Run with `python -m unittest discover -s tests -v` from package folder."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from scipy.special import jn_zeros
from warped_superfluid import (
    rs_exact_scaled_roots,rs_mode_ir_fraction,fe_masses,DomainWall,
    integrate_ds_wall,ads_ds_warp,kessence_fluid,finite_density_anisotropy,
)

class TestWarpedSuperfluid(unittest.TestCase):
    def test_rs_bessel_j1_limit(self):
        roots = rs_exact_scaled_roots(35,4)
        self.assertLess(np.max(np.abs(roots-jn_zeros(1,4))),1e-10)

    def test_unwarped_neumann_and_refinement(self):
        target = np.pi/4
        errs=[]
        for nodes in [65,129,257]:
            y=np.linspace(0,4,nodes)
            m=fe_masses(y,0*y,count=2)
            self.assertLess(abs(m[0]),1e-7)
            errs.append(abs(m[1]-target))
        self.assertGreater(errs[0]/errs[1],3.5)
        self.assertGreater(errs[1]/errs[2],3.5)

    def test_warped_neumann_refinement(self):
        ref=rs_exact_scaled_roots(4,2)[0]*np.exp(-4)
        errs=[]
        for nodes in [65,129,257]:
            y=np.linspace(0,4,nodes)
            m=fe_masses(y,y,count=2)
            errs.append(abs(m[1]-ref))
        self.assertGreater(errs[0]/errs[1],3.5)
        self.assertGreater(errs[1]/errs[2],3.5)
        self.assertLess(errs[-1]/ref,4e-4)

    def test_ir_localization(self):
        root=rs_exact_scaled_roots(35,1)[0]
        self.assertGreater(rs_mode_ir_fraction(35,root),.5)

    def test_principal_symbol_is_insufficient(self):
        y=np.linspace(0,4,193)
        full=fe_masses(y,y,count=1)[1]
        naive=fe_masses(y,y,count=1,naive_speed_only=True)[1]
        self.assertGreater(abs(full-naive)/full,0.01)

    def test_scalar_wall_5d_einstein(self):
        wall=DomainWall()
        self.assertLess(max(wall.exact_residual(np.linspace(0,4,120))),1e-10)
        y,sol,res=integrate_ds_wall(wall,4,0)
        self.assertLess(res,1e-8)
        self.assertLess(np.max(np.abs(sol[0]-wall.warp(y))),1e-8)
        self.assertLess(np.max(np.abs(sol[2]-wall.chi(y))),1e-8)

    def test_ds_sliced_5d_einstein(self):
        wall=DomainWall()
        y,sol,res=integrate_ds_wall(wall,4,0.005)
        self.assertLess(res,1e-8)
        self.assertGreater(sol[0,-1],wall.warp(np.array([4.0]))[0])
        self.assertLess(abs(ads_ds_warp(np.array([4.0]),1e-7)[0]-4.0),1e-6)

    def test_finite_density_no_go_and_kessence(self):
        self.assertAlmostEqual(finite_density_anisotropy(0.5),-1)
        z=np.array([0.0,1.0,3.0])
        rho,p,w,cs2=kessence_fluid(z,1e-3)
        self.assertTrue(np.all((w>-1)&(w<1/3)))
        self.assertTrue(np.allclose(cs2,1/3))
        self.assertAlmostEqual((rho[2]-1)/(rho[0]-1),256)
        self.assertTrue(np.allclose(p/rho,w))

if __name__=='__main__':
    unittest.main()
