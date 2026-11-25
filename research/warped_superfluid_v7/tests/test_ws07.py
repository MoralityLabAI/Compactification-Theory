"""WS-07 self-gravitating charged bulk and exact stationary-brane sum-rule gates."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.integrate import simpson
from scipy.linalg import eigvalsh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from self_gravity import (Params,odes,equilibrium,initial_values,integrate,
                          constraint,brane_actions,pressure_terms,independent_einstein_residual,scalar_potential,
                          potential_derivatives)
from tensor_modes import (tensor_modes,tensor_matrices,tensor_speed_squared,
                          tensor_energy,tensor_energetic_positive)


class TestChargedBackground(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.par=Params()
        cls.sol=equilibrium(cls.par,n=401)

    def test_nodeless_and_nonzero_charge(self):
        self.assertGreater(np.min(self.sol['fields'][3]),0)
        self.assertGreater(self.sol['identities']['bulk_charge'],0)

    def test_both_charged_robin_conditions(self):
        d=self.sol['identities']
        self.assertLess(abs(d['UV_F_robin']),1e-12)
        self.assertLess(abs(d['IR_F_robin']),1e-10)

    def test_einstein_hamiltonian_propagates(self):
        self.assertLess(self.sol['identities']['max_hamiltonian_constraint'],1e-9)

    def test_three_independent_einstein_components(self):
        row=independent_einstein_residual(self.sol)
        self.assertLess(float(np.max(abs(row))),1e-9)

    def test_finite_difference_einstein_components(self):
        row=independent_einstein_residual(self.sol,finite_difference=True)
        self.assertLess(float(np.max(abs(row[:,2:-2]))),5e-4)

    def test_tuned_uv_constraint_root(self):
        u=initial_values(self.sol['omega'],self.par)
        self.assertLess(abs(constraint(u,self.sol['omega'],self.par)),1e-13)
        self.assertAlmostEqual(u[4],u[5],places=14)

    def test_charged_anisotropy_drives_geometric_split(self):
        z=self.sol['fields'];d=self.sol['identities']
        self.assertGreater(np.min(pressure_terms(z,self.sol['omega'],self.par)[1]),0)
        self.assertAlmostEqual(d['uv_D'],0,places=12)
        self.assertGreater(d['ir_D'],0.1)

    def test_local_anisotropy_identity_independent_difference(self):
        z=self.sol['fields'];y=self.sol['y'];A,N,c,F,Ap,Np,cp,Fp=z;p=self.par
        D=Np-Ap; volume=np.exp(N+3*A)
        left=np.gradient(volume*D,y,edge_order=2)
        right=(self.sol['omega']**2/p.M3)*np.exp(-N+3*A)*F*F
        self.assertLess(np.max(np.abs(left[2:-2]-right[2:-2])),1e-4)

    def test_integrated_anisotropy_sum_rule(self):
        d=self.sol['identities']
        self.assertGreater(d['sumrule_integral'],0)
        self.assertLess(d['sumrule_quad_error'],5e-8)

    def test_sum_rule_quadrature_refines(self):
        rows=[equilibrium(self.par,n=n)['identities']['sumrule_quad_error'] for n in (65,129,257)]
        self.assertGreater(rows[0]/rows[1],3.8)
        self.assertGreater(rows[1]/rows[2],3.8)

    def test_both_brane_nec_impossible(self):
        d=self.sol['identities']
        self.assertAlmostEqual(d['uv_rho_plus_p'],0,places=12)
        self.assertLess(d['ir_rho_plus_p'],-0.5)
        self.assertLess(abs(d['ir_rho_plus_p']+2*self.par.M3*d['ir_D']),1e-12)
        self.assertLess(abs(d['uv_rho_plus_p']-2*self.par.M3*d['uv_D']),1e-12)

    def test_positive_omega_cases_force_ir_nec_violation(self):
        values=[]
        for f in [.02,.04,.08]:
            row=equilibrium(replace(self.par,F0=f),n=65)
            values.append(-row['identities']['ir_rho_plus_p'])
        self.assertTrue(np.all(np.diff(values)>0))

    def test_exact_uncharged_limit_recovers_gw_wall(self):
        zero=replace(self.par,F0=0)
        y,z=integrate(0.,zero,points=129)
        c=zero.chi0*np.exp(zero.u*y)
        W=lambda x: 3*zero.M3*zero.k+.5*zero.u*x*x
        A=-zero.k*y-zero.chi0**2*np.expm1(2*zero.u*y)/(12*zero.M3)
        self.assertLess(np.max(abs(z[2]-c)),1e-11)
        self.assertLess(np.max(abs(z[0]-A)),1e-11)
        self.assertLess(np.max(abs(z[1]-z[0])),1e-12)

    def test_analytic_odes_match_finite_difference(self):
        y=self.sol['y'];z=self.sol['fields'];p=self.par
        derivs=np.gradient(z,y,axis=1,edge_order=2)
        predicted=odes(y,z,self.sol['omega'],p)
        self.assertLess(np.max(abs(derivs[:,2:-2]-predicted[:,2:-2])),4e-4)

    def test_ir_ghost_brane_action_reconstructs_israel(self):
        d=self.sol['identities'];a=brane_actions(self.sol)
        self.assertGreater(a['ir_ghost_proper_phase_gradient_squared'],0)
        self.assertAlmostEqual(a['ir_rho_from_action'],d['ir_surface_rho'],places=12)
        self.assertAlmostEqual(a['ir_p_from_action'],d['ir_surface_p'],places=12)
        self.assertAlmostEqual(a['uv_rho_from_action'],d['uv_surface_rho'],places=12)
        self.assertAlmostEqual(a['uv_p_from_action'],d['uv_surface_p'],places=12)
        self.assertLess(max(abs(a[k]) for k in a if k.endswith('slope_action_residual')),1e-9)

    def test_ghost_is_physical_obstruction(self):
        a=brane_actions(self.sol)
        self.assertGreater(a['ir_ghost_proper_phase_gradient_squared'],0)
        self.assertLess(self.sol['identities']['ir_rho_plus_p'],0)

    def test_no_bulk_y_charge_flux(self):
        # Background has a common y-independent U(1) phase: J^y=0 pointwise.
        z=self.sol['fields'];F=z[3];Fp=z[7]
        x=F;v=np.zeros_like(F);xp=Fp;vp=np.zeros_like(F)
        self.assertLess(np.max(abs(x*vp-v*xp)),1e-13)

    def test_boundary_potentials_reconstruct_chi_slope(self):
        a=brane_actions(self.sol);z=self.sol['fields']
        self.assertAlmostEqual(z[6,0],0.5*a['uv_chi_linear'],places=12)
        self.assertAlmostEqual(z[6,-1],-0.5*a['ir_chi_linear'],places=12)


class TestTensorGravitons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sol=TestChargedBackground.sol

    def test_neumann_exact_zero_mode(self):
        m,_=tensor_modes(self.sol,nodes=129,p3=0,count=4)
        self.assertLess(m[0],1e-8)
        self.assertGreater(m[1],0)

    def test_tensor_positive_at_finite_momentum(self):
        for p in [.002,.02,.2]:
            eig,_=tensor_modes(self.sol,nodes=65,p3=p,count=4)
            self.assertGreater(eig[0],0)
            self.assertTrue(np.all(np.diff(eig)>0))

    def test_tensor_speed_exceeds_uv_photon_speed(self):
        c2=tensor_speed_squared(self.sol)
        self.assertGreater(c2,1.005)
        self.assertLess(c2,1.02)

    def test_low_momentum_dispersion_matches_exact_integral(self):
        c2=tensor_speed_squared(self.sol)
        eig,_=tensor_modes(self.sol,nodes=129,p3=.002,count=2)
        self.assertLess(abs(eig[0]/(.002**2)-c2),1e-4)

    def test_tensor_fe_convergence(self):
        low=[tensor_modes(self.sol,nodes=n,p3=0.,count=2)[0][1] for n in (33,65,129,257)]
        d1=abs(low[0]-low[1]);d2=abs(low[1]-low[2]);d3=abs(low[2]-low[3])
        self.assertGreater(d1/d2,3.3)
        self.assertGreater(d2/d3,3.3)

    def test_tensor_zero_charge_limit_speed_one(self):
        zero=replace(Params(),F0=0)
        y,z=integrate(0,zero,points=401)
        sol={'y':y,'fields':z}
        self.assertLess(abs(tensor_speed_squared(sol)-1),1e-12)

    def test_tensor_positive_energy(self):
        self.assertGreater(tensor_energetic_positive(self.sol,p3=.1,nodes=65),0)
        rng=np.random.default_rng(14)
        h=rng.normal(size=65);hd=rng.normal(size=65)
        self.assertGreater(tensor_energy(self.sol,.1,h,hd),0)

    def test_tensor_symmetric_generalized_matrices(self):
        _,K,M,P=tensor_matrices(self.sol,65)
        self.assertLess(np.max(abs(K-K.T)),1e-13)
        self.assertLess(np.max(abs(M-M.T)),1e-13)
        self.assertLess(np.max(abs(P-P.T)),1e-13)
        self.assertGreater(np.min(eigvalsh(M)),0)
        self.assertGreater(np.min(eigvalsh(P)),0)


if __name__ == '__main__':unittest.main()
