"""Run WS-02 checks and generate model diagnostics.

From this folder: python run_experiment.py
All numbers are dimensionless, with declared normalizations in README.md.
"""
from __future__ import annotations
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branes import ChargedBrane
from radion import RadionEFT
from helium3 import summary as he3_summary
from gw import GoldbergerWiseProbe

OUT=Path(__file__).resolve().parent/'results'
OUT.mkdir(parents=True,exist_ok=True)

def native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj,np.generic):
        return obj.item()
    if isinstance(obj,dict):
        return {str(k):native(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):
        return [native(x) for x in obj]
    return obj

def run():
    brane=ChargedBrane()
    Nsample=np.linspace(0,3,61)
    ysample=np.linspace(0,1.5,81)
    bulk_res=brane.einstein_bulk_residuals(Nsample,ysample)
    junction=brane.junction(Nsample,1.5)
    hist=brane.history(Nmax=4,n=501)
    eft=RadionEFT()
    rad=eft.evolve(Nmax=8,steps=1601)
    he=he3_summary(depth=1.0,n=65)
    he0=he3_summary(depth=0,n=65)
    gw35=GoldbergerWiseProbe(z_star=35)
    gw69=GoldbergerWiseProbe(z_star=69)
    effective_radion_mass=eft.Veff_second(rad['equilibrium_r'])
    # Dimensionless forecast only, NOT calibrated to cosmological data.
    report={
      'experiment':'WS-02',
      'status':'toy study; no 5D charged bulk scalar with stabilized two-brane solution; no helium3 or cosmology fit',
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
      'brane_parameters':{'k':brane.k,'M5_cubed':brane.M5cubed,'sigma':brane.sigma,
                          'V0':brane.V0,'alpha':brane.alpha,'alpha_X0_squared':brane.s0,
                          'charge_Q':brane.Q,'second_brane_coordinate_L':1.5},
      'brane_einstein_bulk_component_residuals':bulk_res,
      'brane_uv_israel_residuals':{'a':junction['uv_spatial_residual_max'],'n':junction['uv_temporal_residual_max']},
      'brane_initial_hubble':float(hist['H'][0]),
      'brane_final_hubble':float(hist['H'][-1]),
      'brane_charge_relative_drift':hist['charge_relative_drift'],
      'brane_continuity_residual':hist['continuity_max_abs'],
      'second_brane_required_matter':{
             'initial_rho_IR':float(junction['ir_required_density'][0]),
             'initial_p_IR':float(junction['ir_required_pressure'][0]),
             'late_rho_IR':float(junction['ir_required_density'][-1]),
             'late_p_IR':float(junction['ir_required_pressure'][-1]),
             'max_magnitude_across_samples':junction['ir_tuned_vacuum_mismatch']},
      'radion_parameters':{'M4':eft.M4,'mr':eft.mr,'beta':eft.beta,'V0':eft.V0,'alpha_X0_squared':eft.s0},
      'radion_conformal_identity':'A^4 P_J(X/A^2)=-V0 A^4+alpha X^2 for P_J=-V0+alpha X_J^2',
      'radion_r_equilibrium':rad['equilibrium_r'],
      'radion_min_effective_energy':rad['equilibrium_vacuum_energy'],
      'radion_m_eff_squared_at_min':float(effective_radion_mass),
      'radion_final_r':float(rad['r'][-1]),
      'radion_final_H':float(rad['H'][-1]),
      'radion_predicted_asymptotic_H':rad['asymptotic_H_expected'],
      'radion_friedmann_residual':rad['friedmann_residual_max'],
      'radion_raychaudhuri_finite_difference_error':rad['raychaudhuri_fd_max_abs'],
      'radion_eom_finite_difference_error':rad['radion_eom_fd_max_abs'],
      'radion_continuity_residual':rad['continuity_residual_max'],
      'radion_final_total_w':float(rad['w_total'][-1]),
      'radion_fluid_sound_speed_squared':1/3,
      'gw_probe':{'epsilon':gw35.epsilon,'z_targets':[35,69],
                  'vIR_over_vUV':[gw35.v_ir/gw35.v_uv,gw69.v_ir/gw69.v_uv],
                  'expected_local_minima':[gw35.stabilized_separation(),gw69.stabilized_separation()],
                  'curvature_at_minimum_normalized':[gw35.curvature_at_minimum(),gw69.curvature_at_minimum()],
                  'bulk_stabilized_two_brane_solution':False},
      'helium3_analogue':{'periodic_tetrad_texture':True,
                         'periodic_ring_analogue_not_full_BdG':True,
                         'texture_depth':he['texture_strength'],
                         'local_transverse_velocity_min':he['local_velocity_min'],
                         'local_transverse_velocity_max':he['local_velocity_max'],
                         'hermiticity_error':he['hermiticity_residual'],
                         'particle_hole_energy_symmetry_error':he['particle_hole_spectrum_residual'],
                         'lowest_positive_eigenvalues_textured':he['lowest_positive_eigenvalues'],
                         'lowest_positive_eigenvalues_flat':he0['lowest_positive_eigenvalues'],
                         'relative_first_excited_energy_shift':float((he['lowest_positive_eigenvalues'][1]/he0['lowest_positive_eigenvalues'][1])-1)}
    }
    (OUT/'summary.json').write_text(json.dumps(native(report),indent=2)+'\n',encoding='utf8')

    fig,axs=plt.subplots(2,3,figsize=(14.4,8.2))
    axs[0,0].plot(hist['N'],hist['H']**2,label='Exact RS2 brane H²')
    axs[0,0].plot(hist['N'],hist['rho']/(3*brane.M4squared),'--',label='4D low-energy H²')
    axs[0,0].set(xlabel='ln a',ylabel='H²',title='Expansion and rho² correction')
    axs[0,0].legend(fontsize=8)

    axs[0,1].plot(Nsample,junction['ir_required_density'],label='required IR rho')
    axs[0,1].plot(Nsample,junction['ir_required_pressure'],label='required IR p')
    axs[0,1].axhline(0,linewidth=0.6,color='grey')
    axs[0,1].set(xlabel='ln a',ylabel='dimensionless stress',title='Fixed second-brane junction mismatch')
    axs[0,1].legend(fontsize=8)

    axs[0,2].semilogy(hist['N'],hist['rho'],label='Total brane rho')
    axs[0,2].semilogy(hist['N'],hist['rho']-brane.V0,label='Charged fluid kinetic rho')
    axs[0,2].set(xlabel='ln a',ylabel='energy density',title='Conserved Q gives a^-4 dilution')
    axs[0,2].legend(fontsize=8)

    axs[1,0].plot(rad['N'],rad['r'],label='evolving radion')
    axs[1,0].axhline(rad['equilibrium_r'],linestyle='--',label='Lambert-W minimum')
    axs[1,0].set(xlabel='ln a_E',ylabel='canonically normalized r',title='Stabilized 4D radion EFT')
    axs[1,0].legend(fontsize=8)

    axs[1,1].plot(rad['N'],rad['vacuum'],label='V0 A(r)^4')
    axs[1,1].plot(rad['N'],rad['radion_potential'],label='m_r² r² / 2')
    axs[1,1].plot(rad['N'],rad['fluid_kinetic_density'],label='charged fluid kinetic rho')
    axs[1,1].set(xlabel='ln a_E',ylabel='energy density',title='Sources of effective 4D expansion')
    axs[1,1].legend(fontsize=8)

    eig=np.arange(1,9)
    axs[1,2].plot(eig,he['lowest_positive_eigenvalues'],'o-',label='Weyl texture')
    axs[1,2].plot(eig,he0['lowest_positive_eigenvalues'],'s--',label='uniform Weyl cone')
    axs[1,2].set(xlabel='positive mode index',ylabel='excitation energy',title='Hermitian helium-3-inspired ring')
    axs[1,2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'ws02_diagnostics.png',dpi=170)
    plt.close(fig)
    fig2,ax=plt.subplots(figsize=(7.4,4.2))
    for gw in (gw35,gw69):
        z=np.linspace(gw.z_star-0.8,gw.z_star+0.8,200)
        ax.plot(z-gw.z_star,gw.normalized_potential(z)/(gw.epsilon*gw.v_ir)**2,
                label=f'target kL={gw.z_star:g}, vIR/vUV={gw.v_ir/gw.v_uv:.4g}')
    ax.set(xlabel='z - z_star (z=kL)',ylabel='normalized GW probe potential',
           title='Local Goldberger-Wise probe minima')
    ax.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(OUT/'gw_probe.png',dpi=170)
    plt.close(fig2)
    print(json.dumps(native(report),indent=2))
    return report

if __name__=='__main__':
    run()
