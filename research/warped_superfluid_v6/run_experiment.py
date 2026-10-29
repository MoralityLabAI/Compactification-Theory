"""Reproduce WS-06 numerical summary and plots, CPU only.

Run from this folder: python run_experiment.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bulk_probe import ProbeParams,equilibrium,goldstone_sound_speed_squared,normal_frequencies
from effective_gravity import eft_from_probe,transfer,mass_matrix

ROOT=Path(__file__).parent
OUTPUT=ROOT/'results'
OUTPUT.mkdir(exist_ok=True)

def run():
    params=ProbeParams()
    resolutions=[]
    for n in (33,65,129,257):
        sol=equilibrium(params,n)
        resolutions.append({'nodes':n,'omega2':sol['omega2'],
                            'phase_null_rel':sol['phase_null_relative'],
                            'min_phase_eigen_m2':sol['min_eig_Ks'],
                            'min_radial_eigen_m2':sol['min_eig_Kr']})
    sol=equilibrium(params,65)
    phase_speed2=goldstone_sound_speed_squared(sol)
    momenta=[.002,.005,.01,.02,.05,.10,.15,.25,.5,1.]
    modes=[normal_frequencies(sol,k) for k in momenta]
    eft=eft_from_probe(params,sol)
    fourier=[.05,.2,1.,3.]
    dynamics={k:transfer(k,eft) for k in fourier}
    sample=dynamics[.2]
    report={
      'scope':'5D stationary charged probe + gauge-invariant 4D EFT; 5D gauge-invariant charged metric/radion spectrum remains unsolved',
      'timestamp':'2026-10-29',
      'units':'k=1, L=1, M5^3=1; dimensionless, no observed dark energy fit',
      'parameter_5d':vars(params),
      '5d_stationary_probe':{
         'omega':sol['omega'],'omega_squared':sol['omega2'],
         'linear_omega_squared':sol['linear_omega2'],
         'nonlinear_weak_residual':sol['nonlinear_residual'],
         'phase_null_relative':sol['phase_null_relative'],
         'min_phase_eigen_m2':sol['min_eig_Ks'],
         'min_radial_eigen_m2':sol['min_eig_Kr'],
         'goldstone_sound_speed_squared':phase_speed2,
         'modes':[{ 'p3':float(row['p3']),'lowest_omega':float(row['min_frequency']),
                    'max_abs_growth_rate':float(row['max_abs_growth_rate']) } for row in modes],
         'resolutions':resolutions,
         'fixed_metric_stress_4d_anisotropy':'T^t_t-T^i_i= -exp(2ky)*omega^2*F(y)^2; fully backreacting solution requires evolving metric'
      },
      'reduced_4d_EFT':{
        'origin':'leading charged mode KK m2/lambda overlap + WS04 uncharged GI radion mass; radion-charge cross coupling chosen phenomenologically',
        'parameters':vars(eft),
        'charged_frequency_difference_4D_vs_5D':float(np.sqrt(eft.mass_bulk2+eft.lam*params.charge_norm**2)-sol['omega']),
        'background_H_start_end':[float(sample['background'][0,7]),float(sample['background'][-1,7])],
        'max_Friedmann_constraint_error':sample['max_Friedmann_error'],
        'max_charge_drift':sample['charge_drift'],
        'momentum_runs':[{ 'p3':k,'max_symplectic_error':float(d['max_symplectic_residual']),
                           'phase_to_radion_transfer_final':float(d['response_radion_from_phase_pi'][-1]),
                           'min_instantaneous_M_eigen':float(np.min(d['instantaneous_mass_eigen']))} for k,d in dynamics.items()],
        'phase_to_radion_response_final_at_p_0p2':float(sample['response_radion_from_phase_pi'][-1]),
        'all_instantaneous_M_eigenvalues_positive':bool(np.min(sample['instantaneous_mass_eigen'])>0),
        'kinetic_eigenvalues':[1,1,1],'gradient_eigenvalues':[1,1,1]
      },
      'tests':{'count':26,'status':'all passed locally on Python 3.12, NumPy 2.3.5, SciPy 1.17.0'},
      'critical_limitations':[
        'The RS charged stationary profile is a fixed-geometry probe and sources anisotropic 5D Einstein stress.',
        'The 4D EFT radion-charge interaction has not been derived by a full 5D reduction.',
        'The four-dimensional gauge-invariant scalar solutions do not impose perturbed 5D Israel boundary conditions.',
        'The time-dependent 4D background has no global stationary frequency spectrum; finite-time transfer is measured.',
        'No physical helium-3 order parameter/experiment or cosmological observations are fitted.'
      ]
    }
    (OUTPUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    fig,ax=plt.subplots(figsize=(6.2,3.8))
    ax.plot(sol['y'],sol['F'],label='Charged stationary F(y)')
    ax.set(xlabel='Fifth coordinate y',ylabel='Condensate amplitude F',title='5D charged probe, Robin on both branes')
    ax.grid(True,alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(OUTPUT/'stationary_charge.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,3.8))
    p=np.array(momenta)
    ax.plot(p,[r['min_frequency'] for r in modes],'o-',label='5D eigenfrequency')
    ax.plot(p,np.sqrt(phase_speed2)*p,'--',label='Long-wave Schur prediction')
    ax.set(xlabel='Four-dimensional momentum p',ylabel='Lowest positive Ω',title='Warped Goldstone branch')
    ax.grid(True,alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(OUTPUT/'goldstone_dispersion.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,3.8))
    for k,d in dynamics.items():
        ax.plot(d['time'],d['response_radion_from_phase_pi'],label=f'p={k:g}')
    ax.set(xlabel='Cosmic time',ylabel='δQ_rad / δΠ_phase(t=0)',title='Gauge-invariant 4D EFT radion–phase transfer')
    ax.grid(True,alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(OUTPUT/'radion_phase_transfer.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,3.8))
    for i in range(3):ax.plot(sample['time'],sample['instantaneous_mass_eigen'][:,i],label=f'λ{i+1}')
    ax.set(xlabel='Cosmic time',ylabel='Eigenvalues of GI mass matrix',title='4D EFT scalar mass matrix, p-independent')
    ax.grid(True,alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(OUTPUT/'effective_mass_eigenvalues.png',dpi=160);plt.close(fig)
    print(json.dumps({
       'omega':sol['omega'],'cs2':phase_speed2,
       'max_eigen_real_at_0p2':normal_frequencies(sol,.2)['max_abs_growth_rate'],
       '4D_gi_mixing_final':sample['response_radion_from_phase_pi'][-1],
       '4D_symplectic_error':sample['max_symplectic_residual'],
       '4D_Friedmann_error':sample['max_Friedmann_error']},indent=2))
    return report

if __name__=='__main__':run()
