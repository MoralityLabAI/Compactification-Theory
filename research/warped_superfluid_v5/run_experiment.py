"""Reproduce WS-05 Goldstone and fully coupled k=0 tangent-response diagnostics.

Run from folder: python run_experiment.py
Requires adjacent WS-03 and WS-04 checked-out folders.
"""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'warped_superfluid_v3'),
              str(ROOT.parent/'warped_superfluid_v4'),str(ROOT)]
from bulk_complex import BulkChargedSystem
from goldstone import HomogeneousCondensate
from isocurvature import entropy_projection,unit_entropy_normal
from tangent import (evolve_tangent,finite_difference_response,phase_rotation_tangent,
                     jvp,geometry_derivative,charge_derivative,constraint_jvp)


def run():
    outdir=ROOT/'results';outdir.mkdir(exist_ok=True)
    m=BulkChargedSystem()
    ref=HomogeneousCondensate(F=m.bulk_amp,mass2=m.bulk_mass2,coupling=m.bulk_lambda)
    control=HomogeneousCondensate(F=.5,mass2=m.bulk_mass2,coupling=m.bulk_lambda)
    modes={}
    for F in [.02,.2,.5,1.]:
        c=HomogeneousCondensate(F=F,mass2=m.bulk_mass2,coupling=m.bulk_lambda)
        modes[str(F)]={'omega':c.equilibrium_frequency,'radial_curvature':c.radial_curvature,
                       'mass_gap_squared':c.gap_squared,'sound_speed_squared':c.goldstone_sound_squared}
    references={}
    results={}
    for nodes in (33,65,129):
        x=evolve_tangent(m,n=nodes,time_end=.12,samples=7,initial_parameter='bulk_amp')
        err=x['linear_constraint_errors']
        references[nodes]=x
        results[str(nodes)]={'max_linear_H_constraint':float(np.max(np.abs(err[:,0]))),
                             'max_linear_momentum_constraint':float(np.max(np.abs(err[:,1]))),
                             'initial_linear_H':x['linear_observables'][0,0],
                             'final_linear_H':x['linear_observables'][-1,0],
                             'initial_linear_proper_length':x['linear_observables'][0,1],
                             'final_linear_proper_length':x['linear_observables'][-1,1],
                             'initial_linear_charge':x['linear_observables'][0,2],
                             'final_linear_charge':x['linear_observables'][-1,2]}
    primary=references[65]
    spacing=primary['grid'][1]-primary['grid'][0]
    entropy_series=np.array([entropy_projection(m,s,d,spacing)
                  for s,d in zip(primary['states'],primary['tangent_states'])])
    entropy_rms=np.sqrt(np.mean(entropy_series**2,axis=1))
    normal,rank_mag=unit_entropy_normal(m,primary['states'][-1],spacing)
    compare=finite_difference_response(m,primary,epsilon=5e-4)
    smaller=finite_difference_response(m,primary,epsilon=2.5e-4)
    # Exact symmetry of the semi-discrete field equations, metric and boundary slopes.
    s=primary['states'][-1]; dy=primary['grid'][1]-primary['grid'][0]
    phase=phase_rotation_tangent(s)
    lhs=jvp(m,float(primary['t'][-1]),s,phase,dy)
    rhs=phase_rotation_tangent(m.rhs(float(primary['t'][-1]),s.ravel(),65,dy).reshape(s.shape))
    u1_error=float(np.max(np.abs(lhs-rhs)))
    phase_obs=geometry_derivative(s,phase,primary['grid'])

    summary={'status':'short-time homogeneous k_3D=0 gauge-fixed coupled tangent response; full GI spectrum remains open',
             'version':'WS-05','date':'2026-10-29',
             'background':{'k':m.k,'M5_cubed':m.M3,'proper_brane_separation':m.proper_L,
                           'bulk_charge_initial_amplitude':m.bulk_amp,'bulk_phase_frequency':m.bulk_omega,
                           'bulk_mass2':m.bulk_mass2,'bulk_quartic':m.bulk_lambda,
                           'note':'curved/inhomogeneous dynamical background, NOT homogeneous equilibrium'},
             'flat_homogeneous_control':{'equilibrium_frequency_at_0p02':ref.equilibrium_frequency,
                                        'reference_input_phase_frequency':m.bulk_omega,
                                        'local_UV_equilibrium_frequency_gap':ref.local_frequency_gap(m.bulk_omega,0),
                                        'omega2_minus_equilibrium_omega2':m.bulk_omega**2-ref.equilibrium_frequency**2,
                                        'sample_goldstone_branches':modes},
             'gauge_invariant_entropy':{'construction':'n_I delta q^I, n proportional to dt q cross dy q',
                         'rank_min_final':float(np.min(rank_mag)),
                         'rms_t':entropy_rms.tolist(),
                         'first_profile':entropy_series[0].tolist(),
                         'last_profile':entropy_series[-1].tolist(),
                         'scope':'gauge-invariant local scalar projection; no complete coupled GI spectrum'},
             'metric_phase_response':{'nodes':65,'time_end':float(primary['t'][-1]),
                                      'max_relative_errors_FD_epsilon_5e_minus4':compare['max_relative_error'].tolist(),
                                      'max_relative_errors_FD_epsilon_2p5e_minus4':smaller['max_relative_error'].tolist(),
                                      'u1_covariance_error':u1_error,
                                      'u1_observable_derivative':phase_obs.tolist(),
                                      't':primary['t'].tolist(),
                                      'tangent_observables':primary['linear_observables'].tolist(),
                                      'finite_difference_observables':compare['finite_difference'].tolist(),
                                      'names':['delta_H_UV_proper','delta_constant_t_proper_length','delta_global_U1_charge']},
             'linear_constraint_refinement':results,
             'limitations':['Only x-homogeneous perturbations (3D momentum = 0)',
                            'Conformal-coordinate gauge fixed; complete gauge-invariant radion and phase decomposition still open',
                            'No dynamical background spectrum or long-time Lyapunov exponent',
                            'No holographic UV completion, no helium-3 BdG fit, no dark-energy derivation']}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')

    plt.figure(figsize=(7,4.4))
    p=np.linspace(0,1.5,300)
    for F in [.02,.5]:
        c=HomogeneousCondensate(F=F,mass2=m.bulk_mass2,coupling=m.bulk_lambda)
        low,high=c.omega_squared_branches(p)
        plt.plot(p,np.sqrt(low),label=f'Goldstone F={F:g}')
        plt.plot(p,np.sqrt(high),ls='--',label=f'Gapped F={F:g}')
    plt.xlabel('3D momentum magnitude (flat-space units)')
    plt.ylabel('Mode angular frequency')
    plt.title('Exact homogeneous complex-scalar equilibrium modes')
    plt.grid(alpha=.2);plt.legend(fontsize=8);plt.tight_layout()
    plt.savefig(outdir/'goldstone_dispersion.png',dpi=170);plt.close()

    for j,(name,units) in enumerate([('UV proper H response','dH/dF'),
                                     ('Radion-slice response','dL/dF'),
                                     ('Global charge response','dQ/dF')]):
        plt.figure(figsize=(6.8,4))
        plt.plot(primary['t'],primary['linear_observables'][:,j],marker='o',label='Tangent-linear full 5D system')
        plt.plot(primary['t'],compare['finite_difference'][:,j],marker='x',ls='--',label='Independent nonlinear central difference')
        plt.xlabel('Conformal simulation time')
        plt.ylabel(units)
        plt.title(name)
        plt.grid(alpha=.2);plt.legend(fontsize=8);plt.tight_layout()
        plt.savefig(outdir/['hubble_response.png','radion_slice_response.png','charge_response.png'][j],dpi=170)
        plt.close()

    plt.figure(figsize=(6.8,4))
    plt.plot(primary['grid'],entropy_series[0],label='Initial')
    plt.plot(primary['grid'],entropy_series[-1],label='t=0.12')
    plt.xlabel('Conformal bulk coordinate y')
    plt.ylabel('n · delta(q_chi, q_x, q_v)')
    plt.title('Gauge-invariant field-space entropy projection')
    plt.grid(alpha=.2);plt.legend();plt.tight_layout()
    plt.savefig(outdir/'entropy_isocurvature.png',dpi=170);plt.close()

    plt.figure(figsize=(6.8,4))
    for col,lab in [(0,'Hamiltonian'),(1,'Momentum')]:
        plt.loglog(list(results.keys()),[results[z][['max_linear_H_constraint','max_linear_momentum_constraint'][col]] for z in results],marker='o',label=lab)
    plt.xlabel('Grid nodes');plt.ylabel('Max tangent constraint residual')
    plt.title('Grid refinement of linearized Einstein constraints')
    plt.grid(alpha=.2);plt.legend();plt.tight_layout()
    plt.savefig(outdir/'linear_constraint_convergence.png',dpi=170);plt.close()
    print(json.dumps({
        'equilibrium_frequency':ref.equilibrium_frequency,
        'imposed_frequency':m.bulk_omega,
        'max_relative_FD_response_error':compare['max_relative_error'].tolist(),
        'tangent_last':primary['linear_observables'][-1].tolist(),
        'constraint_refinement':results,
        'U1_symmetry_residual':u1_error,
        'entropy_rms_initial_final':[float(entropy_rms[0]),float(entropy_rms[-1])]},indent=2))
    return summary

if __name__=='__main__': run()
