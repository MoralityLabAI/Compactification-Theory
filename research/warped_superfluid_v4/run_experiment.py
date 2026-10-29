"""Reproduce WS-04 radion and nonlinear globally charged bulk diagnostics."""
from __future__ import annotations
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'warped_superfluid_v3'),str(HERE)]
from ws03 import System
from probes import probe_spectrum
from radion import RadionModel
from bulk_complex import BulkChargedSystem


def run():
    out=HERE/'results'
    out.mkdir(exist_ok=True)
    rad=RadionModel()
    modes=rad.modes(count=4,m2max=30,samples=240,scan_tachyonic=25)
    m2=np.asarray(modes['positive_m2'])
    zero_mode=rad.gauge_invariant_reconstruction(float(m2[0]))
    g0,gL=rad.gvalues()
    bad=RadionModel(stiffness_ir=-3.)
    tachyon=brentq(bad.boundary_residual,-.1,-.00001,xtol=1e-12)
    probe=probe_spectrum(System(s0=0),nodes=257,count=4)
    stiffness=[]
    for value in [.5,1,2,4,8,12]:
        rr=RadionModel(stiffness_uv=value,stiffness_ir=value)
        root=brentq(rr.boundary_residual,0,.02,xtol=2e-13)
        stiffness.append({'stiffness':float(value),'lowest_radion_m2':float(root),
                          'boundary_g_uv':rr.gvalues()[0],'boundary_g_ir':rr.gvalues()[1]})
    bulk=BulkChargedSystem()
    runs={}
    for n in [65,129,257]:
        sim=bulk.evolve(n=n,time_end=.2,samples=9)
        runs[str(n)]={
            'max_constraint_00':sim['max_constraint00'],
            'max_constraint_0y':sim['max_constraint0y'],
            'max_charge_drift':sim['max_charge_drift'],
            'max_junction_error':max(d['junction_error'] for d in sim['diagnostics']),
            'max_endpoint_flux':max(d['normal_current_flux'] for d in sim['diagnostics']),
            'initial_Huv_squared':sim['initial']['H2_uv'],
            'initial_hamiltonian_residual':sim['initial']['constraint_00_exact'],
            'initial_momentum_residual':sim['initial']['constraint_0y_exact'],
            'initial_junction_residual':max(sim['initial']['junction_exact'].values()),
            'initial_charge':sim['diagnostics'][0]['charge'],
            'final_charge':sim['diagnostics'][-1]['charge'],
            'final_Huv':sim['diagnostics'][-1]['Huv'],
            'final_radion_proper_length':sim['diagnostics'][-1]['proper_length'],
            'trajectory':sim['diagnostics']}
    initial=bulk.initial_slice(n=97)
    stress_difference=-np.exp(-2*initial['B'])*bulk.bulk_omega**2*initial['x']**2
    blank=BulkChargedSystem(bulk_amp=0,bulk_omega=0).initial_slice(n=97)
    report={
        'status':'dimensionless prototype; zero-charge GI eigenproblem and distinct fully backreacted charged evolution',
        'radion':{
            'theory':'Lesgourgues & Sorbo 2004, hep-th/0310007 Eqs. 6-10,16-17',
            'background':'static BPS, no bulk charge, one GW scalar',
            'positive_squared_masses':m2.tolist(),
            'first_radion_mass':float(np.sqrt(m2[0])),
            'boundary_coefficients':{'g_minus_uv':g0,'g_plus_ir':gL},
            'stable_under_analytic_sign_theorem':bool(g0>0 and gL<0),
            'negative_scan':modes['negative_m2_scan'],
            'first_root_boundary_residual':float(zero_mode['boundary_residual']),
            'fixed_metric_scalar_probe_m2':probe.tolist(),
            'lowest_probe_to_radion_m2_ratio':float(probe[0]/m2[0]),
            'negative_ir_stiffness_tachyon_m2':float(tachyon),
            'stiffness_sweep':stiffness,
        },
        'bulk':{
            'theory':'full 5D Einstein + GW scalar + two real fields of canonical complex scalar',
            'global_U1':'coherent finite-density phase; no gauge field or demonstrated phase transition',
            'parameters':{
                'M5_cubed':bulk.M3,'k':bulk.k,'proper_length':bulk.proper_L,
                'GW_u':bulk.u,'GW_uv':bulk.chi_uv,
                'bulk_amp':bulk.bulk_amp,'phase_frequency':bulk.bulk_omega,
                'bulk_mass2':bulk.bulk_mass2,'bulk_lambda':bulk.bulk_lambda,
                'bulk_stiffness_uv':bulk.bulk_stiffness_uv,
                'bulk_stiffness_ir':bulk.bulk_stiffness_ir,
                'initial_UV_fluid_s0':bulk.s0,
            },
            'grids':runs,'zero_amplitude_static_H2':blank['H2_uv'],
            'maximum_initial_temporal_spatial_stress_anisotropy':float(np.max(np.abs(stress_difference)))
        },
        'limitations':[
            'Radion eigenvalues apply to uncharged static one-scalar background, not evolving charged two-scalar geometry',
            'No claim of long-time stability or nonlinearly stable radion in charged model',
            'Bulk scalar has global U(1), no gauged Maxwell sector',
            'Dimensionless small-warp kL=1, no helium-3 measurements or cosmic data fits',
            'Dark-energy scale and vacuum renormalization remain unexplained'
        ],
        'tests':{'number':22,'expected':'python -m unittest discover -s tests -v'}
    }
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    axs[0,0].semilogy(np.arange(1,5),m2,'o-',label='GI radion + scalar')
    axs[0,0].semilogy(np.arange(1,5),probe,'s--',label='Fixed-metric GW probe')
    axs[0,0].set(xlabel='Mode index',ylabel='Squared mass in curvature units',title='Light physical radion vs scalar probe')
    axs[0,0].legend(fontsize=8)
    z=zero_mode['z']
    axs[0,1].plot(z,zero_mode['psi'],label='GI metric Ψ')
    axs[0,1].plot(z,zero_mode['Phi'],label='GI metric Φ')
    axs[0,1].plot(z,zero_mode['Delta'],label='GI scalar Δ')
    axs[0,1].set(xlabel='Proper bulk coordinate z',ylabel='Relative eigenfunction amplitude',
                 title='Physical scalar-metric mode')
    axs[0,1].legend(fontsize=8)
    for n in ['65','129','257']:
        ds=runs[n]['trajectory']
        axs[1,0].semilogy([d['time'] for d in ds],[d['charge_drift']+1e-16 for d in ds],
                          label=f'N={n}')
    axs[1,0].set(xlabel='Time (dimensionless)',ylabel='Relative charge error',
                 title='Bulk U(1) charge conservation')
    axs[1,0].legend(fontsize=8)
    nodes=[65,129,257]
    axs[1,1].loglog(nodes,[runs[str(n)]['max_constraint_00'] for n in nodes],
                     'o-',label='Hamiltonian')
    axs[1,1].loglog(nodes,[runs[str(n)]['max_constraint_0y'] for n in nodes],
                     's-',label='Momentum')
    axs[1,1].loglog(nodes,[runs[str(n)]['max_charge_drift'] for n in nodes],
                     'x-',label='Charge drift')
    axs[1,1].set(xlabel='Grid nodes',ylabel='Maximum residual',title='Convergence checks')
    axs[1,1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'diagnostics.png',dpi=170)
    plt.close(fig)
    print(json.dumps({'radion_m2':m2.tolist(),'tachyon_m2':float(tachyon),
                      'grid_results':{n:{'constraint00':runs[n]['max_constraint_00'],
                                         'constraint0y':runs[n]['max_constraint_0y'],
                                         'charge_drift':runs[n]['max_charge_drift']} for n in runs}},indent=2))
    return report


if __name__=='__main__':
    run()
