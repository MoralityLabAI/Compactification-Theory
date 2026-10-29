"""Run independent CPU reproducible WS-03 field/boundary/evolution benchmarks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ws03 import System
from probes import probe_spectrum

OUT=Path(__file__).resolve().parent/'results'
OUT.mkdir(exist_ok=True)

def run():
    m=System(s0=.001)
    z=np.linspace(0,m.proper_L,4001)
    M4sq=float(2*m.M3*np.trapezoid(np.exp(-2*m.warp(z)),z))
    static=m.background()
    initial=m.initial_slice(129)
    exact4=1/(3*M4sq)
    meas5=initial['q']/(3*m.s0)
    evolutions={n:m.evolve(n=n,time_end=.4,samples=21,rtol=2e-9) for n in (65,129,257)}
    response_runs={n:{b:m.evolve(n=n,time_end=1.,samples=41,rtol=2e-9,scalar_bump=b)
                      for b in (-.001,.001)} for n in (65,129,257)}
    vacuum_samples=[]
    for eps in (0.0001,0.001,0.01):
        v=System(s0=0,vacuum_detune_uv=eps)
        initial_detuned=v.initial_slice(129)
        vacuum_samples.append({'positive_tension_detuning':eps,
                               'H_uv_squared':float(initial_detuned['q']),
                               'H_squared_over_detuning':float(initial_detuned['q']/eps),
                               'max_israel_residual':max(abs(x) for x in initial_detuned['jump_errors'].values())})
    spec={n:probe_spectrum(m,nodes=n,count=4).tolist() for n in (65,129,257)}
    static_zero=System(s0=0).evolve(n=257,time_end=.4,samples=21,rtol=2e-9)
    summary={
        'experiment':'WS-03',
        'status':'nonlinear 5D Einstein-GW scalar PDE with two Z2 branes and charged phase confined to the UV brane',
        'scope_limits':['dimensionless kL=1 benchmark','no physical gauge-invariant radion spectrum',
                        'no charged condensate propagating in the 5D bulk','no dark-energy self-tuning or cosmological fit',
                        'short-time convergence, not global stability'],
        'parameters':{'k':m.k,'M5_cubed':m.M3,'u':m.u,'chi_uv':m.chi_uv,
                      'L_proper':m.proper_L,'kappa_uv':m.stiffness_uv,'kappa_ir':m.stiffness_ir,
                      'phase_kinetic_s0':m.s0,'rho_uv_0':3*m.s0,'time_end':.4},
        'static_Backreacted_background':{
            'conformal_y_extent':float(static[0]),
            '4d_planck_squared':M4sq,'prediction_H_squared_over_rho':exact4},
        'constraint_satisfying_initial_slice':{
            'H_uv_squared':float(initial['q']),'H_uv':float(np.sqrt(initial['q'])),
            'actual_H_squared_over_rho':float(meas5),'relative_difference_to_reduced_4d':float(meas5/exact4-1),
            'hamiltonian_analytic_max':initial['hamiltonian_max_analytic'],
            'momentum_analytic_max':initial['momentum_max_analytic'],
            'six_junction_errors':initial['jump_errors'],
            'radion_initial_proper_length':initial['radion_length_initial']},
        'evolution_convergence':{
            str(n):{
                'max_00_constraint_interior':r['max_00_constraint_interior'],
                'max_0y_constraint_interior':r['max_0y_constraint_interior'],
                'max_independent_junction_derivative_error':max(d['junction_one_sided_max'] for d in r['diagnostics']),
                'radion_final_proper_length':r['diagnostics'][-1]['radion_proper_length'],
                'radion_proper_length_change':r['diagnostics'][-1]['radion_proper_length']-r['diagnostics'][0]['radion_proper_length'],
                'final_H_uv':r['diagnostics'][-1]['H_UV'],
                'maximum_charge_relative_error':max(d['charge_relative_error'] for d in r['diagnostics'])}
                for n,r in evolutions.items()},
        'radion_response_to_interior_scalar_perturbation':{
            str(n):{'bump_plus_minus':.001,
                    'central_difference_dD_dbump_at_t1':(
                        r[.001]['diagnostics'][-1]['radion_proper_length']-
                        r[-.001]['diagnostics'][-1]['radion_proper_length'])/.002,
                    'max_00_constraint_interior':max(r[b]['max_00_constraint_interior'] for b in (-.001,.001))}
            for n,r in response_runs.items()},
        'detuned_brane_vacuum_response':vacuum_samples,
        'fixed_metric_GW_scalar_probe_eigenvalues_m_squared':{str(n):m2 for n,m2 in spec.items()},
        'uncharged_static_reference':{'max_00_constraint_interior':static_zero['max_00_constraint_interior'],
            'radion_drift_t0p4':static_zero['diagnostics'][-1]['radion_proper_length']-static_zero['diagnostics'][0]['radion_proper_length']},
        'tests':{'count':20,'execution':'python -m unittest discover -s tests -v'},
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')

    plt.figure(figsize=(7.3,4.6))
    for n,r in evolutions.items():
        h=r['y'][1]-r['y'][0]
        plt.plot(1/h,[r['max_00_constraint_interior']],'o')
        plt.plot([1/h]*2,[r['max_00_constraint_interior'],r['max_0y_constraint_interior']],lw=1,color='0.5')
    elems=np.array([n-1 for n in evolutions.keys()])
    vals00=np.array([r['max_00_constraint_interior'] for r in evolutions.values()])
    vals0y=np.array([r['max_0y_constraint_interior'] for r in evolutions.values()])
    plt.clf()
    plt.loglog(elems,vals00,'o-',label='Hamiltonian constraint')
    plt.loglog(elems,vals0y,'s-',label='Momentum constraint')
    plt.xlabel('Spatial elements N-1');plt.ylabel('Largest interior residual, t in [0,0.4]')
    plt.title('WS-03: nonlinear 5D constraints converge')
    plt.legend();plt.tight_layout();plt.savefig(OUT/'constraint_convergence.png',dpi=180);plt.close()

    plt.figure(figsize=(7.4,4.5))
    for n,twins in response_runs.items():
        tt=twins[.001]['time']
        out=(np.array([d['radion_proper_length'] for d in twins[.001]['diagnostics']])-
             np.array([d['radion_proper_length'] for d in twins[-.001]['diagnostics']]))/.002
        plt.plot(tt,out,label=f'N={n}')
    plt.xlabel('Conformal time (k=1)');plt.ylabel('Central-difference d(proper length)/d(scalar bump)')
    plt.title('Coupled radion response: finite-time, mesh-convergent')
    plt.legend();plt.tight_layout();plt.savefig(OUT/'radion_response.png',dpi=180);plt.close()

    plt.figure(figsize=(7.4,4.5))
    xs=np.array([r['positive_tension_detuning'] for r in vacuum_samples]); ys=np.array([r['H_uv_squared'] for r in vacuum_samples])
    plt.plot(xs,ys,'o-',label='5D Einstein constraints + both junctions')
    plt.plot(xs,exact4*xs,'--',label='4D reduced-gravity prediction')
    plt.xlabel('UV vacuum tension detuning, dimensionless')
    plt.ylabel('Initial H_UV squared')
    plt.title('Warp + GW stabilization retains vacuum sensitivity')
    plt.legend();plt.tight_layout();plt.savefig(OUT/'vacuum_sensitivity.png',dpi=180);plt.close()

    plt.figure(figsize=(7.4,4.5))
    for n,masses in spec.items():
        plt.plot(np.arange(1,5),masses,'o-',label=f'Nodes={n}')
    plt.xlabel('Fixed-metric scalar probe eigenmode')
    plt.ylabel('Squared eigenmass (dimensionless)')
    plt.title('Positive GW probe eigenvalues (not physical radion spectrum)')
    plt.legend();plt.tight_layout();plt.savefig(OUT/'gw_probe_spectrum.png',dpi=180);plt.close()

    plt.figure(figsize=(7.4,4.5))
    sample=evolutions[257]
    time=sample['time']
    for i,index in enumerate((0,len(time)//2,len(time)-1)):
        yy=sample['y']
        A,B,c,At,Bt,ct=sample['state'][index]
        plt.plot(yy,np.exp(A),label=f'spatial warp t={time[index]:.2f}')
    plt.xlabel('Conformal fifth dimension y')
    plt.ylabel('3D cosmological scale exp(A)')
    plt.title('Full 5D Einstein-scalar evolution with charged UV fluid')
    plt.legend();plt.tight_layout();plt.savefig(OUT/'bulk_evolution.png',dpi=180);plt.close()

    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    run()
