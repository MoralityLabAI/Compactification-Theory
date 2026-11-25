"""Generate WS-07 constraint audit, charge sweep, and tensor gravity figures."""
from pathlib import Path
from dataclasses import asdict,replace
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from self_gravity import Params,integrate,equilibrium,brane_actions,independent_einstein_residual
from tensor_modes import tensor_modes,tensor_speed_squared

ROOT=Path(__file__).parent
OUT=ROOT/'results'
OUT.mkdir(parents=True,exist_ok=True)

def run():
    p=Params()
    s=equilibrium(p,n=801)
    d=s['identities'];a=brane_actions(s)
    y=s['y'];A,N,chi,F,Ap,Np,cp,Fp=s['fields']
    denom=float(simpson(np.exp(-N+3*A),x=y))
    num=float(simpson(np.exp(N+A),x=y))
    speed=np.sqrt(num/denom)
    rows=[]
    for charge_amp in (.0,.02,.04,.06,.08,.10,.12):
        if charge_amp==0:
            y0,z0=integrate(0.,replace(p,F0=0.),points=401)
            row={'F0':0.0,'omega':None,'ir_NEC':0.0,
                 'graviton_speed':1.0,'charge':0.0,'hamiltonian_error':0.0}
        else:
            v=equilibrium(replace(p,F0=charge_amp),n=401)
            row={'F0':charge_amp,'omega':v['omega'],
                 'ir_NEC':v['identities']['ir_rho_plus_p'],
                 'graviton_speed':float(np.sqrt(tensor_speed_squared(v))),
                 'charge':v['identities']['bulk_charge'],
                 'hamiltonian_error':v['identities']['max_hamiltonian_constraint']}
        rows.append(row)
    tensor=[]
    for nn in (33,65,129,257):
        for pp in (0.0,.002,.02,.2):
            omega2,_=tensor_modes(s,nodes=nn,p3=pp,count=4)
            tensor.append({'nodes':nn,'p3':pp,'omega2_first_four':omega2.tolist()})
    # The equality uses Simpson quadrature at high spatial resolution.
    lhs=d['sumrule_lhs']
    rhs_simpson=float(simpson(s['source'],x=y))
    out={
      'title':'WS-07: self-gravitating stationary charged slab and its staticity obstruction',
      'status':'Classical dimensionless exact bulk Einstein-scalar branch, with explicit exotic IR brane ghost. Not viable dark energy.',
      'parameters':asdict(p),'omega':s['omega'],
      'equations':{
       'metric':'ds2=-exp(2N(y)) dt2+exp(2A(y)) dx3_squared+dy2',
       'charged_field':'Psi=F(y) exp(i omega t)',
       'local_identity':'d_y[exp(N+3A)*(Nprime-Aprimes)] = omega^2/M3 *exp(-N+3A)*F^2',
       'integral_identity':'sum_i exp(N_i+3 A_i)*(rho_i+p_i)+2 omega^2 integral exp(-N+3A)*F^2 dy = 0'
      },
      'bulk_diagnostics':d,'israel_branes':a,
      'max_exact_all_einstein_components':float(np.max(abs(independent_einstein_residual(s)))),
      'max_fd_all_einstein_components_interior':float(np.max(abs(independent_einstein_residual(s,finite_difference=True)[:,2:-2]))),
      'integrated_sumrule_simpson':{'lhs':lhs,'rhs':rhs_simpson,'abs_error':abs(lhs-rhs_simpson)},
      'tensor_low_p':{'uv_brane_light_speed':1.0,'c_t2':num/denom,
        'c_t':float(speed),'delta_ct':float(speed-1),
        'interpretation':'5D transverse traceless sector only; IR ghost implies scalar-sector instability'},
      'charge_sweep':rows,'tensor_modes':tensor,
      'limitation':'No full scalar-radion-ghost constrained spectrum; no stable NEC-satisfying stationary two-brane solution. No helium-3 data or cosmological constant fit.'
    }
    (OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    # Plot one: warps, relative lapse, charge density and source.
    fig,axes=plt.subplots(2,2,figsize=(10,7.5))
    axes[0,0].plot(y,A,label='Spatial A(y)')
    axes[0,0].plot(y,N,label='Temporal N(y)')
    axes[0,0].set(title='Self-gravitating charged warp',xlabel='y',ylabel='Warp exponent')
    axes[0,0].legend()
    axes[0,1].plot(y,N-A,label='N-A')
    axes[0,1].plot(y,Np-Ap,label="N'-A'")
    axes[0,1].set(title='Lorentz-violating geometric split',xlabel='y')
    axes[0,1].legend()
    axes[1,0].plot(y,F,label='Condensate amplitude')
    axes[1,0].plot(y,chi,label='GW scalar')
    axes[1,0].set(title='Bulk field profiles',xlabel='y')
    axes[1,0].legend()
    axes[1,1].plot(y,s['source'],label='Nonnegative charge source')
    axes[1,1].plot(y,np.exp(N+3*A)*(Np-Ap),label='Anisotropy boundary term')
    axes[1,1].set(title='Integrated anisotropy consistency identity',xlabel='y')
    axes[1,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'static_background.png',dpi=170);plt.close(fig)
    # Plot two: charge sweep with model viability diagnostics.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    xs=[x['F0'] for x in rows]; nec=[x['ir_NEC'] for x in rows]
    speed_minus=[x['graviton_speed']-1 for x in rows]
    axes[0].plot(xs,nec,'o-');axes[0].axhline(0,linestyle='--',linewidth=.8)
    axes[0].set(xlabel='UV condensate amplitude F(0)',ylabel='IR surface rho+p',title='Staticity requires IR NEC violation')
    axes[1].plot(xs,speed_minus,'o-');axes[1].axhline(0,linestyle='--',linewidth=.8)
    axes[1].set(xlabel='UV condensate amplitude F(0)',ylabel='c_tensor/c_UV_photon - 1',title='Tensor low-energy speed departure')
    fig.tight_layout();fig.savefig(OUT/'charge_sweep.png',dpi=170);plt.close(fig)
    # Plot three: direct spectrum at finite momentum.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for idx in (0,1,2):
        pp=np.array([.002,.02,.2]);levels=[]
        for pp0 in pp:
            levels.append(np.sqrt(tensor_modes(s,nodes=129,p3=float(pp0),count=3)[0][idx]))
        axes[0].plot(pp,levels,'o-',label=f'TT branch {idx}')
    axes[0].set(xlabel='3D Fourier momentum',ylabel='TT mode frequency',title='5D tensor frequencies')
    axes[0].legend()
    r=[z for z in tensor if z['p3']==0]
    axes[1].plot([x['nodes'] for x in r],[x['omega2_first_four'][1] for x in r],'o-')
    axes[1].set(xlabel='Finite element nodes',ylabel='1st massive TT omega squared',title='Spectral refinement')
    fig.tight_layout();fig.savefig(OUT/'tensor_modes.png',dpi=170);plt.close(fig)
    return out

if __name__=='__main__':
    results=run()
    print(json.dumps({'omega':results['omega'],'ir_rho_plus_p':results['bulk_diagnostics']['ir_rho_plus_p'],
      'hamiltonian_constraint':results['bulk_diagnostics']['max_hamiltonian_constraint'],
      'sumrule_integral_error':results['integrated_sumrule_simpson']['abs_error'],
      'tensor_speed':results['tensor_low_p']['c_t'],'figure_count':3},indent=2))
