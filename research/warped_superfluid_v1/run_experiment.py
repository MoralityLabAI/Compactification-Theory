"""Generate the standalone RS / superfluid-vacuum toy benchmark report.

Usage from this folder: python run_experiment.py
Dependencies: numpy scipy matplotlib
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import jn_zeros
from warped_superfluid import (
    rs_exact_scaled_roots, rs_mode_ir_fraction, fe_masses,
    DomainWall, integrate_ds_wall, ads_ds_warp, kessence_fluid,
    finite_density_anisotropy,
)

HERE = Path(__file__).parent
OUT = HERE/'results'
OUT.mkdir(exist_ok=True)


def run():
    table = []
    for k in (0.0, 1.0):
        L = 4.0
        if k:
            exact = rs_exact_scaled_roots(k*L, 3) * k*np.exp(-k*L)
        else:
            exact = np.pi*np.arange(1,4)/L
        for n in (65,129,257):
            y = np.linspace(0,L,n)
            modes = fe_masses(y,k*y,3)[1:]
            errs = (modes-exact)/exact
            table.append({"kL":float(k*L),"nodes":n,
                          "masses":modes.tolist(),
                          "relative_errors":errs.tolist(),
                          "max_absolute_relative_error":float(np.max(np.abs(errs)))})

    kLbig = 35.0
    bigroots = rs_exact_scaled_roots(kLbig,4)
    ir = [rs_mode_ir_fraction(kLbig,float(r)) for r in bigroots]
    bessel_zeros = jn_zeros(1,4)
    assert np.max(np.abs(bigroots-bessel_zeros)) < 1e-10

    y = np.linspace(0,4.0,257)
    rs = fe_masses(y,y,3)
    naive = fe_masses(y,y,3,naive_speed_only=True)

    wall=DomainWall()
    back=fe_masses(y,wall.warp(y),3)
    exact_err=wall.exact_residual(y)
    y0,state0,res0=integrate_ds_wall(wall,4.0,0.0,321)
    yh,stateh,resh=integrate_ds_wall(wall,4.0,0.005,321)
    wall_ode_err=float(np.max(np.abs(state0[0]-wall.warp(y0))))
    chi_ode_err=float(np.max(np.abs(state0[2]-wall.chi(y0))))
    # Pure AdS5 dS4 curvature relation and H->0 RS consistency.
    dsw=ads_ds_warp(y,0.005)
    dsw_approx_err=float(np.max(np.abs(ads_ds_warp(y,1e-6)-y)))

    reds=np.linspace(0,3,121)
    curves={}
    for r0 in [1e-4,1e-3,1e-2]:
        rho,p,w,cs2=kessence_fluid(reds,r0)
        curves[str(r0)]={"w0":float(w[0]),"w_z1":float(np.interp(1,reds,w)),
                         "w_z3":float(w[-1]),"cs2":float(cs2[0])}
    report = {
        "status":"dimensionless prototype; no helium-3 or cosmological data fits",
        "conventions":"M5^3=k=1 unless stated; y dimensionless; L=4 for FE",
        "fe_convergence":table,
        "large_warp":{"kL":kLbig,"scaled_roots":bigroots.tolist(),
                      "bessel_J1_zeros":bessel_zeros.tolist(),
                      "fraction_norm_in_IR_last_ln10_over_k":ir,
                      "actual_m_over_k":(bigroots*np.exp(-kLbig)).tolist()},
        "principal_symbol_control":{"true_RS_masses":rs[1:].tolist(),
                                  "speed_only_masses":naive[1:].tolist(),
                                  "relative_differences":((naive[1:]-rs[1:])/rs[1:]).tolist()},
        "backreaction":{"exact_wall_masses":back[1:].tolist(),
                        "rs_masses":rs[1:].tolist(),
                        "relative_mode_shift":((back[1:]-rs[1:])/rs[1:]).tolist(),
                        "exact_residuals_constraint_Asecond_scalar":list(exact_err),
                        "zero_H_numerical_constraint_residual":res0,
                        "H_0_numeric_A_error":wall_ode_err,
                        "H_0_numeric_chi_error":chi_ode_err,
                        "H_0p005_numerical_constraint_residual":resh,
                        "H_0p005_final_A":float(stateh[0,-1]),
                        "H_0p005_final_chi":float(stateh[2,-1]),
                        "pure_AdS_dS_slice_AatL":float(dsw[-1]),
                        "pure_AdS_dS_slice_Hto0_max_A_error":dsw_approx_err},
        "finite_density_stress":{"X_0p5_alpha_1_Tt_minus_Ti":finite_density_anisotropy(0.5),
                                 "constraint":"XP_X=0 for exactly 4D Lorentz-invariant warped slices"},
        "kessence":{"P":"-V0+alpha X**2","r0":"alpha X0**2/V0",
                    "curves":curves,"kinetic_density_scaling":"a^-4"},
    }
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')

    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for kl,marker in [(0.0,'o'),(4.0,'s')]:
        rows=[t for t in table if t['kL']==kl]
        axes[0,0].loglog([t['nodes']-1 for t in rows],
                         [t['max_absolute_relative_error'] for t in rows],
                         marker+'-',label=f'kL={kl:.0f}')
    axes[0,0].set(xlabel='Finite elements',ylabel='Maximum relative KK mass error',title='Consistent FE convergence')
    axes[0,0].legend()
    axes[0,1].plot(np.arange(1,4),rs[1:], 'o-',label='RS operator')
    axes[0,1].plot(np.arange(1,4),naive[1:], 's-',label='Same local speed only')
    axes[0,1].plot(np.arange(1,4),back[1:], 'x-',label='Backreacted wall')
    axes[0,1].set(xlabel='Mode index',ylabel='Dimensionless mass',title='Measure and backreaction matter')
    axes[0,1].legend(fontsize=8)
    axes[1,0].plot(y, y, label='RS A(y)')
    axes[1,0].plot(y,wall.warp(y),label='Exact scalar-wall A(y)')
    axes[1,0].plot(yh,stateh[0],label='dS4 scalar wall H=0.005')
    axes[1,0].set(xlabel='Extra-dimensional coordinate y',ylabel='Warp exponent A(y)',title='5D Einstein-scalar solutions')
    axes[1,0].legend(fontsize=8)
    for r0 in [1e-4,1e-3,1e-2]:
        axes[1,1].plot(reds,kessence_fluid(reds,r0)[2],label=f'r0={r0:g}')
    axes[1,1].set(xlabel='Redshift z',ylabel='Equation of state w(z)',title='4D kinetic superfluid toy')
    axes[1,1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'diagnostics.png',dpi=180)
    plt.close(fig)
    print(json.dumps(report,indent=2))
    return report

if __name__ == '__main__':
    run()
