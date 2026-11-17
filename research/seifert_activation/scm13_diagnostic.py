"""SCM-13 sampled-only Fourier screen of RMP cyclic activation receipts.

The first harmonic of the measured path is the affine response surrogate.
The residual maximum is measured ONLY on sampled angles, so all returned
conditional certificates remain empirical diagnostics, never source-derived
continuous-path proofs. Use source-certified Hessian bounds and SCM-13 affine
disk certificate separately before making topology claims.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scm13_affine_disk import affine_disk_certificate
from winding import pairwise_windings
from knot import InvalidBraidTrace


def _finite_points(frames):
    if not isinstance(frames,list) or len(frames)<9:
        raise ValueError('Require at least 8 angular intervals')
    n=len(frames[0])
    if n<2:
        raise ValueError('Need at least two observed token trajectories')
    points=[]
    for frame in frames:
        if len(frame)!=n:
            raise ValueError('Wrong number of observed strands')
        row=[]
        for p in frame:
            if len(p)!=2 or any(not isinstance(v,(float,int)) or not math.isfinite(v) for v in p):
                raise ValueError('Nonfinite or malformed frame')
            row.append([float(p[0]),float(p[1])])
        points.append(row)
    if any(math.dist(a,b)>1e-6 for a,b in zip(points[0],points[-1])):
        raise ValueError('Observed trajectories fail endpoint closure')
    return points


def screen_frames(frames, *, numeric_guard=1e-8) -> dict:
    """Fit a + B(cos theta,sin theta) from uniform samples, report sampled error.

    The fit measures response at the supplied intervention radius. Its supplied
    B already includes radius; the affine-disk certificate therefore uses r=1.
    Hessian enclosures are left unverified. The row is diagnostic even if the
    fit happens to match the observed sampled braid exactly.
    """
    frames=_finite_points(frames)
    N=len(frames)-1
    n=len(frames[0]);a=[];B=[];err=[]
    # Uniform samples have exact discrete first harmonic orthogonality for N>2.
    for strand in range(n):
        vals=[frames[k][strand] for k in range(N)]
        center=[sum(v[c] for v in vals)/N for c in range(2)]
        bcos=[2*sum(v[c]*math.cos(2*math.pi*k/N) for k,v in enumerate(vals))/N for c in range(2)]
        bsin=[2*sum(v[c]*math.sin(2*math.pi*k/N) for k,v in enumerate(vals))/N for c in range(2)]
        radial_errors=[]
        for k,v in enumerate(vals):
            theta=2*math.pi*k/N
            radial_errors.append(math.dist(v,[center[c]+bcos[c]*math.cos(theta)+bsin[c]*math.sin(theta) for c in range(2)]))
        a.append(center)
        B.append([[bcos[0],bsin[0]],[bcos[1],bsin[1]]])
        err.append(max(radial_errors)+numeric_guard)
    test=affine_disk_certificate(a,B,1,err,margin=str(numeric_guard))
    # The following independent winding check validates the sampled polygon
    # against the *hypothesized* affine surrogate; never upgrades to a proof.
    try:
        obs={f'{i},{j}':v for (i,j),v in pairwise_windings(frames).items()}
        predicted={key:v['pair_winding'] if v['certified'] else None for key,v in test['per_pair'].items()}
        comparison={key:(obs[key]==predicted[key] if predicted[key] is not None else None) for key in obs}
    except InvalidBraidTrace as e:
        obs={'status':'rejected','reason':str(e)}
        comparison=None
    return {
        'schema':'scm13/sampled-affine-screen/v1',
        'status':'empirical_only_unverified_full_circle_error',
        'sample_count':N,
        'n_strands':n,
        'affine_fit':{'bases':a,'matrices':B,'max_sample_residual':err},
        'affine_conditional_check':test,
        'polygonal_pair_windings':obs,
        'sampled_agreement':comparison,
        'source_hessian_certified':False,
    }


def audit_capture(path:Path,out:Path):
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing audit')
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        cap=row.get('capture',{})
        visits=cap.get('frames_by_visit')
        if not isinstance(visits,list):
            raise ValueError('Expected an SCM-04 cyclic capture with frames_by_visit')
        screens=[]
        for visit in visits:
            try:
                screens.append(screen_frames(visit))
            except (ValueError,ArithmeticError) as e:
                screens.append({'status':'refused','reason':str(e)})
        result.append({'task_id_sha256':row.get('task_id_sha256'),
                       'model':row.get('model'),'source_type':row.get('source_type'),
                       'screen_per_visit':screens})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as writer:
        json.dump({'schema':'scm13/sampled-receipt-audit/v1','rows':result,
                   'claim':'sampled_only; no continuous or trained-model source certificate'},
                  writer,indent=2,sort_keys=True,allow_nan=False)
        writer.write('\n')
    return len(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({'audited_rows':audit_capture(args.input,args.out)}))

if __name__=='__main__':
    main()
