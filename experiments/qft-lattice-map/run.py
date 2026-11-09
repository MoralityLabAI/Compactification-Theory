"""Reference CPU run (free Dirac sector only).  Writes JSON atomically."""
import argparse
import json
from pathlib import Path
import numpy as np
from model import (hamiltonian, axial_charge, commutator, resolvent_comparison,
                   projection_boundary_identity, correlation_matrix,
                   entropy_from_correlation, cft_entropy, w_integral, w_kernel)


def run(*, with_entropy=True):
    Ms=(2,6,10,14)
    k=0.7
    spectral=[]
    for M in Ms:
        eig=np.linalg.eigvalsh(hamiltonian(k,M))
        neg=eig[eig<0][-1]
        pos=eig[eig>0][0]
        H = hamiltonian(k,M)
        axial=commutator(H,axial_charge(k,M))
        eigenvalues, U = np.linalg.eigh(H)
        low = U[:,np.argsort(abs(eigenvalues))[:2]]
        axial_low = float(np.linalg.norm(low.conj().T @ axial @ low, 'fro'))
        spectral.append(dict(M=M,k=k,positive=float(pos),negative=float(neg),
                             target=k,max_error=float(max(abs(pos-k),abs(neg+k))),
                             axial_commutator_low_energy=axial_low,
                             axial_commutator_fro=float(np.linalg.norm(axial,'fro')),
                             axial_commutator_per_mode=float(np.linalg.norm(axial,'fro')/M),
                             boundary_identity=projection_boundary_identity(k,M,50),
                             resolvent=resolvent_comparison(k,M,50)))
    entropy=[]
    if with_entropy:
        for ell in (1,2,4):
            for M in Ms:
                C=correlation_matrix(64,ell,M,2048)
                S=entropy_from_correlation(C)
                S_tail=entropy_from_correlation(correlation_matrix(64,ell,M,4096))
                entropy.append(dict(L=64,ell=ell,M=M,entropy=S,
                                    qtail_difference=abs(S-S_tail),
                                    cft=cft_entropy(64,ell), min_eigen=float(np.linalg.eigvalsh(C)[0]),
                                    max_eigen=float(np.linalg.eigvalsh(C)[-1])))
    return dict(source='https://arxiv.org/abs/2611.05329v1',
                scope='free Dirac sector; no gauge-theory or anomaly replication',
                float_dtype='complex128',
                k=k, cutoffs=list(Ms), reference_flavors=50, schur_z=[0.3,0.7],
                integral_check={str(d):float(abs(w_integral(d)-w_kernel(d))) for d in (0,1,2,3,4,8)},
                spectral=spectral,entropy=entropy)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/reference.json')
    parser.add_argument('--skip-entropy',action='store_true')
    args=parser.parse_args()
    result=run(with_entropy=not args.skip_entropy)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    tmp.replace(path)
    print('M spectral_err axial_low_commutator resolvent_err')
    for row in result['spectral']:
        print(row['M'],round(row['max_error'],9),round(row['axial_commutator_low_energy'],8),round(row['resolvent']['error'],8))
    for row in result['entropy']:
        print('S',row['ell'],row['M'],round(row['entropy'],8),round(row['cft'],8))
