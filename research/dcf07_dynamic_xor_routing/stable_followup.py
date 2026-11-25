"""Numerical repair after DCF-07 follow-up: normalize factor marginals on repartition."""
import numpy as np
import gym
import followup
_original=gym.repartition

def stable_repartition(ps,old,new):
    # Repartitioning repeatedly multiplies factor masses; normalize inputs and outputs.
    ps=[p/p.sum(axis=1,keepdims=True) for p in ps]
    q=_original(ps,old,new)
    out=[p/p.sum(axis=1,keepdims=True) for p in q]
    for p in out:
        assert np.isfinite(p).all() and np.min(p)>=-1e-12
        assert np.max(np.abs(p.sum(axis=1)-1))<1e-12
    return out

if __name__=='__main__':
    from pathlib import Path
    gym.repartition=stable_repartition
    followup.repartition=stable_repartition
    followup.main(Path(__file__).parent)
    import json,math
    r=json.loads(Path(__file__).with_name('followup_receipt.json').read_text())
    assert all(math.isfinite(x['kl']) and x['kl']>=-1e-14 for a in r for x in a['rows'])
