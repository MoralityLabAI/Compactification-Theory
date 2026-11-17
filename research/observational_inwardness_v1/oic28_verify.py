"""Independent zero/one/error-layer verifier for OIC-28 n=7 adaptive policies.

Checks every possible public history branch and every true (shift, phase)
under zero or exactly one substitution. No use of search memo or symmetry.
"""
import argparse
import json
from pathlib import Path


def clean(n, d, z, t, m):
    return int((z + t*d) % n == m)


def verify_tree(tree, n=7, q=20):
    if not (2 <= n <= 100 and q >= 1):
        raise ValueError("invalid n/q")
    initial = frozenset((d,z,0) for d in range(n) for z in range(n))
    stats={"nodes":0,"leaves":0,"max_depth":0,"nonempty_children":0}
    def visit(node, candidates, t):
        if not isinstance(node, dict) or not candidates:
            raise AssertionError('invalid node or empty candidate set')
        stats['nodes']+=1
        stats['max_depth']=max(stats['max_depth'],t)
        shifts={d for d,_,_ in candidates}
        if 'shift' in node:
            if set(node)!={'shift'} or len(shifts)!=1 or node['shift']!=next(iter(shifts)):
                raise AssertionError('incorrect shift leaf')
            stats['leaves']+=1
            return
        if set(node)!={'query','children'} or t>=q:
            raise AssertionError('invalid query node/horizon')
        m=node['query']
        if type(m) is not int or not 0<=m<n:
            raise AssertionError('invalid marker')
        children=node['children']
        if not isinstance(children,dict):
            raise AssertionError('invalid children')
        expected=set()
        for y in (0,1):
            nxt=frozenset((d,z,e+(y!=clean(n,d,z,t,m))) for d,z,e in candidates if e+(y!=clean(n,d,z,t,m))<=1)
            if nxt:
                expected.add(str(y))
                stats['nonempty_children']+=1
                if str(y) not in children: raise AssertionError('missing feasible report branch')
                visit(children[str(y)],nxt,t+1)
        if set(children)!=expected: raise AssertionError('unexpected unreachable branch')
    visit(tree,initial,0)
    stats['true_world_fault_paths']=0
    for d in range(n):
      for z in range(n):
        for flip_at in range(-1,q):
          node=tree
          t=0
          while 'shift' not in node:
            if t>=q: raise AssertionError('nonterminal at horizon')
            m=node['query']
            observed=clean(n,d,z,t,m)^(t==flip_at)
            node=node['children'][str(observed)]
            t+=1
          if node['shift']!=d: raise AssertionError('misclassification')
          stats['true_world_fault_paths']+=1
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tree');p.add_argument('--n',type=int,default=7);p.add_argument('--q',type=int,default=20);a=p.parse_args()
    print(json.dumps(verify_tree(json.loads(Path(a.tree).read_text()),n=a.n,q=a.q),indent=2))
