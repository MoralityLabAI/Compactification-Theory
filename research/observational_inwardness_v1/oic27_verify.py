"""Independent OIC-27 certificate verifier for n=6 with one flipped report.

Checks both complete public-history belief branches and real-world, adaptive
fault paths. Every validation remains active under `python -O`.
"""
from __future__ import annotations
import argparse
import json
from itertools import product
from pathlib import Path

N = 6
MAX_FLIPS = 1


def require(condition:bool,message:str) -> None:
    if not condition:raise AssertionError(message)


def clean_bit(d: int, z0: int, t: int, marker: int) -> int:
    return int((z0 + t * d) % N == marker)


def check_belief_tree(tree: dict, horizon: int) -> dict:
    """Check every feasible reported child and uniquely shift-labeled leaf."""
    root = frozenset((d, z0, 0) for d, z0 in product(range(N), repeat=2))
    checked_nodes = 0
    leaves = 0
    max_depth = 0
    reachable_reports = 0

    def visit(node: dict, possibilities: frozenset[tuple[int,int,int]], t: int):
        nonlocal checked_nodes, leaves, max_depth, reachable_reports
        require(isinstance(node,dict) and bool(possibilities),'invalid node / empty belief')
        checked_nodes += 1
        max_depth=max(max_depth,t)
        shift_values={d for d,_,_ in possibilities}
        if 'shift' in node:
            require(set(node)=={'shift'},'invalid leaf fields')
            require(shift_values=={node['shift']},f'ambiguous or wrong leaf shift at time {t}')
            leaves+=1
            return
        require(t<horizon,f'policy went past budget at {t}')
        require(set(node)=={'query','children'},'invalid internal node fields')
        address=node['query']
        require(type(address) is int and 0<=address<N,'invalid marker')
        expected={}
        for y in (0,1):
            next_states=set()
            for d,z0,used in possibilities:
                flip=int(clean_bit(d,z0,t,address)!=y)
                if used+flip<=MAX_FLIPS:
                    next_states.add((d,z0,used+flip))
            if next_states: expected[str(y)]=frozenset(next_states)
        require(set(node['children'])==set(expected),f'wrong report branches at time {t}')
        for y,child_states in expected.items():
            reachable_reports+=1
            visit(node['children'][y],child_states,t+1)
    visit(tree,root,0)
    return dict(nodes=checked_nodes,leaves=leaves,max_depth=max_depth,reachable_edges=reachable_reports)


def replay_worlds(tree: dict,horizon: int) -> dict:
    """Enumerate every true (d,z0) and online one-flip/no-flip path."""
    paths=0
    max_depth=0
    per_world={}
    def run(node: dict, d:int, z0:int,t:int,used:int):
        nonlocal paths,max_depth
        if 'shift' in node:
            require(node['shift']==d,f'false verdict for shift={d},phase={z0}')
            paths+=1;max_depth=max(max_depth,t)
            return 1
        require(t<horizon,'policy exceeded horizon during real-world replay')
        m=node['query']
        require(type(m) is int and 0<=m<N,'invalid sensor address')
        y=clean_bit(d,z0,t,m)
        possible=[(str(y),used)]
        if not used:possible.append((str(1-y),1))
        return sum(run(node['children'][b],d,z0,t+1,new_used) for b,new_used in possible)
    for d,z0 in product(range(N),repeat=2):
        per_world[f'{d}:{z0}']=run(tree,d,z0,0,0)
    return dict(paths=paths,max_depth=max_depth,worlds=len(per_world),min_paths_per_world=min(per_world.values()),max_paths_per_world=max(per_world.values()))


def verify(tree:dict,horizon:int) -> dict:
    return dict(belief=check_belief_tree(tree,horizon),real_world=replay_worlds(tree,horizon))


def main():
    p=argparse.ArgumentParser();p.add_argument('tree',type=Path);p.add_argument('--horizon',type=int,default=15)
    args=p.parse_args()
    tree=json.loads(args.tree.read_text(encoding='utf8'))
    print(json.dumps(verify(tree,args.horizon),sort_keys=True,indent=2))

if __name__=='__main__':main()
