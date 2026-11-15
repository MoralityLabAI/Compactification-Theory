"""SMC-13: independent Sudoku solution family with an exact orbit obstruction.

The upstream checkpoint's full-grid generator only permutes a canonical
cyclic Sudoku solution. Its alternating 2x2 rectangle count is zero.
Randomized backtracking here produces valid complete solutions with a
STRICTLY POSITIVE count, proving exclusion from that entire source orbit.

All public split seeds are fixed before teacher training, and no source
puzzle (or derived full target) is used in more than one cohort.
"""
import hashlib
import json
import random
from itertools import combinations

BASES={
    'teacher_train':202613000000,
    'teacher_val':202613010000,
    'feature_calibration':202613020000,
    'test_novel_30':202613030000,
    'test_novel_34':202613040000,
    'test_canonical_30':202613050000,
}
SIZES={
    'teacher_train':192,'teacher_val':64,
    'feature_calibration':64,'test_novel_30':64,
    'test_novel_34':64,'test_canonical_30':64,
}
HOLES={k:(34 if k=='test_novel_34' else 30) for k in BASES}


def base_solution():
    return [(3*(r%3)+r//3+c)%9+1 for r in range(9) for c in range(9)]


def rectangle_count(grid):
    """Digit/row/column/transpose invariant number of 2x2 intercalates."""
    if len(grid)!=81:
        raise ValueError('expected a 9x9 filled Sudoku')
    total=0
    for r in range(9):
        for s in range(r+1,9):
            for c in range(9):
                a,b=grid[9*r+c],grid[9*s+c]
                for d in range(c+1,9):
                    if a!=b and a==grid[9*s+d] and b==grid[9*r+d]:
                        total+=1
    return total


def valid_solution(grid):
    if len(grid)!=81 or any(not isinstance(v,int) or not 1<=v<=9 for v in grid):
        return False
    target=set(range(1,10))
    for r in range(9):
        if set(grid[9*r:9*r+9])!=target:return False
        if {grid[9*s+r] for s in range(9)}!=target:return False
    for br in (0,3,6):
        for bc in (0,3,6):
            if {grid[9*(br+dr)+bc+dc] for dr in range(3) for dc in range(3)}!=target:
                return False
    return True


def new_full_solution(seed):
    """Seeded randomized MRV Sudoku completion, not a cyclic-grid orbit."""
    rng=random.Random(seed)
    grid=[0]*81; rm=[0]*9;cm=[0]*9;bm=[0]*9
    # Fill independent diagonal boxes to diversify the solution structure.
    for bx in (0,4,8):
        digits=list(range(1,10));rng.shuffle(digits)
        for j,digit in enumerate(digits):
            r=3*(bx//3)+j//3;c=3*(bx%3)+j%3
            grid[9*r+c]=digit;bit=1<<digit
            rm[r]|=bit;cm[c]|=bit;bm[bx]|=bit
    def dfs():
        best=-1;options=[];size=10
        for idx in range(81):
            if grid[idx]:continue
            r,c=divmod(idx,9);bx=3*(r//3)+c//3
            mask=0x3FE & ~(rm[r]|cm[c]|bm[bx])
            n=mask.bit_count()
            if n==0:return False
            if n<size:
                size=n;best=idx
                options=[v for v in range(1,10) if mask&(1<<v)]
                if size==1:break
        if best<0:return True
        rng.shuffle(options)
        r,c=divmod(best,9);bx=3*(r//3)+c//3
        for v in options:
            bit=1<<v
            grid[best]=v;rm[r]|=bit;cm[c]|=bit;bm[bx]|=bit
            if dfs():return True
            grid[best]=0;rm[r]^=bit;cm[c]^=bit;bm[bx]^=bit
        return False
    if not dfs() or not valid_solution(grid):
        raise AssertionError('randomized Sudoku completion failed')
    return grid


def candidate_mask(grid,i):
    r,c=divmod(i,9);used=0
    for k in range(9):
        used|=1<<grid[9*r+k]
        used|=1<<grid[9*k+c]
    for rr in range(3*(r//3),3*(r//3)+3):
        for cc in range(3*(c//3),3*(c//3)+3):
            used|=1<<grid[9*rr+cc]
    return (~used)&0x3FE


def count_solutions(grid,limit=2):
    """Bounded independent exact Sudoku solver; returns 0,1 or >=2."""
    if limit<1:raise ValueError('positive solution limit')
    row=list(grid)
    if len(row)!=81:raise ValueError('invalid Sudoku length')
    def dfs(limit_left):
        best=-1;mask=0;n=10
        for i in range(81):
            if row[i]:continue
            m=candidate_mask(row,i);pop=m.bit_count()
            if pop==0:return 0
            if pop<n:
                best=i;mask=m;n=pop
                if n==1:break
        if best<0:return 1
        total=0
        while mask:
            bit=mask & -mask
            row[best]=bit.bit_length()-1
            total+=dfs(limit_left-total)
            row[best]=0
            if total>=limit_left:return total
            mask^=bit
        return total
    return dfs(limit)


def dig_unique(solution,seed,holes):
    rng=random.Random(seed)
    if not valid_solution(solution) or not 1<=holes<=81:
        raise ValueError('valid solution and positive hole count required')
    puzzle=list(solution);filled=0
    for i in rng.sample(range(81),81):
        if filled==holes:break
        save=puzzle[i];puzzle[i]=0
        if count_solutions(puzzle,2)==1:filled+=1
        else:puzzle[i]=save
    if filled!=holes or count_solutions(puzzle,2)!=1:
        raise ValueError('could not generate unique specified-hole puzzle')
    return puzzle


def cohort(key,n=None):
    """Entire cohort and exact manifest, before any teacher fit is allowed."""
    if key not in BASES:raise ValueError('unknown source-registered split')
    n=SIZES[key] if n is None else n
    if not isinstance(n,int) or not 1<=n<=SIZES[key]:
        raise ValueError('only fixed prefix of registered cohort allowed')
    canonical=(key=='test_canonical_30')
    if canonical:
        # The code is source-pinned in the existing MIT-licensed LSV-05 module.
        import sys
        from pathlib import Path
        p=Path(__file__).resolve().parents[1]/'levin_sieve_v1'/'pretrained_trm'
        sys.path.insert(0,str(p))
        from sudoku_generator import generate_puzzle
    samples=[]
    for i in range(n):
        seed=BASES[key]+i
        if canonical:
            x,y,k=generate_puzzle(n_holes=HOLES[key],seed=seed)
            if k!=HOLES[key]:raise ValueError('canonical under-dug')
        else:
            y=new_full_solution(seed)
            if rectangle_count(y)<=0:
                raise ValueError('novel target unexpectedly in canonical orbit')
            x=dig_unique(y,seed+1000000,HOLES[key])
        if not valid_solution(y) or sum(v==0 for v in x)!=HOLES[key]:
            raise AssertionError('invalid puzzle cohort')
        if any(u and u!=v for u,v in zip(x,y)):
            raise AssertionError('clue disagrees with target')
        if count_solutions(x,2)!=1:
            raise AssertionError('puzzle solution is not unique')
        invariant=rectangle_count(y)
        if canonical and invariant!=0:
            raise AssertionError('source orbit must have zero intercalates')
        samples.append({'seed':seed,'x':x,'target':y,'holes':HOLES[key],
                        'rectangles':invariant})
    return samples


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def build_rosters(quick=False):
    rosters={k:cohort(k,4 if quick else None) for k in BASES}
    targets=[tuple(row['target']) for rows in rosters.values() for row in rows]
    puzzles=[tuple(row['x']) for rows in rosters.values() for row in rows]
    if len(set(targets))!=len(targets) or len(set(puzzles))!=len(puzzles):
        raise AssertionError('solution/puzzle duplicated across cohort groups')
    manifest={
      'schema':'SMC-13/intercalate-disjoint-Sudoku-roster-v1',
      'registered_sizes':{k:len(v) for k,v in rosters.items()},
      'registered_seed_bases':BASES,
      'registered_holes':HOLES,
      'canonical_intercalates':rectangle_count(base_solution()),
      'cohorts':{
       k:{'sha256':fingerprint(v),
          'min_intercalates':min(r['rectangles'] for r in v),
          'max_intercalates':max(r['rectangles'] for r in v),
          'mean_intercalates':sum(r['rectangles'] for r in v)/len(v),
          'n':len(v)}
       for k,v in rosters.items()},
      'disjoint_full_solution_targets':True,
      'disjoint_puzzle_inputs':True,
      'source_orbit_disjoint_novel_family':True,
    }
    return rosters,manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--output')
    a=p.parse_args()
    _,receipt=build_rosters(quick=a.smoke)
    encoded=json.dumps(receipt,indent=2,sort_keys=True)+'\n'
    if a.output:
        from pathlib import Path
        Path(a.output).write_text(encoded)
    print(encoded)
