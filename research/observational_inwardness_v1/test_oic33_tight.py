"""OIC-33 mandatory-address tests; no network or model calls."""
from __future__ import annotations
import unittest
from itertools import product
from oic23_exact import ExactFaultGame
from oic33_tight_reference import forced_addresses, ForcedGame


class TightPairTests(unittest.TestCase):
    def test_small_complete_feasibility(self):
        # Original complete Bellman oracle is independent of new pruning.
        for n,q in ((2,5),(2,6),(3,7),(3,8),(4,5),(4,6)):
            standard=ExactFaultGame(n,q,1,0,node_cap=400000)
            extra=ForcedGame(n,q,node_cap=400000)
            self.assertEqual(standard.solve(),extra.solve(),(n,q))
            if extra._value_cache.get((0,extra.root)):
                self.assertTrue(extra.verify_tree(extra.certificate())>0)

    def test_forced_actions_exact_local_property(self):
        # Enumerate candidate masks independent of the new implementation and
        # check that every excluded address has a child with an OIC-T44 pair.
        import random
        def direct_bad_pair(n,q,t,layers):
            union=layers[0]|layers[1]
            worlds=[i for i in range(n*n) if union&(1<<i)]
            for j,a in enumerate(worlds):
                da,za=divmod(a,n)
                for b in worlds[j+1:]:
                    db,zb=divmod(b,n)
                    if da==db:continue
                    spare=((layers[0]>>a)&1)+((layers[0]>>b)&1)
                    future=sum((za+s*da)%n!=(zb+s*db)%n for s in range(t,q))
                    if future<=spare:return True
            return False
        checked=0
        for n,q,t in ((3,8,6),(4,10,8)):
            oracle=ExactFaultGame(n,q,1,0)
            samples=[]
            if n==3:
                for codes in product((0,1,2),repeat=n*n):
                    samples.append((sum(1<<i for i,c in enumerate(codes) if c==1),
                                    sum(1<<i for i,c in enumerate(codes) if c==2)))
            else:
                rng=random.Random(1349)
                for _ in range(2500):
                    codes=[rng.randrange(3) for _ in range(n*n)]
                    samples.append((sum(1<<i for i,c in enumerate(codes) if c==1),
                                    sum(1<<i for i,c in enumerate(codes) if c==2)))
            for state in samples:
                allowed,reason=forced_addresses(n,q,t,state)
                if reason=='pairwise':continue
                for m in range(n):
                    if allowed&(1<<m):continue
                    children=[oracle.next_state(state,t,m,y) for y in (0,1)]
                    self.assertTrue(any(child is not None and direct_bad_pair(n,q,t+1,child)
                                        for child in children),(n,q,t,m,state))
                    checked+=1
        self.assertGreater(checked,100)

    def test_single_tight_pair_forces_one_of_two_phases(self):
        n,q,t=5,4,2
        # Different shifts, initially different phase positions.
        # Other candidates absent. No sign/shift oracle is part of the test.
        a,b=(0*n+0),(1*n+2)
        allowed,why=forced_addresses(n,q,t,((1<<a)|(1<<b),0))
        self.assertIn(why,('allowed','pairwise'))
        if why=='allowed':self.assertNotEqual(allowed,0)

    def test_affine_gauge_equivariance(self):
        n,q=5,8
        base=ExactFaultGame(n,q,1,0)
        cases=[base.root]
        for t in range(3):
            newer=[]
            for state in cases[:10]:
                for m in range(n):
                    for y in (0,1):
                        child=base.next_state(state,t,m,y)
                        if child is not None:newer.append(child)
            cases=newer
            for zero,one in cases[:100]:
                allowed,why=forced_addresses(n,q,t+1,(zero,one))
                for u in range(1,n):
                    for c in range(n):
                        def perm(mask):
                            ans=0
                            for d,z in product(range(n), repeat=2):
                                if mask&(1<<(d*n+z)):
                                    ans|=1<<(((u*d)%n)*n+(u*z+c)%n)
                            return ans
                        shifted,other=forced_addresses(n,q,t+1,(perm(zero),perm(one)))
                        mask=sum(1<<((u*m+c)%n) for m in range(n) if allowed&(1<<m))
                        self.assertEqual(mask,shifted,(n,q,t,u,c,why,other))

if __name__=='__main__':unittest.main()
