"""OIC-22 fault accounting, exact Bayes recursions, and witness tests."""
import itertools
import unittest
from collections import defaultdict
from oic22_exact import FaultOracle,MixedFaultOracle
from oic22_reference import all_zero_ambiguity,compute


def brute_fixed(n,word,kind):
    """Independent leaf MAP objective from physical (d,z0) and fault locations."""
    Q=len(word)
    receipts=defaultdict(lambda:defaultdict(int))
    for d in range(n):
        for z in range(n):
            clean=tuple(int((z+t*d)%n==m) for t,m in enumerate(word))
            if kind=='mixed':
                receipts[clean][d]+=Q
            for t in range(Q):
                raw=list(clean)
                if kind=='flip' or kind=='mixed':raw[t]=1-raw[t]
                else:raw[t]=None
                receipts[tuple(raw)][d]+=1
    return sum(max(per_shift.values()) for per_shift in receipts.values())


class OIC22ExactTests(unittest.TestCase):
    def test_oracle_one_flip_vs_bruteforce_fixed_words(self):
        n,Q=3,4
        o=FaultOracle(n,Q,'flip')
        for word in itertools.product(range(n),repeat=Q):
            a,b=o.all,0
            # fixed query tree, retaining both branches and masks
            states=[(a,b)]
            for t,m in enumerate(word):
                fresh=[]
                for c0,c1 in states:
                    for y,x,z in o.successors(c0,c1,t,m):
                        if x|z:fresh.append((x,z))
                states=fresh
            dynamic=sum(o.terminal(c1) for _,c1 in states)
            self.assertEqual(dynamic,brute_fixed(n,word,'flip'))

    def test_mixed_fault_oracle_matches_independent_enumeration(self):
        n,Q=3,4
        o=MixedFaultOracle(n,Q)
        for word in itertools.product(range(n),repeat=Q):
            states=[(o.all,0)]
            for t,m in enumerate(word):
                nexts=[]
                for c0,c1 in states:
                    for _,a,b in o.successors(c0,c1,t,m):
                        if a|b:nexts.append((a,b))
                states=nexts
            dynamic=sum(o.terminal_joint(a,b) for a,b in states)
            self.assertEqual(dynamic,brute_fixed(n,word,'mixed'))

    def test_marked_erasure_oracle_against_bruteforce(self):
        n,Q=3,4
        o=FaultOracle(n,Q,'erase')
        for word in itertools.product(range(n),repeat=Q):
            states=[(o.all,0)]
            for t,m in enumerate(word):
                ns=[]
                for c0,c1 in states:
                    for y,a,b in o.successors(c0,c1,t,m):
                        if a|b:ns.append((a,b))
                states=ns
            dynamic=sum(o.terminal(c1) for _,c1 in states)
            self.assertEqual(dynamic,brute_fixed(n,word,'erase'))

    def test_adaptive_advantage_and_pinned_values(self):
        rows=compute()['rows']
        self.assertEqual([(r['n'],r['half_clean_half_flip_numerator']) for r in rows],[(3,71),(4,147),(5,262)])
        self.assertEqual([(r['n'],r['uniform_one_flip_numerator']) for r in rows],[(3,34),(4,73),(5,126)])
        self.assertEqual([(r['n'],r['uniform_one_erasure_numerator']) for r in rows],[(3,45),(4,90),(5,154)])
        o=MixedFaultOracle(3,4)
        best_fixed=max(brute_fixed(3,w,'mixed') for w in itertools.product(range(3),repeat=4))
        self.assertGreaterEqual(o.optimal_value(0,o.all,0),best_fixed)

    def test_one_fault_masks_contain_actual_hidden_configuration(self):
        for n in (3,4):
            o=FaultOracle(n,n+2,'flip');Q=o.Q
            words=((0,)*Q,tuple(t%n for t in range(Q)))
            for word in words:
                for d,z,pos in itertools.product(range(n),range(n),range(Q)):
                    a,b=o.all,0
                    for t,m in enumerate(word):
                        y=int((z+t*d)%n==m)^(t==pos)
                        a,b=o.after(a,b,t,m,y)
                    self.assertTrue(b&(1<<(d*n+z)))
                    self.assertFalse(a&(1<<(d*n+z)))

    def test_lower_bound_pigeonhole_counterexamples(self):
        for n in (3,4,5,6):
            for e,r in ((0,0),(1,0),(0,1),(1,1),(2,0)):
                Q=n*(e+1)+r-1
                witness=all_zero_ambiguity(n,Q,e,r)
                self.assertTrue(witness['compatible_every_shift'])
                self.assertTrue(all(x['flips_required']<=e for x in witness['witnesses'].values()))

    def test_any_fixed_word_all_zero_indistinguishability(self):
        n,Q,e=3,5,1
        for markers in itertools.product(range(n),repeat=Q):
            for d in range(n):
                counts=[sum((z+t*d)%n==markers[t] for t in range(Q)) for z in range(n)]
                self.assertLessEqual(min(counts),e)

    def test_budget_and_validation(self):
        for n in (3,4,5):
            Q=n+2
            self.assertEqual(Q*(1+(n-1).bit_length()),compute()['rows'][n-3]['total_online_bits'])
        with self.assertRaises(ValueError):FaultOracle(3,5,'bad')
        with self.assertRaises(ValueError):all_zero_ambiguity(3,2,0,3)

if __name__=='__main__':unittest.main()
