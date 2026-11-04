"""QO-13 exact independent QO-07 history-replay tests."""
import unittest
from fractions import Fraction as Q
from itertools import product
from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger, certificate
from research.ingression_dynamics_v1.certificate_reachability import optimal_certificate_reachability, select_certificate_probe

def brute(ledger,k,tau,h):
    if certificate(ledger.snapshot(),k,tau)["decision"]!="undetermined": return Q(1)
    if not h: return Q(0)
    best=Q(0)
    for i,a in product(range(ledger.n),range(ledger.d)):
        p=Q(1+ledger.hits[i][a],2+ledger.counts[i][a]);values=[]
        for bit in (0,1):
            child=ProbeLedger(ledger.n,ledger.d,ledger.alpha,ledger.bits)
            try:
                for rec in ledger.receipts: child.observe(*rec)
                child.observe(i,a,bit)
                values.append(brute(child,k,tau,h-1))
            except ValueError: values.append(Q(0))
        best=max(best,(1-p)*values[0]+p*values[1])
    return best

class CertificateReachabilityTests(unittest.TestCase):
    def witness(self):
        x=ProbeLedger(2,1,Q(1,20),10)
        for _ in range(18):
            x.observe(0,0,0);x.observe(1,0,1)
        return x
    def test_19_over_21_delayed_certificate_witness(self):
        x=self.witness()
        vals=[optimal_certificate_reachability(x,1,Q(1,5),h) for h in range(4)]
        self.assertEqual([v["predicted_certificate_probability"] for v in vals[:3]],
                         [Q(0),Q(0),Q(19,21)])
        self.assertEqual(vals[2]["first_cell"],(0,0))
        self.assertEqual(vals[2]["predicted_certificate_probability"],brute(x,1,Q(1,5),2))
        self.assertEqual([v["predicted_certificate_probability"] for v in vals],
                         sorted(v["predicted_certificate_probability"] for v in vals))
        cell,info=select_certificate_probe(x,1,Q(1,5),horizon=3)
        self.assertEqual(cell,(0,0))
        self.assertEqual(info["mode"],"positive_certificate_probability")
    def test_independent_oracle_small_cases(self):
        for n in (2,3):
            for pattern in product((0,1),repeat=n):
                x=ProbeLedger(n,1,Q(1,20),10)
                for i in range(n):
                    for _ in range(4):x.observe(i,0,pattern[i])
                scores=[]
                for h in (0,1,2):
                    fast=optimal_certificate_reachability(x,1,Q(1,5),h)
                    value=brute(x,1,Q(1,5),h)
                    self.assertEqual(fast["predicted_certificate_probability"],value)
                    scores.append(value)
                self.assertEqual(scores,sorted(scores))
    def test_gate_avoids_costly_early_planning(self):
        x=ProbeLedger(3,2)
        _,rec=select_certificate_probe(x,1,Q(1,5),horizon=3)
        self.assertEqual(rec["mode"],"gated_insufficient_addressed_coverage")
        self.assertEqual(rec["effective_horizon"],2)
        self.assertEqual(rec["decision_gap_evaluations"],0)
    def test_certified_state_is_absorbing(self):
        x=ProbeLedger(1,1,Q(1,20),10)
        for _ in range(28):x.observe(0,0,0)
        for h in range(4):
            result=optimal_certificate_reachability(x,1,Q(1,5),h)
            self.assertEqual(result["predicted_certificate_probability"],1)
            self.assertIsNone(result["first_cell"])
    def test_inputs(self):
        x=ProbeLedger(2,1)
        with self.assertRaises(ValueError):optimal_certificate_reachability(x,1,Q(1,5),4)
        with self.assertRaises(ValueError):optimal_certificate_reachability(x,1,Q(1,5),2,max_cells=1)
        with self.assertRaises(ValueError):select_certificate_probe(x,1,Q(1,5),near_threshold=-1)

if __name__=="__main__": unittest.main()
