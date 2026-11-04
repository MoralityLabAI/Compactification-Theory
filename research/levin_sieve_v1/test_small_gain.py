"""Reproducible exact-rational tests for the Levin-sieve repair certificate."""
import unittest
from fractions import Fraction as Q
from small_gain import Certificate, audit_trace, demo, mode_envelope


class RepairCertificateTests(unittest.TestCase):
    def setUp(self):
        self.gains = ((Q(0), Q(3)), (Q(1, 10), Q(0)))

    def test_resolvent_synthesis_works_when_row_sum_fails(self):
        cert = Certificate.synthesize(self.gains)
        self.assertEqual(cert.weights, (Q(40,7), Q(11,7)))
        self.assertEqual(cert.rho, Q(33,40))
        cert.validate()
        with self.assertRaises(ValueError):
            Certificate.from_weights(self.gains, (1, 1))

    def test_optimized_manual_weights(self):
        cert = Certificate.from_weights(self.gains, (5,1))
        self.assertEqual(cert.rho, Q(3,5))
        self.assertEqual(cert.radius((Q(1,100), Q(1,100))), Q(1,40))

    def test_invalid_and_unstable_contracts_rejected(self):
        for matrix in (((0,2),(1,0)), ((1,0),(0,0)), ((0,-1),(0,0))):
            with self.assertRaises(ValueError):
                Certificate.synthesize(matrix)
        with self.assertRaises(TypeError):
            Certificate.synthesize(((0.,),))
        with self.assertRaises(ValueError):
            Certificate.from_weights(((0,1),(0,0)), (0,1))

    def test_tightness_scalar_affine_disturbance(self):
        cert = Certificate.from_weights(((Q(1,2),),), (1,))
        delta = (Q(1,8),)
        x = Q(5,4)
        states = [(x,)]
        for k in range(1,7):
            x = Q(1,2)*x+delta[0]
            states.append((x,))
            self.assertEqual(x, cert.bound(Q(5,4), delta, k))
        self.assertEqual(audit_trace(cert,states,[0]*6,delta,1)['fair_blocks'],6)
        self.assertIsNone(cert.epochs_to_goal((Q(5,4),),delta,Q(1,4)))
        self.assertEqual(cert.epochs_to_goal((Q(5,4),),delta,Q(1,2)),2)

    def test_unfair_schedule_and_contract_violation(self):
        cert = Certificate.from_weights(self.gains, (5,1))
        with self.assertRaisesRegex(ValueError, 'block'):
            audit_trace(cert, [(0,0)]*5, [0,0,1,1], (0,0), 2)
        with self.assertRaisesRegex(ValueError, 'contract'):
            audit_trace(cert, [(0,0),(100,0),(100,0)], [0,1],(0,0),2)
        with self.assertRaisesRegex(ValueError, 'nonlocal'):
            audit_trace(cert, [(0,0),(0,1),(0,0)], [0,1],(0,0),2)

    def test_nonlinear_saturated_repair_with_disturbance(self):
        cert=Certificate.from_weights(self.gains,(5,1))
        x=(Q(10),Q(-3))
        states=[x]
        schedule=[0,1,1,0]*8
        d=(Q(1,50),Q(1,50))
        def clip(v):
            return max(Q(-2),min(Q(2),v))
        for t,i in enumerate(schedule):
            y=list(x)
            y[i]=clip(sum((a*z for a,z in zip(cert.gains[i],x)),Q(0)))+d[i]*(1 if t%2 else -1)
            x=tuple(y)
            states.append(x)
        receipt=audit_trace(cert,states,schedule,d,2)
        self.assertEqual(receipt['fair_blocks'],16)
        self.assertLessEqual(Q(receipt['final_error_norm']),Q(receipt['certified_upper_bound']))

    def test_grid_resolvent_matches_two_node_cycle_condition(self):
        values = (Q(0),Q(1,10),Q(1,2),Q(9,10),Q(2),Q(4))
        count=0
        rescued=0
        for a in values:
            for b in values:
                mat = ((0,a),(b,0))
                if a*b<1:
                    cert=Certificate.synthesize(mat)
                    cert.validate()
                    count+=1
                    if max(a,b)>=1:
                        rescued+=1
                else:
                    with self.assertRaises(ValueError):
                        Certificate.synthesize(mat)
        self.assertGreater(rescued,0)
        self.assertGreater(count,rescued)

    def test_arbitrary_mode_switching_common_envelope(self):
        modes=(((0, 3), (Q(1,20), 0)), ((0,2),(Q(1,10),0)))
        cert=Certificate.synthesize_for_modes(modes)
        self.assertEqual(cert.gains, self.gains)
        x=(Q(5),Q(2))
        states=[x]
        schedule=[0,1]*10
        for t,i in enumerate(schedule):
            a=modes[t%2]
            next_x=list(x)
            next_x[i]=sum((v*y for v,y in zip(a[i],x)),Q(0))
            x=tuple(next_x)
            states.append(x)
        receipt=audit_trace(cert,states,schedule,(0,0),2)
        self.assertEqual(receipt['fair_blocks'],10)

    def test_stable_individual_modes_can_destabilize_asynchronously(self):
        modes=(((0,3),(0,0)), ((0,0),(3,0)))
        for a in modes:
            Certificate.synthesize(a)
        self.assertEqual(mode_envelope(modes),((Q(0),Q(3)),(Q(3),Q(0))))
        with self.assertRaises(ValueError):
            Certificate.synthesize_for_modes(modes)
        state=(Q(0),Q(1))
        for _ in range(4):
            state=(Q(3)*state[1],state[1])
            state=(state[0],Q(3)*state[0])
        self.assertEqual(state[1],Q(9)**4)

    def test_demo_receipt(self):
        result=demo()
        self.assertEqual(result['rho'],'11/20')
        self.assertLessEqual(Q(result['final_error_norm']),Q(result['certified_upper_bound']))


if __name__=='__main__':
    unittest.main()
