import unittest
from fractions import Fraction as Q
from small_gain import Certificate
from delayed import audit_delayed_trace, delayed_superblock


class DelayedRepairTests(unittest.TestCase):
    def test_reduces_to_no_delay_rate(self):
        self.assertEqual(delayed_superblock(3,0),3)
        self.assertEqual(delayed_superblock(3,2),6)
        self.assertEqual(delayed_superblock(3,4),9)
        cert=Certificate.from_weights(((Q(1,2),),),(1,))
        x=Q(1)
        states=[(x,)]
        for _ in range(6):
            x=x/2
            states.append((x,))
        receipt=audit_delayed_trace(cert,states[:1],states,[0]*6,[(0,)]*6,(0,),1,0)
        self.assertEqual(receipt['superblock'],1)
        self.assertEqual(Q(receipt['certified_bound']),Q(1,64))

    def test_one_step_stale_scalar_sharp(self):
        cert=Certificate.from_weights(((Q(1,2),),),(1,))
        history=[(Q(1),),(Q(1),)]
        vals=[Q(1)]
        for t in range(8):
            v=history[0][0] if t==0 else vals[t-1]
            vals.append(v/2)
        states=[(v,) for v in vals]
        receipt=audit_delayed_trace(cert,history,states,[0]*8,[(1,)]*8,(0,),1,1)
        self.assertEqual(receipt['superblock'],2)
        self.assertEqual(Q(receipt['final_error_norm']),Q(1,16))
        self.assertEqual(Q(receipt['certified_bound']),Q(1,16))

    def test_two_component_stale_arbitrary_reads(self):
        gains=((0,Q(3)),(Q(1,10),0))
        cert=Certificate.from_weights(gains,(Q(11,2),1))
        history=[(Q(10),Q(-3))]*3   # e(-2),e(-1),e(0)
        states=[history[-1]]
        schedule=[0,1]*9
        delay_vectors=[]
        for t,i in enumerate(schedule):
            d=((t+1)%3,t%3)
            delay_vectors.append(d)
            old=[]
            for j,k in enumerate(d):
                old.append((states[t-k] if t-k>=0 else history[t-k+2])[j])
            y=list(states[-1]);y[i]=sum((a*v for a,v in zip(gains[i],old)),Q(0))
            states.append(tuple(y))
        receipt=audit_delayed_trace(cert,history,states,schedule,delay_vectors,(0,0),2,2)
        self.assertEqual(receipt['superblock'],4)
        self.assertLessEqual(Q(receipt['final_error_norm']),Q(receipt['certified_bound']))

    def test_invalid_contract_or_unfair_schedule(self):
        cert=Certificate.from_weights(((Q(1,2),),),(1,))
        hist=[(Q(1),),(Q(1),)]
        with self.assertRaisesRegex(ValueError,'contract'):
            audit_delayed_trace(cert,hist,[(Q(1),),(Q(10),)],[0],[(1,)],(0,),1,1)
        with self.assertRaisesRegex(ValueError,'delay'):
            audit_delayed_trace(cert,hist,[(Q(1),),(Q(1,2),)],[0],[(2,)],(0,),1,1)
        c2=Certificate.from_weights(((0,0),(0,0)),(1,1))
        with self.assertRaisesRegex(ValueError,'block'):
            audit_delayed_trace(c2,[(0,0)],[(0,0)]*3,[0,0],[(0,0)]*2,(0,0),2,0)


if __name__=='__main__':
    unittest.main()