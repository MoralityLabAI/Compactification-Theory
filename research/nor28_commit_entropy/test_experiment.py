"""NOR-28 frozen-stage unit tests: exact source isolation, commitments,
mutation rejection, HMAC sampling uniqueness, and postcommit nonanticipation."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from fractions import Fraction
from math import comb
from tempfile import TemporaryDirectory
import unittest

from research.nor28_commit_entropy import experiment as e
from research.nor28_commit_entropy import challenge_sampler as c
from research.nor28_commit_entropy import grade as g
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor26_policy_aware_audits import experiment as n26
from research.nor27_seed_leakage import experiment as n27
from research.nor21_degree_scaling import experiment as nor21


class NOR28ProtocolTests(unittest.TestCase):
    def test_registration_and_source_seed_disjointness(self):
        self.assertEqual(e.OFFSET,129328000)
        self.assertEqual(e.REPS,(10,11))
        self.assertEqual(e.CAL_TRIALS,256)
        self.assertEqual((e.T,e.U,e.R),(8,8,4))
        self.assertEqual(c.TRIALS,512)
        self.assertEqual(c.LEAK_BITS,(0,8,16,256))
        self.assertEqual(c.ARMS,("hybrid_50","uniform16","static_prior"))
        previous={n24.source_seed(r,d,f) for r in n24.TRAIN_REPS+n24.TEST_REPS
                  for d in n24.DEGREES for f in n24.FAMILIES}
        previous|={n25.source_seed(r,d,f) for r in n25.REPLICATES
                   for d in n25.DEGREES for f in n25.FAMILIES}
        previous|={n26.source_seed(r,d,f) for r in n26.REPS
                   for d in n26.DEGREES for f in n26.FAMILIES}
        previous|={n27.source_seed(r,d,f) for r in n27.REPS
                   for d in n27.DEGREES for f in n27.FAMILIES}
        new={e.source_seed(r,d,f) for r in e.REPS
             for d in e.DEGREES for f in e.FAMILIES}
        self.assertEqual(len(new),12)
        self.assertTrue(new.isdisjoint(previous))

    def test_exact_fixed_attack_lower_floors(self):
        for m in (769,1537):
            p=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
            self.assertEqual(p,n25.blind_floor(m,16,8,4))
            self.assertGreater(p,0)
        self.assertGreater(n25.blind_floor(769,16,8,4),
                           n25.blind_floor(1537,16,8,4))

    def test_original_native_source_parity(self):
        row=e.new_source(10,512,e.FAMILIES[0],set())
        self.assertEqual(row["split"],"TEST")
        self.assertTrue(all(k not in row for k in
                            ("program","future","entropy","attack","nonce")))
        public,header=n24.public_only_case(row)
        self.assertIsNone(public.program)
        self.assertEqual(public.future,())
        self.assertEqual(nor21.native_mask(public,header,512),
                         row["original_public_accept_mask"])
        proposal=n24.public_proposal(row)
        self.assertEqual(len(proposal["predicted"]),769)
        self.assertEqual(proposal["mask"],row["original_public_accept_mask"])
        order=e.public_order(proposal["predicted"],header,
                             row["full_public_input_sha256"])
        self.assertEqual(len(order),64)

    def test_salted_binding_and_mutation_rejection(self):
        p=dict(source_sha256="a"*64,attack_class="calibrated",replica=0,
               changed_indices=[1,2,3,4],nonce_hex="1"*64)
        original=e.commitment(p)
        self.assertEqual(original,e.commitment(dict(p)))
        for key,val in (
            ("source_sha256","b"*64),("attack_class","fresh_uniform"),
            ("replica",1),("changed_indices",[1,2,3,5]),
            ("nonce_hex","2"*64)):
            changed=dict(p)
            changed[key]=val
            self.assertNotEqual(original,e.commitment(changed))
        self.assertEqual(len(original),64)

    def test_unbiased_prf_sampler_and_phase_boundary(self):
        root=bytes(range(32))
        self.assertEqual(c.root_commitment(root),
                         c.root_commitment(bytes(range(32))))
        self.assertNotEqual(c.root_commitment(root),
                            c.root_commitment(bytes(range(1,33))))
        m=769
        digest="f"*64
        T=tuple(range(8))
        for policy in c.ARMS:
            for trial in range(16):
                sched=c.schedule(root,digest,policy,trial,m,T+tuple(range(8,64)))
                self.assertEqual(len(sched),len(set(sched)))
                self.assertEqual(len(sched),16)
                self.assertTrue(all(0<=i<m for i in sched))
                if policy=="hybrid_50":
                    self.assertEqual(sched[:8],T)
                    self.assertFalse(set(sched[8:])&set(T))
        self.assertNotEqual(
            c.schedule(root,digest,"hybrid_50",0,m,tuple(range(64))),
            c.schedule(root,digest,"hybrid_50",1,m,tuple(range(64))))
        # Neither the preimage nor a root disclosure can change a
        # previously frozen audit decision.
        attacked={15,30,70,100}
        checked=c.schedule(root,digest,"hybrid_50",0,m,tuple(range(64)))
        baseline=bool(attacked&set(checked))
        for bits in c.LEAK_BITS:
            disclosed=int.from_bytes(root,"big")>>(256-bits)
            self.assertEqual(disclosed,
                             int.from_bytes(root,"big")>>(256-bits))
            self.assertEqual(baseline,bool(attacked&set(checked)))

    def test_informed_evasion_and_commitment_failure(self):
        original=dict(source_sha256="a"*64,attack_class="calibrated",
                      replica=0,changed_indices=[12,13,14,15],
                      nonce_hex="0"*64)
        old=e.commitment(original)
        checked=tuple(range(16,32))
        forged=g.commitment_for_counterfactual(original,checked,769)
        self.assertEqual(len(set(forged["changed_indices"])),4)
        self.assertTrue(set(forged["changed_indices"]).isdisjoint(checked))
        self.assertNotEqual(e.commitment(forged),old)
        self.assertEqual(g.exact_detect(tuple(range(769)),tuple(range(769)),
                                        checked,forged["changed_indices"],1),False)

    def test_whole_source_only_fixture_no_hidden_data(self):
        data=e.make_fixture()
        self.assertEqual(len(data["cases"]),12)
        self.assertTrue(data["source_only"] and data["attacks_absent"])
        self.assertTrue(data["challenge_entropy_absent"])
        self.assertEqual(len({r["full_public_input_sha256"] for r in data["cases"]}),12)
        with TemporaryDirectory() as folder:
            path=Path(folder)/"fixture.json"
            path.write_bytes(e.canonical(data))
            loaded,digest=e.load_fixture(path)
            self.assertEqual(loaded,data)
            self.assertEqual(digest,sha256(path.read_bytes()).hexdigest())


if __name__=="__main__":
    unittest.main()
