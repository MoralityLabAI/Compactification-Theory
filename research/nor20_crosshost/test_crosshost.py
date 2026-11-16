"""NOR-20 frozen wide source generator and public-only cross-host tests."""
from __future__ import annotations

from collections import Counter
import json
import random
import statistics
import unittest

from research.nor20_crosshost import benchmark as n
from research.rrc_compactification import compound25 as corpus
from research.nor19_streamed_verifier import benchmark as nor19


class NOR20CrossHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples=[]
        used=set()
        for wi,fi in ((0,0),(1,1),(2,2),(3,3)):
            c,h,attempt,mask,unique,source_hash=n.sample_case(
                0,wi,fi,used)
            cls.samples.append((c,h,mask,unique,source_hash))
        assert len(used)==4

    def test_wide_distribution_and_seed_registration(self):
        self.assertEqual(n.BITS,(4,8,12,16))
        self.assertEqual(n.FAMILIES,(
            "wide_dense","wide_sparse","wide_late","wide_rational"))
        self.assertEqual(n.COHORTS,4)
        self.assertEqual(n.N_CASES,64)
        self.assertEqual(n.TRIALS,7)
        self.assertEqual(n.WARMUPS,2)
        self.assertEqual(n.BOOTSTRAPS,4096)
        self.assertEqual(n.seed_for(0,0,0),corpus.SEED+94120000)
        with self.assertRaises(ValueError):
            n.make_source(random.Random(1),"invalid",8)
        with self.assertRaises(ValueError):
            n.make_source(random.Random(1),"wide_dense",3)
        with self.assertRaises(ValueError):
            n.seed_for(4,0,0)

    def test_exact_public_16bit_masks_on_wide_source_cases(self):
        for c,h,mask,unique,source_hash in self.samples:
            self.assertEqual(c.program,None)
            self.assertEqual(c.future,())
            self.assertEqual(n.full_public_hash(c),source_hash)
            expected=n.reference_certificate(c,h)
            self.assertEqual(expected,(mask,unique))
            for mode in n.MODES:
                out=nor19.complete(c,h,mode)
                self.assertEqual((out["mask"],out["unanimous"]),expected)
                self.assertEqual(out["chosen_class"] is not None,unique)

    def test_no_true_target_access_even_for_high_coefficient_inputs(self):
        for c,h,mask,unique,digest in self.samples:
            for mode in n.MODES:
                out=nor19.complete(c,h,mode)
                self.assertEqual((out["mask"],out["unanimous"]),(mask,unique))

    def test_normalization_and_actual_coefficient_widths(self):
        for c,h,mask,unique,digest in self.samples:
            self.assertEqual(len(c.demos),2)
            self.assertEqual(len(c.query),257)
            self.assertEqual(c.query[0],1)
            for inp,out in c.demos:
                self.assertEqual(len(inp),257)
                self.assertEqual(len(out),257)
                self.assertEqual(inp[0],1)
            max_source,max_out=n.public_max_bits(c)
            self.assertGreaterEqual(max_source,4)
            self.assertGreaterEqual(max_out,1)

    def test_bigint_work_reuse_exact(self):
        for c,h,mask,unique,digest in self.samples:
            inverse=[n.nor18r.exact_inverse_operand_ledger(inp)
                     for inp,y in c.demos]
            inverse.append(n.nor18r.exact_inverse_operand_ledger(c.query))
            baseline=nor19.observed_bigint_work(c,h,"independent16",inverse)
            cache=nor19.observed_bigint_work(c,h,"cached16",inverse)
            streamed=nor19.observed_bigint_work(c,h,"stream16_verifyfirst",inverse)
            self.assertEqual(cache,streamed)
            self.assertGreater(baseline["R_integer_products"],cache["R_integer_products"])
            self.assertGreater(baseline["R_operand_bitwidth_product_proxy"],
                               cache["R_operand_bitwidth_product_proxy"])

    def test_timing_accounting_and_memory_samples(self):
        c,h,mask,unique,_=self.samples[0]
        for mode in n.MODES:
            measured=nor19.complete(c,h,mode,measure=True)
            self.assertEqual(measured["times"]["total_ns"],
                sum(measured["times"][k] for k in (
                    "query_ns","verification_ns","certificate_ns")))
            profile=nor19.allocation_profile(c,h,mode)
            self.assertEqual(profile["mask"],mask)
            self.assertGreater(profile["python_peak_allocation_bytes"],0)

    def test_block_bootstrap_determinism(self):
        observations=[{"cohort":i,"ratio":.4+.1*i}
                      for i in range(4) for _ in range(16)]
        a=n.block_interval(observations,"ratio")
        b=n.block_interval(observations,"ratio")
        self.assertEqual(a,b)
        self.assertEqual(a["draws"],4096)
        self.assertLessEqual(a["p2_5"],a["p97_5"])

    def test_fixture_has_no_hidden_program_or_target_keys(self):
        for c,h,mask,unique,digest in self.samples:
            record={
                "public_header":{"base":h.base,"delay":h.delay,
                                  "spacing":h.spacing,"amplitude":h.amplitude},
                "demos":[[list(src[:193]),list(y[:193])] for src,y in c.demos],
                "public_query_input":list(c.query),
                "full_public_sha256":digest,
            }
            raw=n.canonical(record)
            self.assertNotIn(b'"program"',raw)
            self.assertNotIn(b'"future"',raw)
            self.assertNotIn(b'"private"',raw)


if __name__=="__main__":
    unittest.main()
