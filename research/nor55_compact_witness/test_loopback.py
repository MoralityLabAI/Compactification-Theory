"""NOR55 separate-process real localhost TCP toy pre-data smoke test.

No first NOR55 source identities are generated. Tests exactly the local
secondary producer/verifier socket protocol on old, tiny synthetic integers.
"""
from __future__ import annotations
import sys,unittest
from hashlib import sha256
from research.nor55_compact_witness import loopback_secondary

class LoopbackToyTests(unittest.TestCase):
    def test_three_nonce_bound_cross_process_signed_requests(self):
        if sys.platform!="linux":
            self.skipTest("registered separate-process TCP test runs on Linux")
        f=[1]+[0]*40
        f[4]=3
        f[8]=-2
        f=tuple(f)
        result=loopback_secondary.source_loopback(
            f,sha256(b"NOR55-LOCAL-LOOPBACK-TOY").hexdigest(),4)
        self.assertEqual(len(result["producer_to_verified_response_ms"]),3)
        self.assertEqual(len(result["native_compressed_same_job_baseline_ms"]),3)
        self.assertTrue(result["all_three_independent_verifier_responses_matched"])
        self.assertTrue(result["separate_producer_and_verifier_processes"])
        self.assertTrue(all(x>4 for x in result["request_bytes"]))
        self.assertTrue(all(x>4 for x in result["response_bytes"]))
        self.assertGreater(result["median_loopback_ms"],0)

if __name__=="__main__":
    unittest.main()
