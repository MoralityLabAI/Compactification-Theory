"""COE-05: no-torch validation of exact phase codes and frozen probe evidence."""
import gzip
import hashlib
import itertools
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    from .phase_register import (WIDTH, affine_word, decode_nearest, evidence,
        hamming, necklace_count, ordered_trace, orbit_id, phase_codebook,
        rotate_right, triple_packing_bound)
    from .phase_order_replay import load_json, pooled, verify_receipt, score_head
except ImportError:
    from phase_register import (WIDTH, affine_word, decode_nearest, evidence,
        hamming, necklace_count, ordered_trace, orbit_id, phase_codebook,
        rotate_right, triple_packing_bound)
    from phase_order_replay import load_json, pooled, verify_receipt, score_head


def source_folder():
    return HERE


class TestPhaseGaugeMath(unittest.TestCase):
    def test_affine_generator_and_tied_dynamics(self):
        codes = phase_codebook()
        self.assertEqual(len(codes), 14)
        self.assertEqual(len(set(codes)), 14)
        for a in range(1, 8):
            self.assertEqual(affine_word(a, 0) ^ affine_word(a, 1), 255)
        for code in codes:
            self.assertEqual(ordered_trace(code), code)
            self.assertEqual(code.bit_count(), 4)
            self.assertEqual(rotate_right(code, 8).bit_count(), 4)

    def test_one_flip_robustness_exact(self):
        codes = phase_codebook()
        pairs = list(itertools.combinations(codes, 2))
        self.assertEqual(len(pairs), 91)
        self.assertEqual(min(hamming(a, b) for a, b in pairs), 4)
        for label, code in enumerate(codes):
            self.assertEqual(decode_nearest(code, codes), label)
            for position in range(8):
                self.assertEqual(decode_nearest(code ^ (1 << position), codes), label)

    def test_triple_packing_attains_upper_bound(self):
        codes = phase_codebook()
        supports = [tuple(i for i in range(8) if w & (1 << i)) for w in codes]
        triples = [t for support in supports for t in itertools.combinations(support, 3)]
        self.assertEqual((len(triples),len(set(triples))), (56,56))
        self.assertEqual(triple_packing_bound(8,4), 14)
        self.assertEqual(max(triple_packing_bound(7,w) for w in range(8)),7)
        # Padding a shorter code to 7 bits leaves all Hamming distances and
        # its constant weight unchanged, so the 7-bit bound covers T<7.

    def test_unknown_phase_gauge_bayes(self):
        codes = phase_codebook()
        self.assertEqual(necklace_count(8,4), 10)
        groups = {}
        for label,code in enumerate(codes):
            groups.setdefault(orbit_id(code),[]).append(label)
        self.assertEqual(sorted(map(len,groups.values())),[2,4,4,4])
        # Under uniform random cyclic phase shifts each codeword in one orbit
        # generates the identical distribution on the orbit.
        for labels in groups.values():
            distributions=[]
            for label in labels:
                counter={}
                state=codes[label]
                for _ in range(8):
                    counter[state]=counter.get(state,0)+1
                    state=rotate_right(state)
                distributions.append(counter)
            self.assertTrue(all(v==distributions[0] for v in distributions))
        self.assertEqual(len(groups),4)
        self.assertEqual(evidence()['unknown_uniform_rotation_bayes_accuracy'], 4/14)

    def test_exact_witness_receipt(self):
        witness = load_json(HERE/'phase_code_receipt.json')
        self.assertEqual(evidence(),witness)


class TestFrozenPhaseExperiment(unittest.TestCase):
    def test_replay_previous_and_replication(self):
        previous=load_json(HERE/'frozen_loop_activations.json.gz')
        old=load_json(HERE/'old_fit_phase_receipt.json')
        verify_receipt(previous,old)
        self.assertEqual(pooled(old)['heldout_order_only'],1)
        newer=load_json(HERE/'replication_activations.json.gz')
        new=load_json(HERE/'new_fit_phase_receipt.json')
        verify_receipt(newer,new)
        self.assertEqual(pooled(new),{
            'model_seeds':[2411,2412,2413], 'heldout_total':243,
            'heldout_count_only':233, 'heldout_order_only':10,
            'denominator':336})

    def test_full_weight_provenance(self):
        bundle=load_json(HERE/'replication_activations.json.gz')
        weights=load_json(HERE/'replication_weights.json.gz')
        self.assertEqual(set(weights),{str(i['model_seed']) for i in bundle['items']})
        for item in bundle['items']:
            payload=weights[str(item['model_seed'])]
            canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
            sha=hashlib.sha256(canonical.encode()).hexdigest()
            self.assertEqual(sha,item['model_checkpoint_sha256'])

    def test_corruption_rejected(self):
        previous=load_json(HERE/'frozen_loop_activations.json.gz')
        old=load_json(HERE/'old_fit_phase_receipt.json')
        old['records'][0]['primary_stable_order_head']['heldout_order_only_hits']+=1
        with self.assertRaises(ValueError):
            verify_receipt(previous,old)

    def test_no_retrospective_fit_over_holdout(self):
        bundle=load_json(HERE/'replication_activations.json.gz')
        receipt=load_json(HERE/'new_fit_phase_receipt.json')
        # The recorded fit-only metrics are independent of the heldout bytes:
        # scrambling heldout activation vectors changes heldout scores but
        # must leave fit selection input and fit scores exactly untouched.
        from copy import deepcopy
        tampered=deepcopy(bundle)
        for item in tampered['items']:
            for a in range(8):
                for ctx in range(4,8):
                    item['activations'][f'{a}:{ctx}']=[[0.0]*24 for _ in range(4)]
        for item,record in zip(tampered['items'],receipt['records']):
            head=record['primary_stable_order_head']
            score=score_head(item,head)
            for key in ('fit_stable_novel_edges','fit_order_only_hits','fit_total_edge_hits'):
                self.assertEqual(score[key],head[key])
            self.assertEqual(score['heldout_total_edge_hits'],0)


if __name__=='__main__': unittest.main()
