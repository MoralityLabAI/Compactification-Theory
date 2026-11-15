"""Verified offline frozen model probe scores without needing torch installed."""
import copy
import json
import unittest
from pathlib import Path

from frozen_loader import read_payload,verify_frozen
from temporal_observers import (
    FIT_CONTEXTS,TEST_CONTEXTS,EDGES,N,T,
    candidate_projections,encode,covered_pairs,head_report,score_multi_heads,
)
from visit_phase_controls import visit_scramble,layer_separation_diagnostic,count_only_score
from run_cap2_control import diameter

ROOT=Path(__file__).resolve().parent

class FrozenTemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.digests=verify_frozen()
        cls.receipt=read_payload('temporal_receipt.json')
        cls.baseline=read_payload('equal_budget_baselines.json')

    def test_weight_hash_and_admission(self):
        self.assertEqual([x['seed'] for x in self.digests],[2406,2407,2408])
        for x in self.data['items']:
            self.assertEqual(x['train_examples'],64)
            self.assertEqual(x['model_predictions'],[a for a in range(8) for _ in range(8)])

    def test_probe_splits_are_disjoint_and_complete(self):
        self.assertEqual(set(FIT_CONTEXTS).intersection(TEST_CONTEXTS),set())
        self.assertEqual(set(FIT_CONTEXTS).union(TEST_CONTEXTS),set(range(8)))
        self.assertEqual(len(EDGES),28)
        self.assertEqual(len(self.data['items'][0]['activations']),64)

    def test_replay_frozen_temporal_head_scores(self):
        expected={2406:(64,90,102),2407:(62,72,86),2408:(61,79,97)}
        for x in self.data['items']:
            row=next(r for r in self.receipt['rows'] if r['seed']==x['model_seed'])
            for key,mode,bits in (
                ('final_1bit','last',1),
                ('temporal_shared_1bit_x4','temporal',1),
                ('temporal_shared_2bit_x4','temporal',2),
                ('final_shared_2bit','last',2),
            ):
                if bits==2: source='temporal_shared_1bit_x4' if mode=='temporal' else 'final_1bit'
                else:source=key
                head=(row['heads'][source]['weight'],row['heads'][source]['threshold'])
                actual=head_report(x,head,mode,bits)
                for a in ('fit_edge_hits','heldout_edge_hits','heldout_label_unique_per_context'):
                    self.assertEqual(row['heads'][key][a],actual[a])
            self.assertEqual(tuple(row['heads'][z]['heldout_edge_hits'] for z in
                                   ('final_1bit','temporal_shared_1bit_x4','temporal_shared_2bit_x4')),
                             expected[x['model_seed']])

    def test_equal_total_bit_baseline_replay(self):
        expected={2406:(112,105),2407:(106,111),2408:(111,111)}
        for item in self.data['items']:
            row=next(r for r in self.baseline['rows'] if r['seed']==item['model_seed'])
            for kind,head_key in (('final_four_bits','final_heads'),('untied_four_visits','untied_heads')):
                heads=[((h['w'],h['threshold']),h['visit']) for h in row[head_key]]
                actual=score_multi_heads(item,heads)
                for key in ('fit_edge_hits','heldout_edge_hits','heldout_unique_per_context'):
                    self.assertEqual(actual[key],row[kind][key])
            self.assertEqual((row['final_four_bits']['heldout_edge_hits'],
                             row['untied_four_visits']['heldout_edge_hits']),expected[item['model_seed']])

    def test_temporal_order_does_not_improve_over_bit_counts(self):
        # Post-hoc diagnostic, intentionally scored on the frozen holdout.
        for item in self.data['items']:
            row=next(r for r in self.receipt['rows'] if r['seed']==item['model_seed'])
            obs=row['heads']['temporal_shared_1bit_x4']
            head=(obs['weight'],obs['threshold'])
            count=count_only_score(item,head)
            self.assertEqual(count['edge_hits'],obs['heldout_edge_hits'])
            self.assertEqual(count['unique_counts'],obs['heldout_label_unique_per_context'])

    def test_last_replay_is_identical(self):
        for row in self.receipt['rows']:
            self.assertEqual(row['heads']['repeat_final_1bit_x4']['heldout_edge_hits'],
                             row['heads']['final_1bit']['heldout_edge_hits'])

    def test_cap2_posthoc_control_replay(self):
        receipt=read_payload('cap2_receipt.json')
        expected={2406:79,2407:59,2408:74}
        self.assertTrue(receipt['posthoc'])
        for item in self.data['items']:
            row=next(r for r in receipt['rows'] if r['seed']==item['model_seed'])
            head=(row['weight'],row['threshold'])
            fit=encode(item,head,FIT_CONTEXTS,'temporal')
            hold=encode(item,head,TEST_CONTEXTS,'temporal')
            self.assertEqual(diameter(fit),row['fit_diameter'])
            self.assertEqual(diameter(hold),row['heldout_diameter'])
            self.assertEqual(diameter(hold),2)
            self.assertEqual(covered_pairs(fit),row['fit_edge_hits'])
            self.assertEqual(covered_pairs(hold),expected[item['model_seed']])

    def test_heads_are_fit_only_on_observed_contexts(self):
        item=self.data['items'][0]
        # Freeze the entire fitting slice; scramble every held-out activation.
        tampered=copy.deepcopy(item)
        for a in range(N):
            for c in TEST_CONTEXTS:
                tampered['activations'][f'{a}:{c}']=[[999.0]*24 for _ in range(T)]
        # Equal candidate pool and order certify the generator's fit-only use.
        heads=candidate_projections(item,seed=item['model_seed']+601)
        new_heads=candidate_projections(tampered,seed=item['model_seed']+601)
        self.assertEqual(heads,new_heads)

    def test_visit_order_shuffle_and_distance_diagnostic(self):
        for x in self.data['items']:
            row=next(r for r in self.receipt['rows'] if r['seed']==x['model_seed'])
            head=(row['heads']['temporal_shared_1bit_x4']['weight'],
                  row['heads']['temporal_shared_1bit_x4']['threshold'])
            vals=[visit_scramble(x,head,j) for j in range(1,6)]
            self.assertEqual(len(vals),5)
            self.assertTrue(all(0<=v<=4*len(EDGES) for v in vals))
            diag=layer_separation_diagnostic(x)
            self.assertEqual(len(diag),4)
            self.assertTrue(all(v['minimum_l2']>0 for v in diag))

if __name__=='__main__':unittest.main()
