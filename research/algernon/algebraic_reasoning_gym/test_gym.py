"""Frozen RRC-19 checkpoint, independent EvalMath oracle, offline-model protocol."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import unittest

from spiral_ln import algebraic_reasoning_oracle as e
try:
    from . import gym as g, external
except ImportError:
    import gym as g
    import external

HERE=Path(__file__).resolve().parent


class BackendTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models=g.checkpoints()
        cls.classes=g.compile_classes()
        cls.cases=e.roster(calibration_per_family=2,test_per_family=3,negative=2,collision_pairs=2)

    def test_source_and_real_pretrained_checkpoint_pins(self):
        p=g.load_pin()
        self.assertTrue(p['evalmath_commit_sha'] is None or len(p['evalmath_commit_sha'])==40)
        self.assertEqual(set(self.models),set(g.CHECKPOINT_ROLES))
        self.assertEqual(self.models['frozen_gru']['parameter_count'],6240)
        self.assertEqual(self.models['frozen_ff']['parameter_count'],6235)
        self.assertEqual(self.models['shuffled_gru']['kind'],'recurrent')

    def test_exact_and_class_prior_complete(self):
        for family in e.ALL_FAMILIES:
            for case in self.cases[family]:
                full=g.score_case(case,'exhaustive_64',self.models,self.classes)
                class_full=g.score_case(case,'certified_mass_full',self.models,self.classes)
                self.assertEqual(full['grade']['accepted'],1)
                self.assertEqual(class_full['grade']['accepted'],1)
                self.assertLessEqual(full['queries'],64)

    def test_model_blind_to_future_and_hidden_labels(self):
        c=self.cases['product'][0]
        changed=replace(c,source_graph=None,future=(1,)*len(c.future),identifier='changed')
        for arm in g.CHECKPOINT_ROLES:
            self.assertEqual(g.model_order(c,self.models[arm],self.classes),
                             g.model_order(changed,self.models[arm],self.classes))

    def test_model_does_not_execute_candidate_graphs_while_ranking(self):
        c=self.cases['signed'][0]
        with patch.object(e,'execute',side_effect=AssertionError('candidate oracle called')):
            for arm in g.CHECKPOINT_ROLES:
                candidates=g.model_order(c,self.models[arm],self.classes)
                self.assertEqual(len(candidates),12)
                self.assertEqual(len(set(candidates)),12)

    def test_paid_exact_query_budget(self):
        c=self.cases['rational'][0]
        original=e.execute
        with patch.object(e,'execute',wraps=original) as called:
            # score_case also runs future grading if a candidate verifies;
            # inference-time *search* alone is counted separately here.
            z=e.search(c,g.model_order(c,self.models['frozen_gru'],self.classes),budget=12)
            self.assertEqual(called.call_count,2*z['queries'])
        self.assertLessEqual(z['queries'],12)

    def test_finite_collision_same_candidate_trace(self):
        for a,b in zip(self.cases['collision'][::2],self.cases['collision'][1::2]):
            self.assertEqual(a.signature,b.signature)
            for arm in g.ARMS:
                x=g.score_case(a,arm,self.models,self.classes)
                y=g.score_case(b,arm,self.models,self.classes)
                self.assertEqual(x['trace'],y['trace'])

    def test_independent_interpreter_matches_original_rrc(self):
        import random
        from research.rrc_compactification import topology16 as original
        rng=random.Random(910271)
        for topology in e.TOPOLOGIES:
            for program in e.PROGRAMS:
                for _ in range(2):
                    x=(1,)+tuple(rng.randint(-2,2) for _ in range(e.PREFIX))
                    self.assertEqual(e.execute(x,topology,program),
                                     original.execute(x,program,topology))

    def test_frozen_gru_and_ff_match_native_pytorch(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch optional for portable CPU CI')
        from research.rrc_compactification import train18,learn18,model18
        torch.set_num_threads(1)
        case=self.cases['signed'][0]
        for role in ('frozen_gru','frozen_ff'):
            ck=self.models[role]
            net=train18.modules()[ck['kind']]()
            net.load_state_dict({key:torch.tensor(tensor,dtype=torch.float32)
                                 for key,tensor in ck['state_dict'].items()},strict=True)
            x=learn18.flat(case) if role=='frozen_ff' else learn18.sequence(case)
            with torch.no_grad():reference=net(torch.tensor([x],dtype=torch.float32))[0].tolist()
            portable=model18.logits(case,ck)
            self.assertLess(max(abs(a-b) for a,b in zip(reference,portable)),5e-5)

    def test_src_drifts_fail_closed(self):
        pin=g.load_pin()
        self.assertEqual(set(pin['checkpoint_sha256']),set(g.CHECKPOINT_ROLES))
        with patch.object(g,'file_digest',return_value='0'*64):
            with self.assertRaises(AssertionError):g.load_pin()


class ExternalAdapterTests(TestCase):
    def test_public_export_hides_answers(self):
        with TemporaryDirectory() as t:
            p=Path(t)/'public.jsonl'
            n=external.write_public(p,splits=('signed',))
            self.assertEqual(n,24)
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            self.assertTrue(all('future' not in v and 'hidden_graph' not in v and 'family' not in v for v in rows))
            self.assertEqual(set(rows[0]),{'case_id','topology','observed_degree','demos','query','public_sha256'})

    def test_valid_model_answer_is_actually_checked(self):
        case=e.roster()['signed'][0]
        p={'case_id':case.identifier,'public_sha256':case.signature,'depth':2,
           'program':list(case.source_graph),'model_revision':'a'*40}
        with TemporaryDirectory() as t:
            output=Path(t)/'answer.jsonl';output.write_text(json.dumps(p)+'\n')
            scores=external.score(output)
            self.assertEqual(scores['scores']['2']['exact_accepted'],1)
            self.assertEqual(scores['scores']['2']['future_correct'],1)
            self.assertEqual(scores['scores']['2']['missing'],191)

    def test_missing_revision_wrong_hash_and_leaks_rejected(self):
        case=e.roster()['signed'][0]
        p={'case_id':case.identifier,'public_sha256':case.signature,'depth':2,
           'program':list(case.source_graph),'model_revision':'a'*40}
        with TemporaryDirectory() as d:
            output=Path(d)/'answers.jsonl'
            output.write_text(json.dumps({**p,'future':case.future})+'\n')
            with self.assertRaises(ValueError):external.score(output)
            output.write_text(json.dumps({**p,'public_sha256':'0'*64})+'\n')
            with self.assertRaises(ValueError):external.score(output)
            output.write_text(json.dumps({**p,'model_revision':''})+'\n')
            with self.assertRaises(ValueError):external.score(output)

    def test_duplicate_proposal_rejected(self):
        case=e.roster()['signed'][0]
        p={'case_id':case.identifier,'public_sha256':case.signature,'depth':2,
           'program':list(case.source_graph),'model_revision':'a'*40}
        with TemporaryDirectory() as d:
            output=Path(d)/'answers.jsonl'
            output.write_text(json.dumps(p)+'\n'+json.dumps(p)+'\n')
            with self.assertRaises(ValueError):external.score(output)


class ReceiptTests(TestCase):
    def test_full_frozen_receipt(self):
        saved=json.loads((HERE/'results'/'arg01_reference.json').read_text())
        self.assertEqual(saved['schema'],'algernon.arg01.receipt.v1')
        self.assertEqual(saved['source_sha256'],g.source_hashes())
        self.assertEqual(saved['evalmath_pin'],g.load_pin())
        self.assertEqual(saved['settings']['actual_qwen_inference'],False)
        self.assertEqual(saved['scores']['negative']['exhaustive_64']['accepted'],0)
        self.assertEqual(saved['scores']['collision']['exhaustive_64']['false_future_acceptance'],6)
        self.assertEqual(saved['scores']['signed']['exhaustive_64']['accepted'],24)


if __name__=='__main__':unittest.main()
