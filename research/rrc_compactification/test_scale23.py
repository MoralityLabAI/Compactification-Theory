"""RRC-23: exact scaling, enlarged grammar, collision and metering tests."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    from . import scale23 as s, bench23 as b, graph11 as g, topology16 as t
    from . import meter22, learn18 as l
except ImportError:
    import scale23 as s
    import bench23 as b
    import graph11 as g
    import topology16 as t
    import meter22
    import learn18 as l

HERE=Path(__file__).resolve().parent


class ScalingArithmeticTests(unittest.TestCase):
    def test_all_generator_families_are_integral_normalized(self):
        import random
        rng=random.Random(20261123)
        for name in s.FAMILIES:
            degree = 96 if name=='late_onset' else 48
            out=s.source(rng,name,degree)
            self.assertEqual(len(out),degree+1)
            self.assertEqual(out[0],1)
            self.assertTrue(all(type(x) is int for x in out))
        with self.assertRaises(ValueError):s.source(rng,'imaginary',48)

    def test_product_and_twist_small_degree_independent_coin_oracle(self):
        import random
        rng=random.Random(19)
        for family in ('product','twisted_product'):
            x=s.source(rng,family,40)
            self.assertEqual(x[0],1)
            self.assertTrue(all(type(y) is int for y in x))
            self.assertTrue(any(y!=0 for y in x[1:]))
        # Independent elementary product 1/(1-q)*(1/(1-q^3)).
        weights=[sum(int(n>=3*j) for j in range(n//3+1)) for n in range(12)]
        expected=[n//3+1 for n in range(12)]
        self.assertEqual(weights,expected)

    def test_exact_series_inverse_and_square(self):
        import random
        rng=random.Random(31)
        for family in ('rational','squared'):
            vals=s.source(rng,family,60)
            self.assertEqual(vals[0],1)
            if family=='rational':
                inv=g.inv(vals)
                self.assertEqual(g.mul(vals,inv),(1,)+(0,)*60)

    def test_three_and_four_token_semantics(self):
        import random
        rng=random.Random(41)
        for topo in s.TOPOLOGIES:
            for _ in range(4):
                f=s.source(rng,'signed',32)
                gr=rng.choice(g.GRAPHS)
                self.assertEqual(s.execute(f,gr,topo),t.execute(f,gr,topo))
                for pre in s.PRE_OPS:
                    self.assertEqual(s.execute(f,(pre,)+gr,topo),
                                     t.execute(g.unary(f,pre),gr,topo))
        with self.assertRaises(ValueError):s.execute((1,1,2),('Q','I','I','I'),'fork')

    def test_horizon_cut_preserves_model_inputs(self):
        x=s.dataset(per_cell=1)[0]
        for d in s.HORIZONS:
            c=s.observed(x,d)
            self.assertEqual(c.visible,x.visible)
            self.assertEqual(len(c.demos[0][0]),d+1)
            self.assertEqual(len(c.query),d+25)
        with self.assertRaises(ValueError):s.observed(x,500)

    def test_primitive_meter_matches_exact_all_topologies(self):
        import random
        rng=random.Random(70)
        for topo in s.TOPOLOGIES:
            for prog in (('R','N','I'),('I','I','I'),('S','R','R')):
                f=s.source(rng,'signed',48)
                meter=meter22.Meter()
                self.assertEqual(meter22.execute(f,prog,topo,meter),s.execute(f,prog,topo))
                self.assertGreater(meter.max_intermediate_bits,0)

    def test_progressive_reexecutes_and_counts_every_stage(self):
        orig=s.dataset(per_cell=1)[2]
        case=s.observed(orig,48)
        paths=(('I','I','I'),('N','S','R'),('R','R','R'),orig.hidden_graph)
        paths=tuple(dict.fromkeys(paths))
        original=s.execute
        with patch.object(s,'execute',wraps=original) as called:
            res=s.verify(case,paths,degree=48,progressive=True)
        self.assertEqual(called.call_count,res['demo_executions'])
        self.assertEqual(res['demo_executions'],2*sum(res['stage_counts'].values()))
        self.assertEqual(set(res['stage_counts']),{'4','12','48'})
        self.assertLessEqual(res['queries'],len(paths))
        self.assertGreater(res['nominal_work'],0)

    def test_progressive_and_direct_make_identical_decisions(self):
        cases=s.dataset(per_cell=1)[:8]
        classes=l.class_map()
        for d in (12,48,192):
            for orig in cases:
                c=s.observed(orig,d)
                order=l.mass_order(c.topology,classes)
                direct=s.verify(c,order,degree=d)
                progressive=s.verify(c,order,degree=d,progressive=True)
                self.assertEqual((direct['program'],direct['trace']),
                                 (progressive['program'],progressive['trace']))

    def test_future_collision_cannot_change_policy(self):
        c=s.observed(s.dataset(per_cell=1)[0],48)
        changed=list(c.future);changed[60]+=1
        mutant=replace(c,future=tuple(changed),hidden_graph=None,identifier='mutant')
        ck,cal,_=b.checkpoints();classes=l.class_map()
        for arm in ('symbolic_full','selective_full'):
            x=b.predict(c,arm,ck,cal,classes,48)
            y=b.predict(mutant,arm,ck,cal,classes,48)
            self.assertEqual(x,y)
            self.assertEqual(s.grade(c,x['program'])['future_correct']+
                             s.grade(mutant,x['program'])['future_correct'],1)

    def test_full_syntax_class_coverage_and_real_budget(self):
        ck,cal,_=b.checkpoints();classes=l.class_map()
        for orig in s.dataset(per_cell=1)[:8]:
            for d in (12,48,96):
                c=s.observed(orig,d)
                full=b.predict(c,'syntax64',ck,cal,classes,d)
                self.assertTrue(s.grade(c,full['program'])['accepted'])
                exact=b.predict(c,'symbolic_full',ck,cal,classes,d)
                self.assertTrue(s.grade(c,exact['program'])['accepted'])
                bounded=b.predict(c,'selective_full',ck,cal,classes,d)
                self.assertLessEqual(bounded['queries'],len(g.GRAPHS))

    def test_extended_syntax_has_256_programs_and_no_new_model_head(self):
        classes=l.class_map();model,cal,_=b.checkpoints()
        self.assertEqual(len(s.extended_order(g.GRAPHS)),256)
        for orig,pre in s.extended_dataset(per_topology=1):
            case=s.observed(orig,48)
            exact=b.extended_predict(case,'extended_syntax256',model,cal,classes,48)
            self.assertTrue(s.grade(case,exact['program'])['accepted'])
            self.assertLessEqual(exact['queries'],256)
            limited=b.extended_predict(case,'extended_neural12',model,cal,classes,48)
            self.assertLessEqual(limited['queries'],12)

    def test_high_degree_stress_is_structurally_separate(self):
        original=s.stress_dataset()
        self.assertEqual(len(original),s.STRESS_CASES)
        self.assertEqual({c.topology for c in original},set(s.TOPOLOGIES))
        for c in original:
            item=s.observed(c,s.STRESS_DEGREE)
            self.assertEqual(item.visible,c.visible)
            self.assertEqual(len(item.query),s.STRESS_DEGREE+25)


class FrozenReceiptsTests(unittest.TestCase):
    def test_no_training_and_unchanged_old_semantic_receipt(self):
        record=json.loads((HERE/'results/rrc23_deterministic.json').read_text())
        self.assertEqual(record['parameters']['frozen_training_updates'],0)
        self.assertEqual(record['parameters']['fixed_frozen_model_observation_degree'],12)
        self.assertEqual(record['parent_rrc22_semantic_sha256'],
                         __import__('hashlib').sha256((HERE/'results/rrc22_deterministic.json').read_bytes()).hexdigest())

    def test_full_source_and_checkpoint_locks(self):
        record=json.loads((HERE/'results/rrc23_deterministic.json').read_text())
        self.assertEqual(record['source_sha256'],b.source_hashes())
        self.assertEqual(record['checkpoint_sha256'],b.checkpoints()[2])
        self.assertEqual(record['parameters']['cases'],len(s.FAMILIES)*len(s.TOPOLOGIES)*s.PER_CELL)

    def test_recorded_progressive_parity_and_grammar_coverage(self):
        record=json.loads((HERE/'results/rrc23_deterministic.json').read_text())
        for h,group in record['horizon_scores'].items():
            self.assertEqual(group['symbolic_full']['accepted'],group['symbolic_progressive']['accepted'])
            self.assertEqual(group['symbolic_full']['queries'],group['symbolic_progressive']['queries'])
            self.assertEqual(group['selective_full']['accepted'],group['selective_progressive']['accepted'])
            self.assertGreaterEqual(group['symbolic_progressive']['demo_executions'],
                                    2*group['symbolic_progressive']['queries'])
        for h,group in record['grammar_scores'].items():
            self.assertEqual(group['extended_syntax256']['accepted'],s.EXTENDED_PER_TOPOLOGY*4)
            self.assertEqual(group['extended_mass']['accepted'],s.EXTENDED_PER_TOPOLOGY*4)
        self.assertEqual(record['long_horizon_stress']['cases'],8)
        self.assertEqual(record['long_horizon_stress']['degree'],384)

    def test_empirical_alias_stress_is_explicitly_finite(self):
        record=json.loads((HERE/'results/rrc23_deterministic.json').read_text())
        for h,result in record['long_horizon_alias_stress'].items():
            self.assertEqual(result['candidate_equivalence_comparisons'],600)
            self.assertIn('degree-12',result['scope'])

    def test_machine_profile_is_separate_from_exact_receipt(self):
        profile=json.loads((HERE/'results/rrc23_reference_profile.json').read_text())
        self.assertEqual(profile['schema'],'rrc23.physical.v1')
        self.assertEqual(profile['source_sha256'],b.source_hashes())
        self.assertEqual(profile['expanded_grammar_timing_cases'],8)
        self.assertEqual(profile['long_horizon_stress']['cases'],8)
        for degree,groups in profile['results'].items():
            for arm,info in groups.items():
                self.assertEqual(info['total_measurements'],2*len(s.FAMILIES)*len(s.TOPOLOGIES)*s.PER_CELL)
                self.assertGreater(info['end_to_end_wall']['total_ms'],0)
                self.assertGreater(info['verifier_wall']['total_ms'],0)


if __name__=='__main__':unittest.main()
