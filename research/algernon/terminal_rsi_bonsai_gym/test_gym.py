import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from gym import (FAMILIES, MAX_ACTION_BYTES, MAX_STEPS, canonical, digest, export_sft,
                 make_task, parse_action, preflight, run_episode, scripted_actions,
                 split_for, validate_record, write_jsonl)


def training_task(family='transform'):
    return next(make_task(family, i, 11) for i in range(100) if make_task(family, i, 11)['split'] == 'train')


def resign(r):
    r.pop('receipt_hash', None)
    r['receipt_hash'] = digest(r)
    return r


class GymTests(unittest.TestCase):
    def test_all_families_scripted(self):
        for family in FAMILIES:
            for seed in (11, 23, 37):
                t = make_task(family, 0, seed)
                r = run_episode(t, scripted_actions(t))
                self.assertTrue(r['summary']['strict_success'], (family, seed))
                validate_record(r)

    def test_model_absent_baseline(self):
        for f in FAMILIES:
            r = run_episode(make_task(f, 0, 11), ['{"op":"finish"}'])
            self.assertFalse(r['summary']['strict_success'])

    def test_split_preserved_across_variants(self):
        for f in FAMILIES:
            self.assertEqual(make_task(f, 4, 11)['split'], make_task(f, 4, 99)['split'])

    def test_task_determinism(self):
        self.assertEqual(make_task('repair', 1, 2), make_task('repair', 1, 2))

    def test_invalid_identity(self):
        for args in [('bad', 0, 1), ('repair', True, 1), ('repair', -1, 1)]:
            with self.assertRaises(ValueError):
                make_task(*args)

    def test_duplicate_keys(self):
        with self.assertRaises(ValueError):
            parse_action('{"op":"finish","op":"test"}')

    def test_extra_fields(self):
        with self.assertRaises(ValueError):
            parse_action('{"op":"finish","success":true}')

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            parse_action('{"op":"write","path":"output.json","value":NaN}')

    def test_shell_rejected(self):
        with self.assertRaises(ValueError):
            parse_action('{"op":"shell","command":"anything"}')

    def test_protected_paths(self):
        for path in ('../input.json', '/etc/passwd', 'protected_policy.json', 'input.json'):
            with self.assertRaises(ValueError):
                parse_action(canonical({'op': 'write', 'path': path, 'value': 0}))

    def test_action_size(self):
        with self.assertRaises(ValueError):
            parse_action('x' * (MAX_ACTION_BYTES + 1))

    def test_budget_stop(self):
        r = run_episode(make_task('chain', 1, 2), ['{"op":"test"}'] * 100)
        self.assertEqual(r['summary']['steps'], MAX_STEPS)
        self.assertEqual(r['summary']['stop_reason'], 'step_budget')
        self.assertFalse(r['summary']['strict_success'])
        self.assertEqual(export_sft([r]), [])

    def test_oversized_failure_retained(self):
        r = run_episode(training_task(), ['x' * (MAX_ACTION_BYTES + 1)])
        self.assertIsNone(r['events'][0]['raw_text'])
        self.assertEqual(r['summary']['unauthorized_attempts'], 1)
        self.assertEqual(export_sft([r]), [])

    def test_rejected_attempt_survives_success(self):
        t = training_task()
        r = run_episode(t, ['{"op":"shell"}'] + scripted_actions(t))
        self.assertTrue(r['summary']['strict_success'])
        self.assertEqual(r['summary']['unauthorized_attempts'], 1)
        self.assertEqual(export_sft([r]), [])

    def test_transient_recovery(self):
        t = make_task('recovery', 0, 11)
        r = run_episode(t, scripted_actions(t))
        self.assertEqual(r['events'][0]['observation']['status'], 'tool_error')
        self.assertTrue(r['summary']['strict_success'])

    def test_public_pass_private_failure(self):
        t = make_task('transform', 0, 11)
        actions = [canonical({'op': 'write', 'path': 'output.json', 'value': []}), '{"op":"test"}', '{"op":"finish"}']
        r = run_episode(t, actions)
        self.assertTrue(r['events'][1]['observation']['public_pass'])
        self.assertFalse(r['summary']['strict_success'])

    def test_finish_requires_tests(self):
        t = training_task()
        r = run_episode(t, [x for x in scripted_actions(t) if json.loads(x)['op'] != 'test'])
        self.assertFalse(r['summary']['strict_success'])

    def test_finish_stops_future_actions(self):
        r = run_episode(training_task(), ['{"op":"finish"}', '{"op":"shell"}'])
        self.assertEqual(len(r['events']), 1)

    def test_forged_goal_rejected(self):
        t = training_task()
        t['initial_files']['input.json']['numbers'] = []
        with self.assertRaises(ValueError):
            run_episode(t, [])

    def test_missing_read_is_tool_error(self):
        r = run_episode(training_task(), ['{"op":"read","path":"output.json"}'])
        self.assertEqual(r['events'][0]['observation']['reason'], 'missing_file')

    def test_tampered_receipt(self):
        t = training_task()
        r = run_episode(t, scripted_actions(t))
        r['summary']['steps'] = 0
        with self.assertRaises(ValueError):
            export_sft([r])

    def test_rehashed_false_success_rejected(self):
        r = run_episode(training_task(), ['{"op":"finish"}'])
        r['summary']['strict_success'] = True
        resign(r)
        with self.assertRaises(ValueError):
            export_sft([r])

    def test_rehashed_split_rejected(self):
        t = next(make_task('transform', i, 11) for i in range(100) if split_for(f'transform:root-{i}') == 'test')
        r = run_episode(t, scripted_actions(t))
        r['split'] = 'train'
        resign(r)
        with self.assertRaises(ValueError):
            export_sft([r])

    def test_heldouts_excluded(self):
        rows = []
        for i in range(30):
            t = make_task('repair', i, 11)
            rows.append(run_episode(t, scripted_actions(t)))
        out = export_sft(rows)
        self.assertTrue(out)
        self.assertTrue(all(split_for(x['group_id']) == 'train' for x in out))

    def test_export_deduplicated(self):
        t = training_task()
        r = run_episode(t, scripted_actions(t))
        self.assertEqual(len(export_sft([r, r])), 1)

    def test_public_only_export(self):
        t = training_task()
        r = run_episode(t, scripted_actions(t))
        row = export_sft([r])[0]
        content = canonical(row['messages'])
        for forbidden in ('strict_success', 'state_after', 'receipt_hash', 'verifier_sha256', 'target'):
            self.assertNotIn(forbidden, content)
        self.assertFalse(row['eligible_for_model_training'])
        self.assertTrue(row['synthetic_fixture'])
        self.assertEqual(row['loss_on_message'], [m['role'] == 'assistant' for m in row['messages']])

    def test_unknown_model_metrics_are_null(self):
        t = training_task()
        r = run_episode(t, scripted_actions(t))
        self.assertIsNone(r['model_revision'])
        self.assertTrue(all(e['critic_value'] is None and e['output_tokens'] is None for e in r['events']))

    def test_preflight_missing_pins(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertGreater(len(preflight({}, Path(d))), 6)

    def test_preflight_file_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'receipt.json').write_text('{"fixture":true}')
            sha = hashlib.sha256((root / 'receipt.json').read_bytes()).hexdigest()
            model = {'repository': 'fixture/repo', 'revision': 'a' * 40, 'weights_sha256': 'b' * 64,
                     'tokenizer_sha256': 'c' * 64, 'license_reviewed': True}
            evidence = {name: {'path': 'receipt.json', 'sha256': sha, 'passed': True} for name in
                        ('forward', 'adapter_backward', 'save_reload', 'export_parity', 'tokenizer_mapping', 'tensor_mapping')}
            manifest = {'models': {r: model for r in ('teacher', 'original_base', 'student')}, 'evidence': evidence}
            self.assertEqual(preflight(manifest, root), [])
            (root / 'receipt.json').write_text('tampered')
            self.assertEqual(len(preflight(manifest, root)), 6)

    def test_create_only_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.jsonl'
            write_jsonl(p, [{'ok': True}])
            with self.assertRaises(FileExistsError):
                write_jsonl(p, [])


if __name__ == '__main__':
    unittest.main()
