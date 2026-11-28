"""Independent checks for the Plan R toy finite-state reference."""

import random
from dataclasses import replace
import unittest

from reference import (ACTIONS, EXPECTED_UNSAFE, MODEL_SHA256, Scenario, State, cases,
                       capability_proxy, distinguishing_word, explore, observe, run, transition)


def walk_all(c):
    """DFS reference oracle independent of explore's BFS/parent bookkeeping."""
    seen = set()
    def visit(s):
        if s in seen:
            return
        seen.add(s)
        for action in ACTIONS:
            nxt = transition(s, action, c)
            if nxt is not None:
                visit(nxt)
    visit(State())
    return seen


class PlanRReferenceTests(unittest.TestCase):
    def test_registered_five_families(self):
        self.assertEqual(len(cases()), 12)
        for name, case in cases().items():
            with self.subTest(name=name):
                result = explore(case)
                self.assertEqual(result['unsafe_reachable'], EXPECTED_UNSAFE[name])
                self.assertEqual(result['model_sha256'], MODEL_SHA256)
                self.assertEqual(result['reachable_states'], len(walk_all(case)))

    def test_fixed_policy_capability_gain_without_gradients(self):
        locked = explore(cases()['01_memory_off'])
        enabled = explore(cases()['01_memory_on'])
        self.assertEqual(locked['model_sha256'], enabled['model_sha256'])
        self.assertEqual(locked['max_capability_proxy'], 0)
        self.assertEqual(enabled['max_capability_proxy'], 4)
        self.assertEqual(enabled['shortest_unsafe_witness']['actions'],
                         ['read_a', 'read_b', 'compile', 'write_external'])

    def test_cross_lab_exchange_and_sham(self):
        self.assertEqual(explore(cases()['02_peer_sham'])['max_capability_proxy'], 1)
        self.assertEqual(explore(cases()['02_peer_exchange'])['max_capability_proxy'], 4)
        witness = explore(cases()['02_peer_exchange'])['shortest_unsafe_witness']['actions']
        self.assertIn('receive_peer', witness)

    def test_tool_privilege_and_release_lag(self):
        self.assertFalse(explore(cases()['03_read_only'])['unsafe_reachable'])
        self.assertTrue(explore(cases()['03_write_tool'])['unsafe_reachable'])
        self.assertFalse(explore(cases()['04_escrow_R1'])['unsafe_reachable'])
        self.assertTrue(explore(cases()['04_escrow_R2'])['unsafe_reachable'])
        self.assertFalse(explore(cases()['04_escrow_R2_lag'])['unsafe_reachable'])

    def test_release_lag_delays_instead_of_permanently_preventing(self):
        c = cases()["04_escrow_R2_lag"]
        self.assertFalse(explore(c)["unsafe_reachable"])
        extended = replace(c, horizon=7)
        self.assertTrue(explore(extended)["unsafe_reachable"])

    def test_strict_gateway_is_invariant_for_every_reachable_state(self):
        gated = cases()['05_full_gateway']
        seen = walk_all(gated)
        self.assertTrue(seen)
        self.assertTrue(all(not s.unsafe for s in seen))
        self.assertEqual(max(map(capability_proxy, seen)), 4)
        self.assertTrue(any(s.instances == 2 for s in seen))
        for s in seen:
            self.assertIsNone(transition(s, 'write_external', gated))
            self.assertIsNone(transition(s, 'coordinate', gated))

    def test_trace_is_exact_replay_not_just_a_label(self):
        for name, c in cases().items():
            result = explore(c)
            witness = result['shortest_unsafe_witness']
            if witness is None:
                continue
            state = State()
            self.assertEqual(witness['states'][0], vars(state))
            for i, action in enumerate(witness['actions']):
                state = transition(state, action, c)
                self.assertIsNotNone(state, (name, i))
                self.assertEqual(witness['states'][i + 1], vars(state))
            self.assertTrue(state.unsafe)
            self.assertLessEqual(len(witness['actions']), c.horizon)

    def test_random_bfs_dfs_differential(self):
        rng = random.Random(20261029)
        for i in range(100):
            c = Scenario(name=f'random_{i}', horizon=rng.randrange(0, 7),
                         memory_slots=rng.randrange(0, 3), local_facts=rng.randrange(0, 4),
                         peer_facts=rng.randrange(0, 4), peer_enabled=bool(rng.randrange(2)),
                         max_instances=rng.randrange(1, 4), release_lag=rng.randrange(0, 7),
                         tool_grants=frozenset(x for x in ('write_external', 'coordinate') if rng.randrange(2)))
            states = walk_all(c)
            result = explore(c)
            self.assertEqual(result['reachable_states'], len(states))
            self.assertEqual(result['unsafe_reachable'], any(s.unsafe for s in states))
            self.assertEqual(result['max_capability_proxy'], max(map(capability_proxy, states)))

    def test_time_belongs_in_history_quotient(self):
        c = Scenario('clock', horizon=6, max_instances=2, release_lag=2)
        x = State(tick=1)
        y = State(tick=2)
        self.assertEqual(observe(x), observe(y))
        self.assertEqual(distinguishing_word(x, y, c, 1), ('release',))
        self.assertIsNone(distinguishing_word(x, y, c, 0))

    def test_gate_removes_edges_not_weights(self):
        c = cases()['05_frozen_only']
        g = cases()['05_full_gateway']
        self.assertEqual(explore(c)['model_sha256'], explore(g)['model_sha256'])
        for state in walk_all(g):
            for action in ACTIONS:
                gated = transition(state, action, g)
                if gated is not None:
                    self.assertEqual(gated, transition(state, action, c))

    def test_validate_fail_closed(self):
        invalid = [dict(memory_slots=True), dict(horizon=-1), dict(max_instances=0),
                   dict(tool_grants=frozenset(('shell',))), dict(peer_enabled=1),
                   dict(enforce_gateway=True), dict(approved_actions=frozenset(('bad',)))]
        for kw in invalid:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                Scenario('invalid', **kw)
        with self.assertRaises(ValueError):
            transition(State(), 'TRAIN_WEIGHTS', Scenario('valid'))
        with self.assertRaises(ValueError):
            distinguishing_word(State(), State(), Scenario('valid'), True)

    def test_json_is_deterministic(self):
        self.assertEqual(run(), run())
        self.assertTrue(run()['passed_fixture_expectations'])
        self.assertEqual(set(run()['arms']), set(EXPECTED_UNSAFE))


if __name__ == '__main__':
    unittest.main()
