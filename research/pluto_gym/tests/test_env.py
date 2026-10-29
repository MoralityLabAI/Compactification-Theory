"""Independent finite checks of the synthetic environment and IRC references."""

import itertools
import math
import unittest

from research.pluto_gym.env import InterfaceSpec, MultiFrontEnv, RuleSpec, freeze_observation, optimal_schedule
from research.pluto_gym.math_ref import binary_entropy, deadline_capacity, gf2_rank, query_memory_lower_bound, reset_required_bits


class SpecTests(unittest.TestCase):
    def test_round_trip_and_strict_validation(self):
        for spec in (RuleSpec(), InterfaceSpec()):
            self.assertEqual(spec, type(spec).from_dict(spec.to_dict()))
        invalid_rules = [dict(fronts=True), dict(deadline_ticks=1.2), dict(target_xor=float("nan")),
                         dict(reward_weights=[1, 1, 1, False]), dict(reward_weights=[1]),
                         dict(reward_weights=[1, 1, 1, 0]), dict(fronts=9)]
        invalid_interfaces = [dict(command_ticks=True), dict(switch_ticks=float("inf")),
                              dict(switch_ticks=-1), dict(command_ticks=0), dict(observation="all")]
        for params in invalid_rules:
            with self.subTest(params=params), self.assertRaises(ValueError):
                RuleSpec(**params)
        for params in invalid_interfaces:
            with self.subTest(params=params), self.assertRaises(ValueError):
                InterfaceSpec(**params)
        with self.assertRaises(ValueError):
            RuleSpec.from_dict({"typo": 1})
        with self.assertRaises(ValueError):
            InterfaceSpec.from_dict({"latency": 1})
        try:
            import numpy as np
        except ImportError:
            return
        for value in (np.bool_(False), np.bool_(True)):
            with self.assertRaises(ValueError):
                RuleSpec(target_xor=value)
            with self.assertRaises(ValueError):
                deadline_capacity(4, value, 1, 10)
        self.assertEqual(InterfaceSpec(command_ticks=np.int64(3)).command_ticks, 3)


class EnvironmentTests(unittest.TestCase):
    def test_seeded_replay_and_oracle_value(self):
        traces = []
        for _ in range(2):
            env = MultiFrontEnv()
            first = env.reset(seed=72)
            expected = env.oracle_value()
            trace, total = [first], 0
            while env.legal_actions():
                result = env.step(env.optimal_action())
                total += result[1]
                trace.append(result)
            self.assertEqual(expected, total)
            self.assertEqual(total, 0.5)
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_hidden_cues_do_not_change_observation_info_or_schedule(self):
        env = MultiFrontEnv()
        initial, info = env.reset(seed=5)
        state = freeze_observation(initial)
        old_cues = env._cues
        env._cues = (old_cues[0],) + tuple(1 - x for x in old_cues[1:])
        self.assertEqual(state, freeze_observation(env.observation()))
        self.assertEqual(info, env._info("reset", 0))
        self.assertEqual(env.optimal_front(), 0)
        self.assertNotIn("seed", info)
        self.assertNotIn("cues", info)
        self.assertTrue(all(c is None for c in initial["visible_cues"][1:]))
        env.step({"kind": "switch", "front": 1})
        self.assertEqual(env.observation()["visible_cues"][1], 1 - old_cues[1])
        self.assertIsNone(env.observation()["visible_cues"][0])

    def test_camera_and_global_addressing_costs(self):
        camera = MultiFrontEnv()
        camera.reset(seed=1)
        with self.assertRaises(ValueError):
            camera.step({"kind": "attack", "front": 1, "target": 0})
        camera.step({"kind": "switch", "front": 1})
        self.assertEqual(camera.tick, 4)
        cue = camera.observation()["visible_cues"][1]
        camera.step({"kind": "attack", "front": 1, "target": cue})
        self.assertEqual(camera.tick, 7)
        global_env = MultiFrontEnv(interface=InterfaceSpec(name="global", observation="global"))
        obs, _ = global_env.reset(seed=1)
        self.assertNotIn(None, obs["visible_cues"])
        global_env.step({"kind": "attack", "front": 1, "target": obs["visible_cues"][1]})
        self.assertEqual(global_env.tick, 7)

    def test_invalid_actions_leave_every_state_field_unchanged(self):
        env = MultiFrontEnv()
        env.reset(seed=6)
        invalid = [None, {}, {"kind": "attack", "front": True, "target": 0},
                   {"kind": "attack", "front": 0, "target": 0.5},
                   {"kind": "attack", "front": 0, "target": float("nan")},
                   {"kind": "switch", "front": 0}, {"kind": "switch", "front": 8},
                   {"kind": "wait", "ticks": 0}, {"kind": "wait", "ticks": True},
                   {"kind": "attack", "front": 0, "target": 0, "extra": 1}]
        for action in invalid:
            before = (freeze_observation(env.observation()), env._steps, env._return_weight,
                      env._terminated, env._truncated, env._cues)
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
            after = (freeze_observation(env.observation()), env._steps, env._return_weight,
                     env._terminated, env._truncated, env._cues)
            self.assertEqual(before, after)

    def test_deadline_equality_and_overrun(self):
        for deadline, expected in [(2, 0), (3, 0.25)]:
            env = MultiFrontEnv(RuleSpec(deadline_ticks=deadline))
            obs, _ = env.reset(seed=1)
            result = env.step({"kind": "attack", "front": 0, "target": obs["visible_cues"][0]})
            self.assertEqual(result[1], expected)
            self.assertTrue(result[2])
            self.assertFalse(result[3])
            self.assertEqual(env.tick, deadline)
            self.assertEqual(result[0]["resolved"][0], deadline == 3)
            with self.assertRaises(RuntimeError):
                env.step({"kind": "wait", "ticks": 1})
            env.reset(seed=1)
            self.assertEqual(env.tick, 0)

    def test_wrong_attack_resolves_front_once_and_public_rule_changes_target(self):
        env = MultiFrontEnv(RuleSpec(target_xor=1))
        obs, _ = env.reset(seed=9)
        result = env.step({"kind": "attack", "front": 0, "target": obs["visible_cues"][0]})
        self.assertEqual(result[1], 0)
        self.assertEqual(result[0]["status"][0], "miss")
        with self.assertRaises(ValueError):
            env.step({"kind": "attack", "front": 0, "target": 1 - obs["visible_cues"][0]})

    def test_zero_cost_switch_loop_truncates(self):
        env = MultiFrontEnv(interface=InterfaceSpec(switch_ticks=0))
        env.reset(seed=1)
        for _ in range(env.decision_limit):
            result = env.step({"kind": "switch", "front": 1 - env.current_front})
        self.assertEqual(env.tick, 0)
        self.assertFalse(result[2])
        self.assertTrue(result[3])
        self.assertEqual(env.oracle_value(), 0)
        with self.assertRaises(RuntimeError):
            env.optimal_action()

    def test_oracle_counts_decisions_remaining(self):
        env = MultiFrontEnv(interface=InterfaceSpec(switch_ticks=0))
        env.reset(seed=1)
        for _ in range(env.decision_limit - 1):
            env.step({"kind": "switch", "front": 1 - env.current_front})
        self.assertEqual(env.oracle_value(), 0.25)
        result = env.step(env.optimal_action())
        self.assertEqual(result[1], 0.25)
        self.assertTrue(result[3])

    def test_observations_are_detached_copies(self):
        env = MultiFrontEnv()
        obs, _ = env.reset(seed=1)
        frozen = freeze_observation(env.observation())
        obs["rules"]["reward_weights"][0] = 99
        obs["visible_cues"][0] = 22
        obs["status"][0] = "hit"
        self.assertEqual(frozen, freeze_observation(env.observation()))

    def test_oracle_matches_independent_ordered_subsets(self):
        settings = 0
        for n, c, s, deadline in itertools.product(range(1, 5), (1, 3), (0, 2), (1, 4, 9, 13)):
            weights = tuple(range(1, n + 1))
            rules = RuleSpec(fronts=n, deadline_ticks=deadline, reward_weights=weights)
            interface = InterfaceSpec(command_ticks=c, switch_ticks=s)
            best = 0
            for size in range(n + 1):
                for order in itertools.permutations(range(n), size):
                    time, selected = 0, 0
                    for front in order:
                        time += c + (s if front != selected else 0)
                        selected = front
                    if time <= deadline:
                        best = max(best, sum(weights[i] for i in order))
            value, _ = optimal_schedule(rules, interface)
            self.assertEqual(value, best)
            for observation in ("camera", "global"):
                env = MultiFrontEnv(rules, InterfaceSpec(observation=observation, command_ticks=c, switch_ticks=s))
                env.reset(seed=2)
                result = 0
                while env.legal_actions():
                    result += env.step(env.optimal_action())[1]
                self.assertAlmostEqual(result, best / sum(weights))
            settings += 1
        self.assertEqual(settings, 64)

    def test_current_state_oracle_excludes_already_resolved_fronts(self):
        env = MultiFrontEnv(RuleSpec(reward_weights=(1, 9, 2, 1)))
        env.reset(seed=1)
        self.assertEqual(env.optimal_front(), 0)
        initial = env.oracle_value()
        reward = env.step(env.optimal_action())[1]
        self.assertAlmostEqual(initial, reward + env.oracle_value())


class MathReferenceTests(unittest.TestCase):
    def test_deadline_formula_by_independent_enumeration(self):
        count = 0
        for n, c, s, D in itertools.product(range(1, 7), (1, 3), (0, 2), range(10)):
            expected = max([0] + [k for k in range(1, n + 1) if k * c + (k - 1) * s <= D])
            self.assertEqual(deadline_capacity(n, c, s, D), expected)
            count += 1
        self.assertEqual(count, 240)

    def test_reset_rank_from_distinguishable_consequences(self):
        for n in range(1, 7):
            parity_row = (1 << n) - 1
            for mask in range(1 << n):
                addressable = [i for i in range(n) if mask & (1 << i)]
                rows = [parity_row] + [parity_row ^ (1 << i) for i in addressable]
                consequences = {tuple((x & row).bit_count() % 2 for row in rows) for x in range(1 << n)}
                bits = reset_required_bits(n, addressable)
                self.assertEqual(len(consequences), 1 << bits)
                self.assertEqual(gf2_rank(rows, width=n), bits)

    def test_information_bound_and_invalid_inputs(self):
        self.assertEqual(binary_entropy(0), 0)
        self.assertEqual(binary_entropy(0.5), 1)
        self.assertEqual(query_memory_lower_bound(64, 0), 64)
        self.assertEqual(query_memory_lower_bound(64, 0.5), 0)
        self.assertEqual(math.ceil(query_memory_lower_bound(64, 0.05)), 46)
        for p in (True, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(p=p), self.assertRaises(ValueError):
                binary_entropy(p)
        for args in ((2, [0, 0]), (2, [2]), (2, [False]), (2, [0.5])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                reset_required_bits(*args)
        with self.assertRaises(ValueError):
            deadline_capacity(2, True, 1, 10)
        with self.assertRaises(ValueError):
            gf2_rank([4], width=2)


if __name__ == "__main__":
    unittest.main()
