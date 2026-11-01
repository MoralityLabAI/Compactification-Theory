"""Independent checks of the G1 council engine, scenario suite and matched baseline."""
import copy
import random
import unittest

from council import (
    AUTHORITIES, NETWORK, OPERATOR, RATIFIER, REGISTRY, Constitution, Council, Keyring,
    audit_chain, digest, replay,
)
from g1_eval import (
    ARCHITECTURES, FAMILIES, GRID_P_C, GRID_P_O, Harness, closed_form, council_constitution,
    fault_outcomes, fault_rates, run_family,
)

C5 = council_constitution()
KEYS = Keyring(AUTHORITIES + C5.members)


def ready_council():
    c = Council(C5, KEYS)
    c.process(KEYS.sign(OPERATOR, "open_request", {"request": "r1", "scope": "publish_result", "evidence": "e1"}))
    c.process(KEYS.sign(OPERATOR, "grant_mandate", {}))
    c.process(KEYS.sign(REGISTRY, "validate_evidence", {"evidence": "e1"}))
    return c


def vote(c, member, approve=True, rid="r1"):
    return c.process(KEYS.sign(member, "vote", {"request": rid, "approve": approve, "reason": "", **c.ballot()}))


class TestConstitution(unittest.TestCase):
    def test_quorum_must_be_strict_majority(self):
        with self.assertRaises(ValueError):
            Constitution(1, ("a", "b", "c", "d"), 2, ("s",)).validate()
        Constitution(1, ("a", "b", "c", "d"), 3, ("s",)).validate()

    def test_member_cannot_hold_authority_identity(self):
        with self.assertRaises(ValueError):
            Constitution(1, ("a", OPERATOR, "c"), 2, ("s",)).validate()

    def test_hash_depends_on_content(self):
        other = Constitution(1, C5.members, C5.quorum, C5.scopes + ("x",))
        self.assertNotEqual(C5.hash, other.hash)


class TestAuthentication(unittest.TestCase):
    def test_tampered_body_rejected(self):
        c = ready_council()
        msg = KEYS.sign("m0", "vote", {"request": "r1", "approve": False, "reason": "", **c.ballot()})
        tampered = type(msg)(msg.kind, msg.sender, {**msg.body, "approve": True}, msg.sig)
        self.assertEqual(c.process(tampered), (False, "bad_signature"))

    def test_forged_and_unknown_senders_rejected(self):
        c = Council(C5, KEYS)
        self.assertEqual(c.process(KEYS.forge("m0", OPERATOR, "grant_mandate", {})), (False, "bad_signature"))
        self.assertEqual(c.process(KEYS.forge("m0", "director", "grant_mandate", {})), (False, "unknown_sender"))
        self.assertFalse(c.mandate_active)

    def test_roles(self):
        c = ready_council()
        self.assertEqual(c.process(KEYS.sign("m0", "revoke_mandate", {})), (False, "role"))
        self.assertEqual(c.process(KEYS.sign(OPERATOR, "vote", {"request": "r1", "approve": True, **c.ballot()})), (False, "role"))
        self.assertEqual(c.process(KEYS.sign("m1", "validate_evidence", {"evidence": "e9"})), (False, "role"))
        self.assertTrue(c.mandate_active)

    def test_partitioned_member_cannot_vote(self):
        c = ready_council()
        c.process(KEYS.sign(NETWORK, "link_down", {"member": "m0"}))
        self.assertEqual(vote(c, "m0"), (False, "partitioned"))

    def test_malformed_body_rejected_not_crashing(self):
        c = ready_council()
        self.assertEqual(c.process(KEYS.sign("m0", "vote", {"request": "r1"})), (False, "malformed"))


class TestQuorumRevocationVersioning(unittest.TestCase):
    def test_quorum_counts_only_linked_current_votes(self):
        c = ready_council()
        for m in ("m0", "m1"):
            vote(c, m)
        self.assertEqual(c.decide("r1")["outcome"], "hold")
        vote(c, "m2")
        self.assertEqual(c.decide("r1")["outcome"], "approve")
        c.process(KEYS.sign(NETWORK, "link_down", {"member": "m2"}))
        self.assertEqual(c.decide("r1")["outcome"], "hold")
        c.process(KEYS.sign(NETWORK, "link_up", {"member": "m2"}))
        self.assertEqual(c.decide("r1")["outcome"], "approve")

    def test_old_epoch_vote_rejected_after_regrant(self):
        c = ready_council()
        old = KEYS.sign("m0", "vote", {"request": "r1", "approve": True, "reason": "", **c.ballot()})
        c.process(KEYS.sign(OPERATOR, "revoke_mandate", {}))
        c.process(KEYS.sign(OPERATOR, "grant_mandate", {}))
        self.assertEqual(c.process(old), (False, "stale_epoch"))

    def test_ratification_requires_ratifier_and_invalidates_votes(self):
        c = ready_council()
        for m in ("m0", "m1", "m2"):
            vote(c, m)
        new = Constitution(2, C5.members, 3, C5.scopes + ("release_dataset",))
        c.process(KEYS.sign("m0", "propose_amendment", {"constitution": new.to_json()}))
        self.assertEqual(c.process(KEYS.sign("m1", "ratify_amendment", {"constitution": new.hash})), (False, "role"))
        self.assertEqual(c.constitution, C5)
        old_vote = KEYS.sign("m3", "vote", {"request": "r1", "approve": True, "reason": "", **c.ballot()})
        self.assertTrue(c.process(KEYS.sign(RATIFIER, "ratify_amendment", {"constitution": new.hash}))[0])
        self.assertEqual(c.constitution, new)
        self.assertEqual(c.decide("r1")["outcome"], "hold")
        self.assertEqual(c.process(old_vote), (False, "stale_constitution"))

    def test_amendment_must_increment_version(self):
        c = ready_council()
        skip = Constitution(3, C5.members, 3, C5.scopes)
        self.assertEqual(c.process(KEYS.sign("m0", "propose_amendment", {"constitution": skip.to_json()})), (False, "bad_version"))


class TestReplayAndAudit(unittest.TestCase):
    def setUp(self):
        self.h, _ = FAMILIES["R01"]("council5_q3")
        self.log = self.h.council.log

    def test_replay_is_exact(self):
        self.assertTrue(replay(self.h.genesis, self.h.keyring, self.log)["ok"])

    def test_two_runs_have_identical_logs(self):
        other, _ = FAMILIES["R01"]("council5_q3")
        self.assertEqual(other.council.log[-1]["hash"], self.log[-1]["hash"])

    def test_edit_without_rechaining_breaks_hash_chain(self):
        log = copy.deepcopy(self.log)
        log[3]["accepted"] = not log[3]["accepted"]
        self.assertEqual(replay(self.h.genesis, self.h.keyring, log)["reason"], "hash_chain")

    def test_rechained_vote_tampering_is_caught_by_signatures(self):
        log = copy.deepcopy(self.log)
        i = next(i for i, e in enumerate(log) if e["type"] == "message" and e["message"]["kind"] == "vote" and e["accepted"])
        log[i]["message"]["body"]["reason"] = "edited after the fact"
        prev = log[i - 1]["hash"] if i else "0" * 64
        for e in log[i:]:   # attacker recomputes the whole chain
            e["prev"] = prev
            e["hash"] = digest({k: v for k, v in e.items() if k != "hash"})
            prev = e["hash"]
        report = replay(self.h.genesis, self.h.keyring, log)
        self.assertFalse(report["ok"])
        self.assertEqual((report["index"], report["reason"]), (i, "message_outcome"))

    def test_audit_flags_fabricated_approval(self):
        i = next(i for i, e in enumerate(self.log) if e["type"] == "decision" and e["decision"]["outcome"] == "approve")
        approved = self.log[i]["decision"]
        later = next(j for j, e in enumerate(self.log) if j > i and e["type"] == "decision")
        log = copy.deepcopy(self.log)
        log[later]["decision"] = {**approved}   # claim the pre-revocation authority still holds
        audit = audit_chain(self.h.genesis, log, later)
        self.assertFalse(audit["closed"])
        self.assertIn("mandate_revoked", audit["missing"])


class TestScenarioSuite(unittest.TestCase):
    def test_every_family_matches_expectation_replays_and_audits(self):
        for family in FAMILIES:
            for arch in ARCHITECTURES:
                with self.subTest(family=family, arch=arch):
                    r = run_family(family, arch)
                    self.assertTrue(r["matches_expected"], r["steps"])
                    self.assertTrue(r["replay_ok"])
                    self.assertEqual(r["audit_closed"], r["approvals"])

    def test_dissent_retained_through_replay(self):
        h, steps = FAMILIES["D01"]("council5_q3")
        self.assertEqual(steps[0][1]["outcome"], "approve")
        self.assertEqual([m for m, _ in steps[0][1]["dissent"]], ["m4"])
        self.assertTrue(replay(h.genesis, h.keyring, h.council.log)["ok"])

    def test_consensus_is_not_correctness(self):
        for arch in ARCHITECTURES:
            r = run_family("C01", arch)
            self.assertTrue(r["matches_expected"])
            self.assertFalse(r["matches_ground_truth"])


class TestMatchedBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {arch: fault_outcomes(arch) for arch in ARCHITECTURES}

    def test_engine_rates_equal_binomial_closed_form(self):
        sizes = {"council5_q3": (5, 3), "controller": (1, 1), "controller_x5": (1, 1)}
        for arch, (n, q) in sizes.items():
            for pc in GRID_P_C:
                for po in GRID_P_O:
                    got = fault_rates(self.rows[arch], pc, po)
                    want = closed_form(n, q, pc, po)
                    self.assertAlmostEqual(got["wrong_approval"], want["wrong_approval"], places=12)
                    self.assertAlmostEqual(got["needless_hold"], want["needless_hold"], places=12)

    def test_extra_samples_of_a_deterministic_controller_change_nothing(self):
        for pc in GRID_P_C:
            a = fault_rates(self.rows["controller"], pc, 0.1)
            b = fault_rates(self.rows["controller_x5"], pc, 0.1)
            self.assertEqual((a["wrong_approval"], a["needless_hold"]), (b["wrong_approval"], b["needless_hold"]))
            self.assertAlmostEqual(b["evaluations_per_decision"], 5 * 0.9)


def oracle_outcomes(messages, members, quorum):
    """Brute-force reading of the rules from the raw message stream (no engine code)."""
    mandate, epoch, evidence, linked, votes, out = False, 0, False, set(members), {}, []
    for msg in messages:
        if msg == "decide":
            yes = [m for m, (a, ep) in votes.items() if a and ep == epoch and m in linked]
            out.append("approve" if mandate and evidence and len(yes) >= quorum else "hold")
            continue
        if not KEYS.verify(msg):
            continue
        s, k, b = msg.sender, msg.kind, msg.body
        if s == OPERATOR and k == "grant_mandate" and not mandate:
            mandate, epoch = True, epoch + 1
        elif s == OPERATOR and k == "revoke_mandate":
            mandate, votes = False, {}
        elif s == REGISTRY and k in ("validate_evidence", "invalidate_evidence") and b["evidence"] == "e1":
            evidence = k == "validate_evidence"
        elif s == NETWORK and k in ("link_down", "link_up"):
            (linked.discard if k == "link_down" else linked.add)(b["member"])
        elif s in members and k == "vote" and s in linked and mandate and b["epoch"] == epoch:
            votes[s] = (b["approve"], b["epoch"])
    return out


class TestRandomTracesAgainstOracle(unittest.TestCase):
    def test_engine_matches_oracle_and_never_breaches_authority(self):
        rng = random.Random(20261101)
        members = C5.members
        approvals = 0
        for _ in range(300):
            c = Council(C5, KEYS)
            c.process(KEYS.sign(OPERATOR, "open_request", {"request": "r1", "scope": "publish_result", "evidence": "e1"}))
            stream, engine = [], []
            for _ in range(40):
                r = rng.random()
                m = rng.choice(members)
                if r < 0.08:
                    msg = KEYS.sign(OPERATOR, rng.choice(["grant_mandate", "revoke_mandate"]), {})
                elif r < 0.16:
                    msg = KEYS.sign(REGISTRY, rng.choice(["validate_evidence", "invalidate_evidence"]), {"evidence": "e1"})
                elif r < 0.30:
                    msg = KEYS.sign(NETWORK, rng.choice(["link_down", "link_up"]), {"member": m})
                elif r < 0.35:
                    msg = KEYS.forge(m, OPERATOR, "grant_mandate", {"claim": "trust me"})
                elif r < 0.38:
                    msg = KEYS.sign(m, "grant_mandate", {"claim": "trust me"})
                elif r < 0.42:   # replayed vote addressed to an old epoch
                    msg = KEYS.sign(m, "vote", {"request": "r1", "approve": True, "reason": "", "epoch": max(0, c.mandate_epoch - 1), "constitution": C5.hash})
                elif r < 0.80:
                    msg = KEYS.sign(m, "vote", {"request": "r1", "approve": rng.random() < 0.7, "reason": "", **c.ballot()})
                else:
                    stream.append("decide")
                    d = c.decide("r1")
                    engine.append(d["outcome"])
                    if d["outcome"] == "approve":
                        approvals += 1
                        self.assertTrue(all(d["checks"].values()))
                    continue
                stream.append(msg)
                c.process(msg)
            self.assertEqual(engine, oracle_outcomes(stream, members, C5.quorum))
            self.assertTrue(replay(C5, KEYS, c.log)["ok"])
        self.assertGreater(approvals, 50)


if __name__ == "__main__":
    unittest.main()
