"""Designer boundary tests; transport is injected or loopback-only."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from research.pluto_gym.design import (
    SCHEMA, DesignBudget, DesignerError, OfflineDesigner, WebhookDesigner,
    make_request, validate_response,
)
from research.pluto_gym.io import canonical, digest, strict_json


def request_for(*, model="luna-test", reasoning="high", iteration=0,
                max_output_tokens=800):
    return make_request(
        iteration=iteration, backend="synthetic_multifront", model=model,
        reasoning=reasoning, max_output_tokens=max_output_tokens,
        candidates=[{"variant_id": "accessible", "rules": {
            "fronts": 4, "deadline_ticks": 12, "target_xor": 0,
            "reward_weights": [1, 1, 1, 1]}},
                    {"variant_id": "mistakes", "rules": {
            "fronts": 4, "deadline_ticks": 12, "target_xor": 1,
            "reward_weights": [1, 1, 1, 1]}}],
        training_summary={
            "accessible": {"mean_reward": 0.25, "episodes": 4,
                           "interface_oracle_value": 0.25, "mean_interface_regret": 0.0,
                           "heldout_seeds": [99999], "evaluation_transcript": "SEALED"},
            "mistakes": {"mean_reward": 0.5, "episodes": 4,
                         "interface_oracle_value": 1.0, "mean_interface_regret": 0.5,
                         "test_results": {"sentinel": "SEALED"}},
        }, run_id="test-run")


def response_for(request, **changes):
    result = {"schema": SCHEMA, "request_id": request["request_id"],
              "variant_id": "mistakes", "rationale": "Training regret is largest."}
    result.update(changes)
    return result


@contextmanager
def local_broker(mode):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            request = strict_json(body)
            received.append({"request": request, "headers": dict(self.headers), "path": self.path})
            if mode == "redirect":
                self.send_response(307)
                self.send_header("Location", "/redirected")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if mode == "slow_headers":
                try:
                    self.wfile.write(b"HTTP/1.1 200 OK\r\n")
                    self.wfile.flush()
                    # Frequent bytes defeat an inactivity-only timeout. The
                    # elapsed parent deadline must still terminate the worker.
                    for j in range(30):
                        self.wfile.write(f"X-Slow-{j}: alive\r\n".encode())
                        self.wfile.flush()
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            payload = canonical(response_for(request))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError as exc:
        raise unittest.SkipTest("loopback sockets are disabled by this execution sandbox") from exc
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost",
                                     "no_proxy": "127.0.0.1,localhost"}):
            yield f"http://127.0.0.1:{server.server_port}/design", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


class DesignTests(unittest.TestCase):
    def test_strict_json_rejects_duplicates_and_nonfinite_numbers(self):
        for data in (b'{"x":1,"x":2}', b'{"nested":{"x":1,"x":2}}',
                     b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}',
                     b'{"x":1e999}', b'{"x":-1e999}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                strict_json(data)
        self.assertEqual(strict_json(b'{"x":"NaN","y":1e3}'), {"x": "NaN", "y": 1000.0})
        with self.assertRaises(ValueError):
            strict_json(b'{"x":1}', max_bytes=2)

    def test_summary_export_whitelist_and_request_digest(self):
        request = request_for()
        encoded = canonical(request).decode()
        for excluded in ("SEALED", "99999", "heldout_seeds", "test_results"):
            self.assertNotIn(excluded, encoded)
        for row in request["training_summary"].values():
            self.assertEqual(set(row), {"mean_reward", "episodes", "interface_oracle_value",
                                        "mean_interface_regret"})
        self.assertEqual(request["request_id"], digest({k: v for k, v in request.items() if k != "request_id"}))

    def test_candidate_boundary_rejects_extra_metadata(self):
        request = request_for()
        candidates = request["candidates"]
        candidates[0]["heldout_seeds"] = [99999]
        with self.assertRaises(DesignerError):
            make_request(iteration=0, backend="synthetic_multifront", candidates=candidates,
                         training_summary=request["training_summary"])

    def test_response_binding_grammar_and_byte_limit(self):
        request = request_for()
        self.assertEqual(validate_response(canonical(response_for(request)), request)["variant_id"], "mistakes")
        variants = [response_for(request, request_id="wrong"),
                    response_for(request, variant_id="unknown"),
                    response_for(request, schema="wrong"),
                    response_for(request, command="launch arbitrary code"),
                    response_for(request, rationale="x" * 2001)]
        for response in variants:
            with self.subTest(response=response), self.assertRaises(DesignerError):
                validate_response(response, request)
        with self.assertRaises(DesignerError):
            validate_response(response_for(request, rationale="x" * 500), request, max_bytes=256)
        duplicate = canonical(response_for(request)).decode().replace(
            '"variant_id":"mistakes"', '"variant_id":"accessible","variant_id":"mistakes"')
        with self.assertRaises(DesignerError):
            validate_response(duplicate, request)

    def test_offline_policy_uses_regret_instead_of_low_raw_reward(self):
        request = request_for(model="offline")
        designer = OfflineDesigner(DesignBudget(max_calls=1))
        self.assertLess(request["training_summary"]["accessible"]["mean_reward"],
                        request["training_summary"]["mistakes"]["mean_reward"])
        self.assertEqual(designer.propose(request)["variant_id"], "mistakes")
        with self.assertRaises(DesignerError):
            designer.propose(request)
        self.assertEqual(designer.ledger()["network_calls"], 0)

    def test_failed_call_consumes_budget_and_request_id(self):
        def fail(*args):
            raise DesignerError("injected transport failure")
        designer = WebhookDesigner("https://broker.invalid/design", model="luna-test",
                                   budget=DesignBudget(max_calls=2), exchange=fail)
        request = request_for()
        with self.assertRaises(DesignerError):
            designer.propose(request)
        self.assertEqual(designer.calls, 1)
        with self.assertRaises(DesignerError):
            designer.propose(request)
        self.assertEqual(designer.calls, 1)
        with self.assertRaises(DesignerError):
            designer.propose(request_for(iteration=1))
        self.assertEqual(designer.calls, 2)
        with self.assertRaises(DesignerError):
            designer.propose(request_for(iteration=2))

    def test_route_effort_token_limit_and_hash_are_enforced_before_transport(self):
        called = []
        designer = WebhookDesigner("https://broker.invalid/design", model="luna-test",
                                   budget=DesignBudget(max_output_tokens=800),
                                   exchange=lambda *args: called.append(args))
        altered = request_for()
        altered["iteration"] = 9
        for request in (request_for(model="other"), request_for(reasoning="medium"),
                        request_for(max_output_tokens=801), altered):
            with self.subTest(request=request), self.assertRaises(DesignerError):
                designer.propose(request)
        self.assertEqual(called, [])
        self.assertEqual(designer.calls, 0)

    def test_auth_is_transport_only_and_configuration_is_bound(self):
        observed = []
        def exchange(endpoint, body, headers, budget):
            observed.append((endpoint, body, headers, budget))
            return canonical(response_for(strict_json(body)))
        endpoint = "https://broker.invalid/design"
        designer = WebhookDesigner(endpoint, model="luna-test", token="dummy-test-secret",
                                   exchange=exchange)
        designer.propose(request_for())
        self.assertEqual(observed[0][2]["Authorization"], "Bearer dummy-test-secret")
        self.assertNotIn(b"dummy-test-secret", observed[0][1])
        for exported in (designer.configuration(), designer.ledger()):
            text = canonical(exported).decode()
            self.assertNotIn("dummy-test-secret", text)
            self.assertNotIn(endpoint, text)
            self.assertEqual(exported["endpoint_sha256"], digest(endpoint))
        changed = WebhookDesigner(endpoint, model="luna-test", reasoning="medium")
        self.assertNotEqual(digest(designer.configuration()), digest(changed.configuration()))
        changed = WebhookDesigner(endpoint + "/other", model="luna-test")
        self.assertNotEqual(designer.configuration()["endpoint_sha256"], changed.configuration()["endpoint_sha256"])
        changed = WebhookDesigner(endpoint, model="luna-test", budget=DesignBudget(max_calls=1))
        self.assertNotEqual(digest(designer.configuration()), digest(changed.configuration()))

    def test_input_byte_budget_prevents_dispatch(self):
        base = request_for()
        candidates = [{"variant_id": f"variant_{j}", "rules": base["candidates"][0]["rules"]}
                      for j in range(16)]
        summary = {row["variant_id"]: base["training_summary"]["accessible"] for row in candidates}
        request = make_request(iteration=0, backend="synthetic_multifront", model="luna-test",
                               candidates=candidates, training_summary=summary)
        called = []
        designer = WebhookDesigner("https://broker.invalid/design", model="luna-test",
                                   budget=DesignBudget(max_input_bytes=1024),
                                   exchange=lambda *args: called.append(args))
        with self.assertRaises(DesignerError):
            designer.propose(request)
        self.assertEqual(called, [])
        self.assertEqual(designer.calls, 0)

    def test_oversized_transport_result_consumes_call_and_fails_closed(self):
        designer = WebhookDesigner("https://broker.invalid/design", model="luna-test",
                                   budget=DesignBudget(max_response_bytes=256),
                                   exchange=lambda *args: b"x" * 257)
        with self.assertRaises(DesignerError):
            designer.propose(request_for())
        self.assertEqual(designer.calls, 1)
        self.assertEqual(designer.response_bytes, 257)

    def test_loopback_broker_roundtrip(self):
        with local_broker("ok") as (endpoint, observed):
            designer = WebhookDesigner(endpoint, model="luna-test", token="loopback-only",
                                       budget=DesignBudget(timeout_seconds=2))
            response = designer.propose(request_for())
            self.assertEqual(response["variant_id"], "mistakes")
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["headers"]["Authorization"], "Bearer loopback-only")

    def test_loopback_redirect_is_rejected_without_following(self):
        with local_broker("redirect") as (endpoint, observed):
            designer = WebhookDesigner(endpoint, model="luna-test",
                                       budget=DesignBudget(timeout_seconds=2))
            with self.assertRaises(DesignerError):
                designer.propose(request_for())
            self.assertEqual(len(observed), 1)
            self.assertEqual(designer.calls, 1)

    def test_parent_timeout_bounds_trickled_headers(self):
        with local_broker("slow_headers") as (endpoint, observed):
            designer = WebhookDesigner(endpoint, model="luna-test",
                                       budget=DesignBudget(timeout_seconds=0.25))
            start = time.monotonic()
            with self.assertRaises(DesignerError):
                designer.propose(request_for())
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.9)
            self.assertGreater(elapsed, 0.15)
            self.assertEqual(designer.calls, 1)
            self.assertEqual(len(observed), 1)

    def test_actual_subprocess_timeout_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary) / "sleep_worker.py"
            worker.write_text("import time\ntime.sleep(2)\n", encoding="utf-8")
            with patch("research.pluto_gym.design.Path") as paths:
                paths.return_value.with_name.return_value = worker
                designer = WebhookDesigner("https://broker.invalid/design", model="luna-test",
                                           budget=DesignBudget(timeout_seconds=0.15))
                start = time.monotonic()
                with self.assertRaisesRegex(DesignerError, "TimeoutExpired"):
                    designer.propose(request_for())
                self.assertLess(time.monotonic() - start, 0.8)
                self.assertEqual(designer.calls, 1)

    def test_redirect_handler_rejects_before_followup_request(self):
        from research.pluto_gym._http_worker import NoRedirect
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(None, None, 307, "redirect", {},
                                          "http://127.0.0.1/redirected")


if __name__ == "__main__":
    unittest.main()
