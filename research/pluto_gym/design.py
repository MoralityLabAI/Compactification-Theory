"""Budgeted, provider-neutral game-design webhook.

The response chooses a complete, prevalidated rule variant. Neither code nor a
launch command can enter this contract. A Luna broker and a future local Bonsai
server can implement the same request/response schema. Calls are synchronous and
bounded; there are no automatic retries or background jobs.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable
from urllib.parse import urlsplit

from .io import canonical, digest, strict_json

SCHEMA = "pluto-gym.design.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class DesignerError(ValueError):
    """Invalid, exhausted or failed design request."""


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise DesignerError(f"{name} must be an integer in {low}..{high}")


@dataclass(frozen=True)
class DesignBudget:
    max_calls: int = 4
    max_input_bytes: int = 32768
    max_response_bytes: int = 16384
    max_output_tokens: int = 800
    timeout_seconds: float = 25.0

    def __post_init__(self):
        for name, low, high in (("max_calls", 0, 100),
                               ("max_input_bytes", 1024, 262144),
                               ("max_response_bytes", 256, 65536),
                               ("max_output_tokens", 64, 4096)):
            _integer(getattr(self, name), name, low, high)
        if (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds)
                or not 0 < self.timeout_seconds <= 30):
            raise DesignerError("timeout_seconds must be finite and in (0,30]")


def make_request(*, iteration: int, backend: str, candidates: list[dict],
                 training_summary: dict, model: str = "offline", reasoning: str = "high",
                 max_output_tokens: int = 800, run_id: str = "local") -> dict:
    _integer(iteration, "iteration", 0, 99)
    _integer(max_output_tokens, "max_output_tokens", 64, 4096)
    if backend not in ("synthetic_multifront", "tensorcraft_variants"):
        raise DesignerError("unknown design backend")
    if not isinstance(model, str) or not _ID.fullmatch(model):
        raise DesignerError("model must be a nonempty configured route ID")
    if reasoning not in ("low", "medium", "high"):
        raise DesignerError("invalid reasoning effort")
    if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
        raise DesignerError("invalid run ID")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 32:
        raise DesignerError("provide 1..32 prevalidated candidates")
    ids, exported_candidates = [], []
    for item in candidates:
        if not isinstance(item, dict) or not isinstance(item.get("variant_id"), str):
            raise DesignerError("candidate requires variant_id")
        ident = item["variant_id"]
        if not _ID.fullmatch(ident):
            raise DesignerError("invalid variant_id")
        if backend == "synthetic_multifront":
            from .env import RuleSpec
            if set(item) != {"variant_id", "rules"}:
                raise DesignerError("synthetic candidate requires only variant_id and rules")
            try:
                rules = RuleSpec.from_dict(item["rules"]).to_dict()
            except (ValueError, TypeError) as exc:
                raise DesignerError("candidate rules failed validation") from exc
            exported_candidates.append({"variant_id": ident, "rules": rules})
        else:
            variants = {"control", "devourer_speed_550", "devourer_cooldown_90",
                        "devourer_speed_550_cooldown_90"}
            if (set(item) != {"variant_id", "description"} or ident not in variants
                    or not isinstance(item["description"], str) or len(item["description"]) > 2000):
                raise DesignerError("native candidate must name a pinned mod variant with a bounded description")
            exported_candidates.append({"variant_id": ident, "description": item["description"]})
        ids.append(ident)
    if len(set(ids)) != len(ids):
        raise DesignerError("duplicate candidate ID")
    # Export a deliberately small whitelist. Evaluation receipts, seeds, paths,
    # observation transcripts and native runtime settings cannot enter the prompt.
    summary = {}
    if not isinstance(training_summary, dict) or set(training_summary) != set(ids):
        raise DesignerError("training summary must cover exactly the candidates")
    for ident in ids:
        row = training_summary[ident]
        if not isinstance(row, dict):
            raise DesignerError("training score must be an object")
        score = row.get("mean_reward")
        count = row.get("episodes")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise DesignerError("training score outside [0,1]")
        _integer(count, "episodes", 1, 10000)
        ceiling = row.get("interface_oracle_value")
        regret = row.get("mean_interface_regret")
        if backend == "synthetic_multifront":
            for name, value in (("interface_oracle_value", ceiling), ("mean_interface_regret", regret)):
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not 0 <= value <= 1):
                    raise DesignerError(f"invalid {name}")
            if score > ceiling + 1e-12 or abs(ceiling - score - regret) > 1e-9:
                raise DesignerError("training reward/ceiling/regret disagree")
        elif ceiling is not None or regret is not None:
            raise DesignerError("native scores do not supply an exact interface oracle")
        summary[ident] = {"mean_reward": float(score), "episodes": count,
                          "interface_oracle_value": ceiling, "mean_interface_regret": regret}
    payload = {
        "schema": SCHEMA, "run_id": str(run_id), "iteration": iteration,
        "backend": backend, "model": model, "reasoning_effort": reasoning,
        "max_output_tokens": max_output_tokens,
        "objective": "Choose one offered rule variant for the next fixed-budget training block.",
        "feedback_scope": "training_partition_only",
        "candidates": exported_candidates, "training_summary": summary,
        "response_contract": {
            "schema": SCHEMA, "request_id": "copy request_id exactly",
            "variant_id": "one offered variant_id", "rationale": "at most 2000 characters",
        },
    }
    payload["request_id"] = digest(payload)
    return payload


def validate_response(raw: bytes | str | dict, request: dict, *, max_bytes: int = 16384) -> dict:
    try:
        response = strict_json(canonical(raw) if isinstance(raw, dict) else raw,
                               max_bytes=max_bytes)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise DesignerError("designer returned invalid bounded JSON") from exc
    expected = {"schema", "request_id", "variant_id", "rationale"}
    if not isinstance(response, dict) or set(response) != expected:
        raise DesignerError("response must contain exactly the response-contract fields")
    if response["schema"] != SCHEMA or response["request_id"] != request["request_id"]:
        raise DesignerError("response schema/request binding mismatch")
    offered = {c["variant_id"] for c in request["candidates"]}
    if not isinstance(response["variant_id"], str) or response["variant_id"] not in offered:
        raise DesignerError("designer selected an unoffered rule variant")
    if not isinstance(response["rationale"], str) or len(response["rationale"]) > 2000:
        raise DesignerError("invalid rationale length")
    return response


class OfflineDesigner:
    """Deterministic train-score curriculum control for offline experiments."""
    profile = "offline_largest_training_interface_regret"
    model = "offline"
    reasoning = "high"

    def __init__(self, budget: DesignBudget | None = None):
        self.budget = budget or DesignBudget()
        self.calls = 0

    def propose(self, request: dict) -> dict:
        if self.calls >= self.budget.max_calls:
            raise DesignerError("designer call budget exhausted")
        if len(canonical(request)) > self.budget.max_input_bytes:
            raise DesignerError("design request exceeds input byte budget")
        self.calls += 1
        if request["backend"] != "synthetic_multifront":
            raise DesignerError("offline curriculum control requires an exact synthetic interface oracle")
        chosen = min(request["training_summary"], key=lambda ident:
                     (-request["training_summary"][ident]["mean_interface_regret"], ident))
        return validate_response({
            "schema": SCHEMA, "request_id": request["request_id"],
            "variant_id": chosen,
            "rationale": "Selected the largest exact interface-relative regret on training probes.",
        }, request, max_bytes=self.budget.max_response_bytes)

    def ledger(self):
        return {"profile": self.profile, "model": self.model, "calls": self.calls,
                "max_calls": self.budget.max_calls, "network_calls": 0,
                "billed_cost_usd": 0.0}

    def configuration(self):
        return {"profile": self.profile, "model": self.model, "reasoning_effort": self.reasoning,
                "budget": asdict(self.budget)}


def _http_exchange(endpoint: str, body: bytes, headers: dict, budget: DesignBudget) -> bytes:
    # A separate process permits the parent to terminate a slow DNS lookup,
    # trickled headers or body. Socket inactivity timeouts alone do not do this.
    payload = canonical({"endpoint": endpoint, "body_base64": base64.b64encode(body).decode("ascii"),
                         "headers": headers, "timeout_seconds": budget.timeout_seconds,
                         "max_response_bytes": budget.max_response_bytes})
    try:
        result = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("_http_worker.py"))],
                                input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                shell=False, timeout=budget.timeout_seconds, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise DesignerError(f"designer transport failed ({type(exc).__name__})") from None
    if result.returncode != 0:
        raise DesignerError("designer transport rejected the HTTP response")
    return result.stdout


class WebhookDesigner:
    """An explicit configured broker, called at most once per curriculum block.

    The broker must honor max_output_tokens and its own account spending cap.
    This client enforces calls/bytes/time, and does not infer provider token prices.
    """
    profile = "webhook"

    def __init__(self, endpoint: str, *, model: str, token: str | None = None,
                 reasoning: str = "high", budget: DesignBudget | None = None,
                 exchange: Callable | None = None):
        if not isinstance(endpoint, str):
            raise DesignerError("configure a webhook URL")
        parts = urlsplit(endpoint)
        loopback = parts.hostname in ("127.0.0.1", "localhost", "::1")
        if (parts.scheme != "https" and not (parts.scheme == "http" and loopback)):
            raise DesignerError("webhook needs HTTPS or an explicit loopback HTTP endpoint")
        if (not parts.hostname or parts.username or parts.password or parts.fragment
                or parts.query):
            raise DesignerError("webhook URL cannot carry credentials, query or fragment")
        if not isinstance(model, str) or not _ID.fullmatch(model):
            raise DesignerError("set a real provider route/model ID")
        if reasoning not in ("low", "medium", "high"):
            raise DesignerError("invalid reasoning effort")
        if token is not None and (not isinstance(token, str) or "\n" in token or "\r" in token):
            raise DesignerError("invalid webhook token")
        self.endpoint, self.model, self.token = endpoint, model, token
        self.reasoning = reasoning
        self.budget = budget or DesignBudget()
        self.calls, self.request_bytes, self.response_bytes = 0, 0, 0
        self._exchange = exchange or _http_exchange
        self._seen: set[str] = set()

    @classmethod
    def from_env(cls, *, budget: DesignBudget | None = None):
        endpoint, model = os.getenv("PLUTO_GYM_DESIGN_URL"), os.getenv("PLUTO_GYM_DESIGN_MODEL")
        if not endpoint or not model:
            raise DesignerError("set PLUTO_GYM_DESIGN_URL and PLUTO_GYM_DESIGN_MODEL")
        return cls(endpoint, model=model, token=os.getenv("PLUTO_GYM_DESIGN_TOKEN"),
                   reasoning=os.getenv("PLUTO_GYM_DESIGN_REASONING", "high"), budget=budget)

    def propose(self, request: dict) -> dict:
        if self.calls >= self.budget.max_calls:
            raise DesignerError("designer call budget exhausted")
        if request.get("model") != self.model or request.get("reasoning_effort") != self.reasoning:
            raise DesignerError("request route differs from configured designer")
        _integer(request.get("max_output_tokens"), "max_output_tokens", 64,
                 self.budget.max_output_tokens)
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or request_id in self._seen:
            raise DesignerError("request ID missing or already consumed")
        payload = {k: v for k, v in request.items() if k != "request_id"}
        if digest(payload) != request_id:
            raise DesignerError("request content/hash mismatch")
        body = canonical(request)
        if len(body) > self.budget.max_input_bytes:
            raise DesignerError("design request exceeds input byte budget")
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "Idempotency-Key": request_id}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        # Failed calls consume budget too; retries need a new explicit run.
        self.calls += 1
        self._seen.add(request_id)
        self.request_bytes += len(body)
        raw = self._exchange(self.endpoint, body, headers, self.budget)
        if not isinstance(raw, bytes):
            raise DesignerError("transport must return bytes")
        self.response_bytes += len(raw)
        return validate_response(raw, request, max_bytes=self.budget.max_response_bytes)

    def ledger(self):
        return {"profile": self.profile, "model": self.model,
                "reasoning_effort": self.reasoning, "calls": self.calls,
                "max_calls": self.budget.max_calls, "request_bytes": self.request_bytes,
                "response_bytes": self.response_bytes,
                "max_output_tokens_per_call": self.budget.max_output_tokens,
                "endpoint_sha256": digest(self.endpoint), "billed_cost_usd": None,
                "cost_scope": "Provider billing and account cap belong to the configured broker."}

    def configuration(self):
        return {"profile": self.profile, "model": self.model, "reasoning_effort": self.reasoning,
                "budget": asdict(self.budget), "endpoint_sha256": digest(self.endpoint),
                "transport": "HTTP worker with parent subprocess timeout"}

