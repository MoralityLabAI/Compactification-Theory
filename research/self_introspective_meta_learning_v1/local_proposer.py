"""Optional, explicitly invoked localhost-only 27B proposal interface.

No model is loaded by importing this module. The untrusted model supplies a
proposal record; the controller determines authorized actions, training budgets,
and promotion procedure. Never execute proposer-generated Python or shell code.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import json
import math

OPERATORS={
    "successor_consistency": ("T1",),
    "precision_probe": ("T1", "T4"),
    "adapter_placement": ("T0", "T1", "T4"),
    "replay_allocation": ("T0", "T1", "T4"),
    "curriculum_focus": ("T0", "T1", "T4"),
}
REQUIRED={"schema","target_node","operator","predicted_gain",
          "predicted_cost_updates","diagnostic_refs","competing_explanation"}

class ProposerError(ValueError):
    pass

@dataclass(frozen=True)
class Proposal:
    schema:str
    target_node:str
    operator:str
    predicted_gain:float
    predicted_cost_updates:int
    diagnostic_refs:tuple[str,...]
    competing_explanation:str


def validate_proposal(obj:dict,available_budget:int)->Proposal:
    if not isinstance(obj,dict) or set(obj)!=REQUIRED:
        raise ProposerError("proposal schema keys mismatch")
    if obj.get("schema")!="siml01.proposal.v1":
        raise ProposerError("invalid schema version")
    op=obj.get("operator")
    target=obj.get("target_node")
    if op not in OPERATORS or target not in OPERATORS[op]:
        raise ProposerError("unregistered target-operator pair")
    gain=obj.get("predicted_gain")
    cost=obj.get("predicted_cost_updates")
    if isinstance(gain,bool) or not isinstance(gain,(float,int)) or not math.isfinite(gain) or not -1<=gain<=1:
        raise ProposerError("gain outside bounded registered scale")
    if type(cost)!=int or not 1<=cost<=available_budget:
        raise ProposerError("operator cost exceeds allowed training updates")
    refs=obj.get("diagnostic_refs")
    if (not isinstance(refs,list) or not refs or len(refs)>8 or
        any(not isinstance(s,str) or not 1<=len(s)<=128 for s in refs)):
        raise ProposerError("invalid diagnostic identifiers")
    rival=obj.get("competing_explanation")
    if not isinstance(rival,str) or not 8<=len(rival)<=400:
        raise ProposerError("missing bounded competing explanation")
    return Proposal(obj["schema"],target,op,float(gain),cost,tuple(refs),rival)


def parse_model_reply(text:str,available_budget:int)->Proposal:
    if len(text)>4096:
        raise ProposerError("model output too large")
    try:
        obj=json.loads(text)
    except (ValueError,TypeError) as exc:
        raise ProposerError("invalid JSON") from exc
    return validate_proposal(obj,available_budget)


def validate_endpoint(endpoint:str)->str:
    if not isinstance(endpoint,str) or len(endpoint)>250:
        raise ProposerError("endpoint must be short text")
    u=urlsplit(endpoint)
    if u.scheme!="http" or u.hostname not in ("127.0.0.1","localhost","::1"):
        raise ProposerError("only loopback HTTP model endpoints are supported")
    if u.username or u.password or u.query or u.fragment or not u.port:
        raise ProposerError("endpoint must have a port and no credentials or query")
    if u.path not in ("/v1/chat/completions",):
        raise ProposerError("unsupported endpoint path")
    return endpoint


def local_chat_proposal(endpoint:str,model_id:str,diagnostics:dict,
                        available_budget:int,*,timeout_seconds:float=30.0)->Proposal:
    """One opt-in OpenAI-compatible HTTP request. Does not train or promote."""
    validate_endpoint(endpoint)
    if not isinstance(model_id,str) or not model_id or len(model_id)>200:
        raise ProposerError("model ID required")
    if not isinstance(diagnostics,dict):
        raise ProposerError("diagnostics must be structured")
    prompt={"graph":"SIML-01", "measurements":diagnostics,
            "allowed_operators":OPERATORS,"remaining_updates":available_budget,
            "required_schema":sorted(REQUIRED)}
    payload={"model":model_id,"temperature":0,"max_tokens":450,
             "messages":[{"role":"system","content":"Return only one JSON object conforming to siml01.proposal.v1. Base claims on supplied diagnostics. Never emit executable instructions."},
                         {"role":"user","content":json.dumps(prompt,sort_keys=True)}]}
    body=json.dumps(payload).encode()
    if len(body)>20000:
        raise ProposerError("diagnostics exceed request budget")
    request=Request(endpoint,data=body,method="POST",headers={"Content-Type":"application/json"})
    # Explicit opt-in only. Not called from library import or from tests.
    with urlopen(request,timeout=timeout_seconds) as response:
        if response.status!=200:
            raise ProposerError("local inference rejected request")
        data=response.read(12000)
    try:
        content=json.loads(data)["choices"][0]["message"]["content"]
    except (KeyError,IndexError,TypeError,ValueError) as exc:
        raise ProposerError("malformed local completion response") from exc
    return parse_model_reply(content,available_budget)
