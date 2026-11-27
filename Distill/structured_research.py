"""Provider-neutral structured research wire adapter, no API calls.

Use the generated closed object schema with Claude output_config.format.
Raw wire bytes remain immutable; deterministic conversion is a distinct
experimental condition, never a repair of the official v2 archive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .core import (MAX_BYTES, RESEARCH_RESPONSE, canonical, digest, object_keys,
                   read_json, require, string, validate_response, write_json,
                   _packet, _unique, _reject_constant)
from .research_cases import DOCKET

SCHEMA="distill.structured_wire.v1"
STRING={"type":"string"}
INTEGER={"type":"integer"}
BOOL={"type":"boolean"}

def obj(props):
    return {"type":"object","properties":props,"required":list(props),"additionalProperties":False}

def arr(items):
    return {"type":"array","items":items}

def enum(*xs):
    return {"type":"string","enum":list(xs)}

def union(*xs):
    return {"anyOf":list(xs)}

def exact_answer_schema(family):
    if family=="adaptive_evidence":
        # A flat list avoids unsupported recursive grammar definitions.
        # The local adapter checks the complete rooted binary tree.
        leaf=obj({"path":STRING,"decision":enum("A","B","defer")})
        branch=obj({"path":STRING,"test":STRING})
        return obj({"nodes":arr(union(leaf,branch)),"value":STRING})
    if family=="robust_discrimination":
        return union(obj({"status":enum("plan"),"experiments":arr(STRING),"cost":INTEGER}),
                     obj({"status":enum("infeasible")}))
    if family=="research_portfolio":
        return obj({"jobs":arr(STRING),"worst_gain":INTEGER})
    if family=="paired_causal_gate":
        row=obj({"id":STRING,"mean_delta":STRING,"positive_roots":INTEGER,"matched":BOOL})
        return obj({"arms":arr(row),"eligible":arr(STRING)})
    if family=="proof_obligations":
        return obj({"current":arr(STRING),"jobs":arr(STRING),"conditional":arr(STRING),
                    "conditional_value":INTEGER})
    require(family==DOCKET,"unsupported research family")
    return obj({"ordinary_loss_millionths":INTEGER,"ood_loss_millionths":INTEGER,
                "learned_promotion_supported":BOOL})

def wire_schema(packet):
    _packet(packet)
    require("research" in packet,"structured research wire only")
    questions={}
    for q in packet["questions"]:
        props={key:exact_answer_schema(packet["research"]["family"]) if key=="answer" else STRING
               for key in q["fields"]}
        questions[q["id"]]=union(
            obj({"status":enum("answered"),"fields":obj(props)}),
            obj({"status":enum("abstain"),"fields":obj({"reason":STRING})}))
    return obj({"schema_version":enum(SCHEMA),"packet_id":STRING,"questions":obj(questions)})

def render(packet):
    _packet(packet)
    require("research" in packet,"research packet required")
    instruction=(
        "Complete this research examination using only the public packet. Provide concise, "
        "checkable calculations, proofs or witnesses and grounded scientific judgments. "
        "Keep proposed measurements separate from observed evidence. Return the JSON "
        "object required by the supplied schema. Copy packet_id exactly. The questions "
        "object is keyed by question ID; for each question use exactly the requested "
        "fields, or abstain with a nonempty reason. Answer every question once. "
        "All prose fields are nonempty strings; exact answers use the native JSON "
        "shape in the schema. For adaptive_evidence only, replace the requested "
        "recursive policy tree with a flat nodes array: path is the binary observation "
        "history (empty string for the root); each node has either a decision A/B/defer "
        "or a test ID. Test nodes have both child paths path+'0' and path+'1'; "
        "terminal nodes have no descendants. List every reachable node, exactly once. "
        "The value field is still a reduced fraction string. This is reference "
        "evaluation, no solver or referee access and no invented experiment outcomes."
    )
    return instruction+"\n\n"+canonical(packet).decode("ascii")

def check_schema(schema,x):
    """Local strict validation of the small schema vocabulary generated above."""
    if "anyOf" in schema:
        matches=0
        for alt in schema["anyOf"]:
            try:
                check_schema(alt,x)
            except ValueError:
                continue
            matches+=1
        require(matches==1,"wire union mismatch")
        return
    t=schema["type"]
    if t=="object":
        object_keys(x,set(schema["properties"]),"structured object")
        for key,v in schema["properties"].items():
            check_schema(v,x[key])
    elif t=="array":
        require(isinstance(x,list),"expected array")
        for v in x:
            check_schema(schema["items"],v)
    elif t=="string":
        require(isinstance(x,str),"expected string")
        if "enum" in schema:
            require(x in schema["enum"],"wrong enum")
    elif t=="integer":
        require(type(x) is int,"expected integer")
    elif t=="boolean":
        require(type(x) is bool,"expected boolean")
    else:
        raise ValueError("unknown local schema type")

def unflatten(nodes,max_depth):
    require(isinstance(nodes,list) and 1<=len(nodes)<=2**(max_depth+1)-1,"invalid node count")
    by_path={}
    for node in nodes:
        path=node["path"]
        require(isinstance(path,str) and len(path)<=max_depth and set(path)<=set("01")
                and path not in by_path,"invalid or duplicate policy path")
        by_path[path]=node
    visited=set()
    def tree(path):
        require(path in by_path,"missing policy child")
        node=by_path[path]
        visited.add(path)
        if "decision" in node:
            return {"decision":node["decision"]}
        require(len(path)<max_depth,"test beyond public depth bound")
        return {"test":node["test"],"zero":tree(path+"0"),"one":tree(path+"1")}
    result=tree("")
    require(visited==set(by_path),"unreachable policy node")
    return result

def convert(packet,wire,model):
    check_schema(wire_schema(packet),wire)
    require(wire["packet_id"]==packet["packet_id"],"wire packet ID mismatch")
    object_keys(model,{"provider","revision","run_id"},"host model")
    for k,v in model.items():
        string(v,k,256)
    answers=[]
    for q in packet["questions"]:
        a=wire["questions"][q["id"]]
        fields=dict(a["fields"])
        if q["id"]=="case.solve" and a["status"]=="answered" and packet["research"]["family"]=="adaptive_evidence":
            value=fields["answer"]
            fields["answer"]={"policy":unflatten(value["nodes"],len(packet["public_case"]["data"]["tests"])),
                              "value":value["value"]}
        answers.append({"question_id":q["id"],"status":a["status"],"fields":fields})
    result={"schema_version":RESEARCH_RESPONSE,"packet_id":packet["packet_id"],
            "model":model,"answers":answers}
    validate_response(packet,result)
    return result

def convert_bytes(packet,b,model):
    require(len(b)<MAX_BYTES//2,"oversized wire response")
    wire=json.loads(b.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_reject_constant)
    response=convert(packet,wire,model)
    receipt={"schema_version":"distill.structured_conversion.v1","packet_id":packet["packet_id"],
             "raw_sha256":hashlib.sha256(b).hexdigest(),"schema_sha256":digest(wire_schema(packet)),
             "response_sha256":digest(response),"model_claimed":model,
             "condition":"structured_wire_new_condition","training_admission":False}
    return response,receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare");a.add_argument("--packet",required=True);a.add_argument("--out",required=True)
    a=sub.add_parser("convert")
    for key in ("packet","raw","provider","revision","run-id","out"):
        a.add_argument("--"+key,required=True)
    a=parser.parse_args()
    packet=read_json(a.packet)
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False)
    if a.command=="prepare":
        schema=wire_schema(packet)
        write_json(root/"format.json",{"type":"json_schema","schema":schema})
        (root/"prompt.txt").write_text(render(packet),encoding="utf-8")
        receipt={"packet_id":packet["packet_id"],"schema_sha256":digest(schema),
                 "prompt_sha256":hashlib.sha256(render(packet).encode("utf-8")).hexdigest(),
                 "training_admission":False}
        write_json(root/"manifest.json",receipt)
    else:
        raw=Path(a.raw).read_bytes()
        response,receipt=convert_bytes(packet,raw,{"provider":a.provider,"revision":a.revision,"run_id":a.run_id})
        # Retain exact bytes, never normalize malformed raw wire text.
        (root/"raw.json").write_bytes(raw)
        write_json(root/"response.json",response)
        write_json(root/"receipt.json",receipt)
    print(json.dumps(receipt,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
