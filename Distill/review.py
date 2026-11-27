"""Independent semantic-review rubric for visible worked answers.

A format pass or exact answer pass does not verify the derivation. This module
creates a review form and validates completed human review records. It never
assigns scores itself or authorizes training.
"""
from __future__ import annotations
import argparse
import json
from .core import digest, read_json, require, validate_response, write_json
from .inky_free import Journal

RUBRIC={
    "mathematical_validity": {
        "0":"Material mathematical error, unsupported inference or contradiction.",
        "1":"Main idea plausible; a material step is missing or needs repair.",
        "2":"Claim and derivation valid under the stated premises."},
    "checkability": {
        "0":"No reproducible computation, witness, proof step or test.",
        "1":"Partly checkable; a relevant intermediate quantity or procedure is missing.",
        "2":"Independent reviewer can reproduce the necessary intermediate work."},
    "assumptions_and_scope": {
        "0":"Missing or false premises, or an unsupported extension of the claim.",
        "1":"Most premises stated; scope or limitations incomplete.",
        "2":"Necessary premises, types, bounds and limits are explicit."},
    "explanatory_value": {
        "0":"Repeats the question or gives irrelevant content.",
        "1":"Useful explanation with a material gap in how or why.",
        "2":"Clear, technically grounded explanation of mechanism and purpose."},
}

def template(packet,response):
    answers=validate_response(packet,response)
    return {"schema_version":"distill.semantic_review.v1","packet_id":packet["packet_id"],
            "response_sha256":digest(response),"reviewer":None,"rubric":RUBRIC,
            "rows":[{"question_id":q["id"],"scores":{k:None for k in RUBRIC},"evidence":None}
                    for q in packet["questions"] if answers[q["id"]]["status"]=="answered"],
            "custody":"reference_only","training_admission":False}

def validate_review(packet,response,record):
    expected=template(packet,response)
    require(isinstance(record,dict) and set(record)==set(expected),"invalid review fields")
    for k in ("schema_version","packet_id","response_sha256","rubric","custody","training_admission"):
        require(record[k]==expected[k] and (k!="training_admission" or record[k] is False),"review binding mismatch")
    require(isinstance(record["reviewer"],str) and 0<len(record["reviewer"].strip())<=256,"reviewer required")
    require(isinstance(record["rows"],list) and len(record["rows"])==len(expected["rows"]),"review row count mismatch")
    totals={k:0 for k in RUBRIC}
    for row,ref in zip(record["rows"],expected["rows"]):
        require(isinstance(row,dict) and set(row)==set(ref) and row["question_id"]==ref["question_id"],"review row binding mismatch")
        require(isinstance(row["scores"],dict) and set(row["scores"])==set(RUBRIC),"review score fields mismatch")
        for key,value in row["scores"].items():
            require(type(value) is int and 0<=value<=2,"invalid rubric score")
            totals[key]+=value
        require(isinstance(row["evidence"],str) and 0<len(row["evidence"].strip())<=12000,"review evidence required")
    return {"packet_id":record["packet_id"],"response_sha256":record["response_sha256"],
            "review_sha256":digest(record),"reviewer_claimed":record["reviewer"],"reviewed_questions":len(record["rows"]),
            "dimension_totals":totals,"max_per_dimension":2*len(record["rows"]),
            "scope":"reviewer-supplied judgments; not automatically verified",
            "training_admission":False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("template","validate"))
    p.add_argument("--campaign",required=True)
    p.add_argument("--packet-id",required=True)
    p.add_argument("--review")
    p.add_argument("--out",required=True)
    args=p.parse_args()
    j=Journal(args.campaign)
    packet,raw,receipt=j.result(args.packet_id)
    require(j.mode=="live","fixture answers are not review evidence")
    require(receipt["format"]=="valid","response format not valid")
    response=json.loads(raw["text"])
    if args.command=="template":
        result=template(packet,response)
    else:
        require(bool(args.review),"provide completed review")
        result=validate_review(packet,response,read_json(args.review))
    write_json(args.out,result)
    print(json.dumps({"packet_id":packet["packet_id"],"written":args.out,"training_admission":False}))

if __name__=="__main__":
    main()
