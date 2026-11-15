"""NOR-16S: retrospective risk/coverage audit of frozen public MAP evidence.

The threshold is a certificate predicate on the complete public
candidate-output equivalence partition. A source is certified when
ALL publicly accepted candidate programs predict the SAME output on the
public query input, so any planted true program consistent with the demos
must yield that output. This guarantee is conditional on exact generator
membership in the finite candidate family.

Read only the frozen original NOR-16 full JSON. No training, new source
evaluation, hindsight ranking or model rewrites occur. Other thresholds
are descriptive posthoc sensitivity points, not preregistered outcomes.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/"results"/"nor16_full.zip"
SHA="b1ca49c9fb0d95510f020d4318d29a10abc25759fa110370eef967d538a20cef"


def risk_curve(source=ARCHIVE):
    with ZipFile(source) as archive:
        assert archive.namelist()==["nor16_full.json"]
        assert archive.testzip() is None
        raw=archive.read("nor16_full.json")
    assert len(raw)==1331006 and sha256(raw).hexdigest()==SHA
    result=json.loads(raw)
    assert result["schema"]=="nor16.public_query_predictive_consensus_source_locked.v1"
    primary=[x for x in result["records"] if x["family"]=="late_source"]
    assert len(primary)==192
    rows=[]
    for row in primary:
        mask=row["public_accept_mask"]
        pred=row["candidate_public_prediction_sha256"]
        accepted=[pred[j] for j in range(16) if mask&(1<<j)]
        counts=Counter(accepted)
        probability=Fraction(max(counts.values()),len(accepted))
        assert probability==Fraction(
            row["uniform_accepted_candidate_MAP_future_probability_num"],
            row["uniform_accepted_candidate_MAP_future_probability_den"])
        grade=row["arms"]["full_mask_map_16"]["grade"]
        assert grade["accepted"]==1
        assert grade["future_correct"]+grade["false_accept"]==1
        rows.append({"p":probability,"correct":grade["future_correct"],
                     "false":grade["false_accept"],
                     "certainty":len(counts)==1,
                     "has_ambiguous_future":row[
                         "counterfactual_indistinguishable_pair"] is not None})
    assert sum(r["certainty"] for r in rows)==52
    assert sum(r["has_ambiguous_future"] for r in rows)==140
    assert all(r["certainty"]!=r["has_ambiguous_future"] for r in rows)
    out=[]
    for bound in (Fraction(1),Fraction(1,2),Fraction(1,4),Fraction(0)):
        selected=[z for z in rows if z["p"]>=bound]
        n=len(selected)
        successes=sum(z["correct"] for z in selected)
        failures=sum(z["false"] for z in selected)
        expected=sum((z["p"] for z in selected),Fraction(0))
        assert successes+failures==n
        out.append({
            "minimum_uniform_prior_conditional_success_probability":str(bound),
            "selected_sources":n,
            "abstained_sources":len(rows)-n,
            "future_correct":successes,
            "false_accept":failures,
            "expected_correct_under_uniform_accepted_program_prior":float(expected),
            "expected_false_accept_under_uniform_accepted_program_prior":float(n-expected),
            "observed_conditional_precision":successes/n if n else None,
            "extra_exact_public_checks_for_complete_input_collection":16*len(rows),
            "public_query_candidate_simulations_for_complete_input_collection":16*len(rows),
            "interpretation":"posthoc sensitivity from frozen source outcome, not a trained policy"
        })
    assert out[0]["selected_sources"]==52
    assert out[0]["future_correct"]==52
    assert out[0]["false_accept"]==0
    assert out[1]["selected_sources"]==63
    assert out[1]["future_correct"]==56
    assert out[1]["false_accept"]==7
    assert out[-1]["selected_sources"]==192
    assert out[-1]["future_correct"]==80
    assert out[-1]["false_accept"]==112
    return {
        "schema":"nor16s.conditional_public_output_identifiability_risk_coverage.v1",
        "full_nor16_source_sha256":SHA,
        "primary_source_cases":192,
        "public_only_uniquely_predicted_output_cases":52,
        "counterfactual_ambiguous_possible_future_cases":140,
        "exactly_certified_future_case_count_if_planted_program_member":52,
        "actual_false_accept_on_certified_set":0,
        "scope":{
            "generator_membership_required_for_certification":True,
            "full_public_demo_and_query_input_only_used_for_certificate":True,
            "future_truth_only_used_for_retrospective_grading":True,
            "abstaining_policies_not_preregistered_heldout_arms":True,
            "all_sixteen_public_checks_required_for_proof_of_uniqueness":True,
            "nontrivial_public_query_simulation_cost":True
        },
        "retrospective_threshold_table":out
    }


def main():
    z=risk_curve()
    print(json.dumps(z,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
