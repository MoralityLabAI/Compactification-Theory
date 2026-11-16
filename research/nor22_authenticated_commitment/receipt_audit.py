"""NOR-22 independent exact modular commitment and private future auditor.

No import of the NOR22 experiment or policy implementation. Reconstruct
every original RRC25 public source and planted program from preregistered
source seeds. Recompute all true target worlds after viewing the frozen
public decisions, use separate power-mod polynomial evaluation to verify
the target-dependent commitments, and independently audit all 192
world-arm decisions.

Source case kinds are four paired counterfactual *worlds* per one unique
public source, not four independent input cases. The extra authenticated
target commitments are explicitly additional target-dependent data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21

P=65537
Q=65521
EVAL_POINT=3
OFFSET=94322000
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_sparse_small","wide_rational_short")
KINDS=("valid","unit_shift","p_shift","product_shift")
ARMS=("public_only","trusted_p","trusted_pq","forged_pq")
SHIFT={"valid":0,"unit_shift":1,"p_shift":P,"product_shift":P*Q}
FIRST_RECEIPT=HERE/"results"/"nor22_first_frozen_receipt.json"


def direct_evaluation(coeffs,modulus):
    # Independent from experiment's reversed-list Horner implementation.
    return sum((a%modulus)*pow(EVAL_POINT,i,modulus)
               for i,a in enumerate(coeffs))%modulus


def digest_target(coeffs):
    return sha256(json.dumps(coeffs,separators=(",",":")).encode()).hexdigest()


def public_digest(case,degree):
    return nor21.public_digest(case,degree)


def candidate_program(header,j):
    return (header.base,header.delay,header.spacing,header.amplitude,
            corpus.OPS[j//4],corpus.OPS[j%4])


def replay_source(row):
    rep=row["replicate"]
    degree=row["observation_degree"]
    family=row["family"]
    assert rep in (0,1) and degree in DEGREES and family in FAMILIES
    expected_seed=(corpus.SEED+OFFSET+1000003*rep+
                   100009*DEGREES.index(degree)+10007*FAMILIES.index(family))
    assert row["source_seed"]==expected_seed
    rng=random.Random(expected_seed)
    observed_degree=degree
    query_degree=3*degree//2
    program=demos=query=None
    assert 0<=row["selected_attempt"]<100
    for _ in range(row["selected_attempt"]+1):
        program=nor21.new_header_program(rng)
        demos=tuple((f:=nor21.public_source(rng,family,query_degree),
                     corpus.execute(f,program)) for _ in range(2))
        query=nor21.public_source(rng,family,query_degree)
    assert program is not None
    head=row["public_header"]
    header=corpus.Header(head["base"],head["delay"],head["spacing"],
                         head["amplitude"])
    assert program[:4]==(header.base,header.delay,header.spacing,
                         header.amplitude)
    assert [[list(f[:observed_degree+1]),list(y[:observed_degree+1])]
            for f,y in demos]==row["demos"]
    assert list(query)==row["public_query_input"]
    case=corpus.Case("nor22-independent-original",family,demos,query,(),None)
    assert public_digest(case,observed_degree)==row["full_public_input_sha256"]
    assert len(corpus.possible_headers(case))==1
    public_case=corpus.Case("nor22-independent-public",family,
       tuple((tuple(f),tuple(y)) for f,y in row["demos"]),
       tuple(row["public_query_input"]),(),None)
    assert public_case.future==() and public_case.program is None
    mask=nor21.native_mask(public_case,header,observed_degree)
    assert mask==row["original_public_accept_mask"]
    assert mask>0
    return case,header,program,mask


def actual_outcome(emitted,predicted,target):
    if emitted and predicted==target:
        return "correct"
    if emitted:
        return "false_accept"
    return "abstain"


def audit(fixture_bytes,result_bytes):
    fixture=json.loads(fixture_bytes)
    result=json.loads(result_bytes)
    assert fixture["schema"]==(
        "nor22.public_with_explicit_authenticated_target_commitments.v1")
    assert result["schema"]=="nor22.authenticated_modular_target_commitment.v1"
    assert fixture["registration_offset"]==result["registered_moduli"][0]-65537+OFFSET
    assert fixture["registered_moduli"]==[P,Q]
    assert result["registered_moduli"]==[P,Q]
    assert fixture["registered_polynomial_evaluation_point"]==EVAL_POINT
    assert result["registered_evaluation_point"]==EVAL_POINT
    assert fixture["public_sources_include_extra_target_dependent_attested_values"]
    assert fixture["target_output_arrays_and_generating_programs_absent"]
    assert fixture["trusted_publisher_authenticity_is_an_explicit_assumption"]
    assert result["fixture_sha256"]==sha256(fixture_bytes).hexdigest()
    assert result["distinct_public_source_cases"]==12
    assert result["counterfactual_target_world_records"]==48
    assert result["all_source_decisions_frozen_before_private_program_and_target_replay"]
    assert result["all_target_dependent_commitments_explicitly_trusted_or_forged"]
    assert result["all_registered_outcomes_match"]
    assert fixture["registered_target_world_kinds"]==list(KINDS)
    assert result["world_kinds"]==list(KINDS)
    assert result["arms"]==list(ARMS)
    assert len(fixture["cases"])==len(result["source_records"])==12
    assert len({x["full_public_input_sha256"] for x in fixture["cases"]})==12
    assert Counter(x["observation_degree"] for x in fixture["cases"])=={
        512:6,1024:6}
    counts={
        kind:{arm:Counter({"correct":0,"false_accept":0,"abstain":0})
              for arm in ARMS} for kind in KINDS}
    for source,record in zip(fixture["cases"],result["source_records"],strict=True):
        case,header,program,mask=replay_source(source)
        assert record["full_public_input_sha256"]==source[
            "full_public_input_sha256"]
        assert record["original_public_accept_mask"]==mask
        assert record["replicate"]==source["replicate"]
        assert record["observation_degree"]==source["observation_degree"]
        assert record["family"]==source["family"]
        assert record["query_degree"]==source["query_degree"]
        candidate_j=record["chosen_class"]
        assert 0<=candidate_j<16 and mask&(1<<candidate_j)
        predicted=corpus.execute(case.query,candidate_program(header,candidate_j))
        valid_target=corpus.execute(case.query,program)
        assert predicted==valid_target
        assert digest_target(predicted)==record["predicted_output_sha256"]
        expected_self={
            "eval_mod_65537":direct_evaluation(predicted,P),
            "eval_mod_65521":direct_evaluation(predicted,Q)}
        assert record["predicted_modular_commitments"]==expected_self
        assert set(source["trusted_target_commitments_by_counterfactual_world"])==set(KINDS)
        assert set(record["worlds"])==set(KINDS)
        assert len({z["same_complete_public_input_sha256"]
                    for z in record["worlds"].values()})==1
        for kind in KINDS:
            target=list(valid_target)
            target[-1]+=SHIFT[kind]
            target=tuple(target)
            expected_auth={
                "trusted_eval_mod_65537":direct_evaluation(target,P),
                "trusted_eval_mod_65521":direct_evaluation(target,Q),
            }
            assert source[
                "trusted_target_commitments_by_counterfactual_world"][kind]==expected_auth
            actual=record["worlds"][kind]
            assert actual["authentic_commitments"]==expected_auth
            assert actual["target_sha256_sealed_grader_only"]==digest_target(target)
            assert actual["same_complete_public_input_sha256"]==source[
                "full_public_input_sha256"]
            assert actual["counterfactual_target_differs_from_valid"]==(kind!="valid")
            assert set(actual["arms"])==set(ARMS)
            for arm in ARMS:
                observed=actual["arms"][arm]
                emitted=observed["chosen_class"] is not None
                expected_match=(
                    True if arm in ("public_only","forged_pq")
                    else expected_self["eval_mod_65537"]==expected_auth[
                        "trusted_eval_mod_65537"] if arm=="trusted_p"
                    else expected_self["eval_mod_65537"]==expected_auth[
                        "trusted_eval_mod_65537"] and expected_self[
                        "eval_mod_65521"]==expected_auth["trusted_eval_mod_65521"]
                )
                assert emitted==expected_match,(kind,arm)
                assert observed["arm"]==arm
                assert observed["accepted"]==emitted
                assert observed["chosen_class"]==(candidate_j if emitted else None)
                assert observed["status"]==("candidate_emitted" if emitted else
                                           "modular_mismatch_abstain")
                assert observed["extra_trusted_target_dependent_modular_values"]==(
                    0 if arm in ("public_only","forged_pq") else
                    1 if arm=="trusted_p" else 2)
                if arm in ("public_only","forged_pq"):
                    assert observed["commitment_provenance"] in {
                        "none","untrusted_self_generated"}
                else:
                    assert observed["commitment_provenance"]==(
                        "assumed_authenticated_target_publisher")
                label=actual_outcome(emitted,predicted,target)
                assert observed["future_correct"]==(label=="correct")
                assert observed["false_accept"]==(label=="false_accept")
                assert observed["abstained"]==(label=="abstain")
                assert label==result["registered_exact_matrix"][kind][arm]
                counts[kind][arm][label]+=1
        assert source["trusted_target_commitments_by_counterfactual_world"][
            "valid"]!=source["trusted_target_commitments_by_counterfactual_world"][
            "unit_shift"]
        assert source["trusted_target_commitments_by_counterfactual_world"][
            "valid"]["trusted_eval_mod_65537"]==source[
            "trusted_target_commitments_by_counterfactual_world"][
            "p_shift"]["trusted_eval_mod_65537"]
        assert source["trusted_target_commitments_by_counterfactual_world"][
            "valid"]==source[
            "trusted_target_commitments_by_counterfactual_world"][
            "product_shift"]
    reconstructed={
        kind:{arm:dict(counts[kind][arm]) for arm in ARMS}
        for kind in KINDS}
    assert reconstructed==result["observed_matrix"]
    for kind in KINDS:
        for arm in ARMS:
            assert reconstructed[kind][arm][
                result["registered_exact_matrix"][kind][arm]]==12
    for flag in (
       "target_commitments_are_extra_target_dependent_information",
       "trusted_target_publisher_is_required_for_authentic_modular_checks",
       "rejected_fingerprint_mismatch_proves_wrong_prediction",
       "matching_modular_fingerprints_not_sufficient_for_correctness",
       "forged_commitment_is_untrusted_negative_control",
       "target_program_identity_never_passed_to_public_selection",
       "target_truth_replayed_after_frozen_decisions",
       "same_source_multiple_worlds_not_independent_cases",
       "no_claim_of_collision_resistant_authentication",
    ):
        assert result["scope"][flag],flag
    return {
        "independently_regenerated_public_source_programs":12,
        "same_public_source_world_sets":12,
        "counterfactual_worlds":48,
        "policy_outcomes_independently_audited":192,
        "authentic_modular_commitments_reconstructed":96,
        "all_exact_registered_outcomes_match":True,
        "additional_trusted_target_information_required":True,
        "product_modulus_collision_remains":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--full",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    if args.archive:
        assert args.fixture is None and args.full is None
        with ZipFile(args.archive) as z:
            assert sorted(z.namelist())==["nor22_fixture.json","nor22_full.json"]
            assert z.testzip() is None
            fixture=z.read("nor22_fixture.json")
            full=z.read("nor22_full.json")
    elif args.fixture and args.full:
        fixture=args.fixture.read_bytes()
        full=args.full.read_bytes()
    elif args.fixture or args.full:
        p.error("both --fixture and --full required")
    else:
        print("NOR22 INDEPENDENT AUDITOR AWAITS EXACT ORIGINAL SOURCE FILES")
        return
    if FIRST_RECEIPT.exists():
        receipt=json.loads(FIRST_RECEIPT.read_text())
        assert len(fixture)==receipt["public_fixture_bytes"]
        assert len(full)==receipt["full_result_bytes"]
        assert sha256(fixture).hexdigest()==receipt["public_fixture_sha256"]
        assert sha256(full).hexdigest()==receipt["full_result_sha256"]
    result=audit(fixture,full)
    print("NOR22 INDEPENDENT AUTHENTIC COMMITMENT AND 48-WORLD AUDIT PASSED",result)
    if args.check:
        assert result["policy_outcomes_independently_audited"]==192
        assert result["all_exact_registered_outcomes_match"]


if __name__=="__main__":
    main()
