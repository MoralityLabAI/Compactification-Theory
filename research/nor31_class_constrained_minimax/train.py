"""NOR-31 historical TRAIN-only optimizer with policy-class collapse.

Reads only the two ALREADY merged original NOR-26 and NOR-30 archives.
Never creates NOR-31 sources, witness values, or prospective attack data.
All constraints/provenance are registered in REGISTRATION.md.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations_with_replacement
import json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
ARMS=("uniform","static_prior","posterior",
      "hybrid_25","hybrid_50","hybrid_75")
ATTACKERS=("uniform","static_prior","posterior","shuffled_prior",
           "hybrid_25","hybrid_50","hybrid_75")
FAMILIES=("tail","magnitude","lag","uniform")
FAMILY_WEIGHTS=(4,3,2,1)
NQUANTA=16
HIST26_SOURCE="e3f5a3df07d062d8681a39a7ddcba9a981b58af6f149f13bd372daf08ac6606f"
HIST26_FULL="a451897b780b024aac49f543833bc5e3ec32a734489d3f7a9b14a929bd11669a"
HIST30_SOURCE="9961afd81cac78b32fc5280fe54291f7db4dca0cf85cb6624560ddac420e1878"
HIST30_FULL="4189235f7cb5c14f3c7d1028abda264d893c7b5e5d2eda0c4c1de091f052b9c8"
SCHEMA="nor31.historical_two_cohort_alias_collapsed_training.v1"
TEMPLATE_DENOM=12*4096*10


def canonical(x):
    return (json.dumps(x,indent=2,sort_keys=True,ensure_ascii=True)+"\n").encode()


def bytes_from_archive(study,zip_name,file_name,expected_sha,expected_length):
    path=ROOT/"research"/study/"results"/zip_name
    with ZipFile(path) as z:
        assert z.testzip() is None
        raw=z.read(file_name)
    assert len(raw)==expected_length,(path,file_name,len(raw))
    assert sha256(raw).hexdigest()==expected_sha,file_name
    return raw


def historical_data():
    old26=json.loads(bytes_from_archive(
        "nor26_policy_aware_audits","nor26_original_hosted.zip",
        "nor26_full.json",HIST26_FULL,217954))
    src26=json.loads(bytes_from_archive(
        "nor26_policy_aware_audits","nor26_original_hosted.zip",
        "nor26_fixture.json",HIST26_SOURCE,2838554))
    old30=json.loads(bytes_from_archive(
        "nor30_minimax_audits","nor30_original_hosted.zip",
        "nor30_full.json",HIST30_FULL,337563))
    src30=json.loads(bytes_from_archive(
        "nor30_minimax_audits","nor30_original_hosted.zip",
        "nor30_fixture.json",HIST30_SOURCE,2284577))
    assert old26["schema"]=="nor26.policy_aware_blind_attack_crossplay.v1"
    assert old30["schema"]=="nor30.heldout_constrained_minimax_blind_audit.v1"
    assert len(old26["source_records"])==len(src26["cases"])==12
    assert len(old30["source_records"])==len(src30["cases"])==12
    assert old26["registered_primary_pass"] and old30["registered_primary_pass"]
    hashes26=sorted(x["full_public_input_sha256"] for x in src26["cases"])
    hashes30=sorted(x["full_public_input_sha256"] for x in src30["cases"])
    assert len(set(hashes26))==len(set(hashes30))==12
    assert set(hashes26).isdisjoint(hashes30)
    counts={(atk,aud):0 for atk in ATTACKERS for aud in ARMS}
    for rec in old26["source_records"]:
        assert len(rec["crossplay"])==49
        for cell in rec["crossplay"]:
            assert cell["total"]==4096
            atk,aud=cell["attacker_policy"],cell["auditor_policy"]
            if aud in ARMS:
                counts[(atk,aud)]+=cell["detected"]
    crossplay=[[counts[(atk,aud)] for aud in ARMS] for atk in ATTACKERS]
    assert all(0<=c<=49152 for row in crossplay for c in row)
    template_hits={aud:0 for aud in ARMS}
    for rec in old30["source_records"]:
        measured={(cell["attacker"],cell["auditor"]):cell
                  for cell in rec["ordinary_arm_hit_counts"]}
        assert len(measured)==14*7
        for aud in ARMS:
            for family,w in zip(FAMILIES,FAMILY_WEIGHTS):
                cell=measured[("template_"+family,aud)]
                assert cell["total"]==4096
                template_hits[aud]+=w*cell["detected"]
    template_numerators=[template_hits[aud] for aud in ARMS]
    assert all(0<=v<=TEMPLATE_DENOM for v in template_numerators)
    return crossplay,template_numerators,hashes26,hashes30


def fit(matrix,template_numerators,constrain_template=True,constrain_uniform=True):
    """Exact six-class 16-quantum grid, fixed lexicographic tie rule."""
    assert len(matrix)==7 and all(len(row)==6 for row in matrix)
    assert len(template_numerators)==6
    best=None
    key_best=None
    enumerated=0
    eligible=0
    for picks in combinations_with_replacement(range(6),NQUANTA):
        enumerated+=1
        weights=tuple(picks.count(i) for i in range(6))
        if constrain_uniform and not 2<=weights[0]<=12:
            continue
        if constrain_template:
            mixed_template=sum(w*c for w,c in zip(weights,template_numerators))
            if mixed_template*100 < 18*NQUANTA*TEMPLATE_DENOM:
                continue
        eligible+=1
        scores=[sum(x*y for x,y in zip(row,weights)) for row in matrix]
        key=(min(scores),sum(scores),tuple(-w for w in weights))
        if best is None or key>key_best:
            best=weights
            key_best=key
    assert eligible>0
    return {"weights":list(best),
            "min_weighted_integer_detection":key_best[0],
            "sum_weighted_integer_detection":key_best[1],
            "candidate_grid_enumerated":enumerated,
            "candidate_grid_eligible":eligible,
            "historical_weighted_template_numerator":sum(
                w*c for w,c in zip(best,template_numerators)),
            "historical_weighted_template_denominator":NQUANTA*TEMPLATE_DENOM,
            "historical_in_template_rate":sum(
                w*c for w,c in zip(best,template_numerators))/
                    (NQUANTA*TEMPLATE_DENOM)}


def freeze_train():
    matrix,templates,hashes26,hashes30=historical_data()
    selected=fit(matrix,templates,True,True)
    baseline=fit(matrix,templates,False,True)
    unrestricted=fit(matrix,templates,False,False)
    matrix_digest=sha256(canonical({
        "attackers":ATTACKERS,"auditors":ARMS,"integer_hits":matrix,
        "denominator_per_cell":49152})).hexdigest()
    coverage_digest=sha256(canonical({
        "auditors":ARMS,"weighted_hits":templates,
        "denominator_per_auditor":TEMPLATE_DENOM})).hexdigest()
    assert selected["historical_in_template_rate"]>=0.18
    return {
        "schema":SCHEMA,
        "preregistration_commit":"7e37e8c57a68e7224a67626d2495e9bcf922a424",
        "historical_nor26_public_sha256":HIST26_SOURCE,
        "historical_nor26_outcome_sha256":HIST26_FULL,
        "historical_nor30_public_sha256":HIST30_SOURCE,
        "historical_nor30_outcome_sha256":HIST30_FULL,
        "historical_nor26_public_hashes":hashes26,
        "historical_nor30_public_hashes":hashes30,
        "training_public_source_count":24,
        "test_source_information_used":False,
        "auditors":list(ARMS),"attacker_classes":list(ATTACKERS),
        "grid_quanta":NQUANTA,
        "historical_nor26_matrix_integer_hits":matrix,
        "historical_matrix_sha256":matrix_digest,
        "historical_nor30_template_weighted_numerators":templates,
        "historical_template_denominator":TEMPLATE_DENOM,
        "historical_template_sha256":coverage_digest,
        "template_floor_numerator":18,"template_floor_denominator":100,
        "uniform_weight_min":2,"uniform_weight_max":12,
        "tie_rule":"max min row counts, max total row counts, lexicographically smallest six weights",
        "fitted_constrained":selected,
        "diagnostic_no_template_floor":baseline,
        "diagnostic_unrestricted":unrestricted
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    result=freeze_train()
    out=canonical(result)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(out)
    print("NOR31 FROZEN TRAIN WEIGHTS",result["fitted_constrained"])
    print("NOR31 TRAIN ORIGINAL HISTORICAL MATRIX SHA",result["historical_matrix_sha256"])
    print("NOR31 TRAIN ORIGINAL TEMPLATE SHA",result["historical_template_sha256"])
    print("NOR31 TRAIN FROZEN RECEIPT BYTES SHA",len(out),sha256(out).hexdigest())


if __name__=="__main__":
    main()
