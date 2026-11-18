"""NOR-43 historical NOR42 paired-source signed-root smoke.

NO NEW NOR43 source identities. Reconstructs 64 already source-frozen and
independently graded NOR42 pairs from immutable first raw archive, with a
PUBLIC deterministic test-only Ed25519 authority. No claim of deployed target
truth, authenticated signing service, or model evaluation is made.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor43_authenticated_witness import signed_witness as producer
from research.nor43_authenticated_witness import verifier_reference as verifier

FIRST_ARCHIVE=ROOT/"research/nor42_paired_program_ood/prospective/results/nor42_original_first_hosted.zip"
FIRST_ARCHIVE_BYTES=2799867
FIRST_ARCHIVE_SHA="e60e97fd933f660c2274cb4a473ede7a98968d24eaf0dd3ae45ace1e1176cd0d"
FIRST_SOURCE_BYTES=9059929
FIRST_SOURCE_SHA="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
FIRST_GRADE_BYTES=201956
FIRST_GRADE_SHA="bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae"
CHALLENGE_STREAM=202611430041
PROVIDER_ID=producer.PROVIDER_ID
K=16


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def seed(*values):
    return int.from_bytes(sha256("|".join(map(str,values)).encode()).digest(),"big")


def original_first_archive():
    buf=FIRST_ARCHIVE.read_bytes()
    assert len(buf)==FIRST_ARCHIVE_BYTES
    assert sha256(buf).hexdigest()==FIRST_ARCHIVE_SHA
    with ZipFile(FIRST_ARCHIVE) as z:
        assert z.testzip() is None
        p=z.read("nor42_fixture.json")
        q=z.read("nor42_full.json")
    assert len(p)==FIRST_SOURCE_BYTES and sha256(p).hexdigest()==FIRST_SOURCE_SHA
    assert len(q)==FIRST_GRADE_BYTES and sha256(q).hexdigest()==FIRST_GRADE_SHA
    fixture=json.loads(p);grade=json.loads(q)
    assert fixture["schema"]=="nor42.paired_grammar_ood_public_only_reps100to115.v1"
    assert grade["schema"]=="nor42.paired_program_ood_authenticated_witness.v1"
    assert fixture["source_units"]==grade["fresh_public_source_pairs"]==64
    assert grade["two_world_targets"]==128
    assert grade["registered_primary_pass"] is True
    assert len(grade["source_records"])==64
    return fixture,grade


def attacker_private_key():
    """PUBLIC untrusted negative-control key, distinct from pinned provider."""
    return Ed25519PrivateKey.from_private_bytes(
        sha256(b"NOR43/REPRODUCIBLE-UNTRUSTED-MODEL-KEY/v1").digest())


def attacker_public_key(key):
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)


def historical_source_smoke():
    fixture,grade=original_first_archive()
    pinned=producer.test_reference_public_key()
    attacker=attacker_private_key()
    attacker_pub=attacker_public_key(attacker)
    assert pinned!=attacker_pub
    measurements=[]
    for index,(row,receipt) in enumerate(zip(
            fixture["cases"],grade["source_records"],strict=True)):
        src=row["full_public_input_sha256"]
        assert src==receipt["public_sha256"]
        pred=nor42.nor37.public_proposal(row)
        clean=pred["pred"]
        assert pred["sha"]==receipt["clean_model_prediction_sha256"]
        model_first_commit=producer.sha_of_proposed_output(clean)
        # Strict chronological modeling: proposal digest exists before signed
        # root or true target are opened to the verifier.
        true_clean,true_hidden,beta,op=nor42.restore_worlds(row,clean)
        assert true_clean==clean
        assert beta==receipt["hidden_beta"] and op==receipt["hidden_transform"]
        assert nor42.nor24.sha_target(true_hidden)==receipt[
            "private_hidden_truth_sha256_after_freeze"]
        S=frozenset(i for i,(a,b) in enumerate(zip(clean,true_hidden)) if a!=b)
        assert len(S)==receipt["difference_support_size"]
        assert min(S)==row["degree"]+32
        m=len(clean)
        checked=random.Random(seed(CHALLENGE_STREAM,src)).sample(range(m),K)
        assert len(set(checked))==K

        # Synthetic honest external reference provider signs TRUE hidden
        # target. This is a test oracle; its private key is public knowledge.
        hidden_tree=producer.OutputTree(true_hidden)
        signed=producer.signed_reference_manifest(hidden_tree,src,row["degree"],signing_key=producer.test_private_key())
        authority=verifier.verify_signed_manifest(
            signed,pinned,src,row["degree"],m,PROVIDER_ID)
        openings=[hidden_tree.opening(i) for i in checked]
        actual=verifier.verified_sample_rejection(
            clean,authority,openings,challenged_indices=checked)
        assert actual==bool(set(checked)&S)

        # The clean world is also a valid potential true target, and the same
        # proposed clean output is accepted when that is authenticated truth.
        clean_tree=producer.OutputTree(true_clean)
        clean_signed=producer.signed_reference_manifest(
            clean_tree,src,row["degree"],signing_key=producer.test_private_key())
        clean_authority=verifier.verify_signed_manifest(
            clean_signed,pinned,src,row["degree"],m,PROVIDER_ID)
        clean_openings=[clean_tree.opening(i) for i in checked]
        assert verifier.verified_sample_rejection(
            clean,clean_authority,clean_openings,
            challenged_indices=checked) is False

        # Adversarial self root may be perfectly consistent and signed under
        # an untrusted model-owned key, but fails pinned provider verification.
        attacker_signed=producer.signed_reference_manifest(
            clean_tree,src,row["degree"],signing_key=attacker)
        try:
            verifier.verify_signed_manifest(
                attacker_signed,pinned,src,row["degree"],m,PROVIDER_ID)
        except verifier.VerificationError:
            rejected_selfroot=True
        else:
            rejected_selfroot=False
        assert rejected_selfroot

        # Demonstrate the insecure arbitrary-public-key acceptance control.
        insecure_model_claim=verifier.verify_signed_manifest(
            attacker_signed,attacker_pub,src,row["degree"],m,PROVIDER_ID)
        assert verifier.verified_sample_rejection(
            clean,insecure_model_claim,clean_openings,
            challenged_indices=checked) is False

        source_exact=nor42.uniform_exact(m,len(S))
        assert [source_exact.numerator,source_exact.denominator]==receipt[
            "exact_uniform16_hit_fraction"]
        proofbytes=len(json.dumps(openings,sort_keys=True,
                            separators=(",",":"),ensure_ascii=True).encode())
        authbytes=len(json.dumps(signed,sort_keys=True,
                           separators=(",",":"),ensure_ascii=True).encode())
        assert model_first_commit==producer.sha_of_proposed_output(clean)
        measurements.append({
            "original_nor42_public_sha256":src,
            "original_replicate":row["pair_replicate"],
            "degree":row["degree"],"family":row["family"],
            "future_length":m,
            "hidden_difference_support_size":len(S),
            "proposed_clean_output_sha256":model_first_commit,
            "true_hidden_root_sha256":hidden_tree.root.hex(),
            "test_provider_manifest_sha256":signed["manifest_sha256"],
            "test_provider_public_key_sha256":sha256(pinned).hexdigest(),
            "true_hidden_uniform16_exact_hit_fraction":[
                source_exact.numerator,source_exact.denominator],
            "true_hidden_uniform16_exact_hit_probability":float(source_exact),
            "historical_k16_checked_indices":sorted(checked),
            "challenge_draw_hits_hidden_difference_support":actual,
            "self_signed_model_root_rejected_by_pinned_key":rejected_selfroot,
            "self_signed_model_root_passes_insecure_arbitrary_key_verifier":True,
            "witness_free_clean_self_consistency_accepts_hidden_world":True,
            "signed_manifest_payload_bytes":authbytes,
            "sixteen_openings_payload_bytes":proofbytes,
            "bytes_per_opening_mean":proofbytes/K,
            "merkle_depth":hidden_tree.depth,
            "first_hidden_difference_coefficient":min(S),
            "hidden_world_equal_public_prefix":(
                true_hidden[:row["degree"]+1]==clean[:row["degree"]+1])
        })
        print("NOR43 archived-only source",index+1,"/64",flush=True)
    assert len(measurements)==64
    assert len({r["original_nor42_public_sha256"] for r in measurements})==64
    assert all(r["self_signed_model_root_rejected_by_pinned_key"]
               and r["self_signed_model_root_passes_insecure_arbitrary_key_verifier"]
               and r["hidden_world_equal_public_prefix"]
               for r in measurements)
    groups={family:[r for r in measurements if r["family"]==family]
            for family in ("wide_dense_small","wide_regime_blocks")}
    familymeans={k:{
        "source_pairs":len(v),
        "uniform16_exact_support_hit_mean":statistics.mean(
            x["true_hidden_uniform16_exact_hit_probability"] for x in v),
        "fixed_seed_k16_hidden_hit_rate":statistics.mean(
            x["challenge_draw_hits_hidden_difference_support"] for x in v),
        "mean_sixteen_proof_bytes":statistics.mean(
            x["sixteen_openings_payload_bytes"] for x in v),
        "mean_signed_manifest_bytes":statistics.mean(
            x["signed_manifest_payload_bytes"] for x in v),
        "mean_merkle_depth":statistics.mean(x["merkle_depth"] for x in v),
    } for k,v in groups.items()}
    return {
        "schema":"nor43.historical_nor42_signed_merkle_reference_smoke.v1",
        "original_nor42_public_fixture_sha256":FIRST_SOURCE_SHA,
        "original_nor42_first_grade_sha256":FIRST_GRADE_SHA,
        "historical_source_pairs":64,
        "new_prospective_nor43_source_units":0,
        "synthetic_test_reference_key_public_sha256":sha256(pinned).hexdigest(),
        "challenge_stream":CHALLENGE_STREAM,
        "checked_coefficients_per_pair":K,
        "all_proposals_committed_before_true_reference_manifest":True,
        "provider_private_key_public_test_only":True,
        "all_pinned_key_self_signed_attacks_rejected":True,
        "all_witnessfree_hidden_selfconsistency_controls_accept":True,
        "family_means":familymeans,
        "source_records":measurements,
        "limitations":"Synthetic hidden targets and PUBLIC reproducible test signing keys. "
                      "Ed25519 and SHA256 trust are computational assumptions. "
                      "No real-world authenticated publisher, external truth "
                      "or secret audit coins are supplied."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=historical_source_smoke()
    raw=canonical(result)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(raw)
    print("NOR43 HISTORICAL ROOT SMOKE BYTES SHA",len(raw),sha256(raw).hexdigest())
    print("NOR43 FAMILY MEANS",result["family_means"])


if __name__=="__main__":
    main()
