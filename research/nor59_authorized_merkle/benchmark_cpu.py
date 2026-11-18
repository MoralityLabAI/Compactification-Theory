"""NOR59 first source-frozen actual signed NOR45 authorized-program CPU benchmark.

All three arms verify identical NOR45 program-authority certificates, NOR44
verifier precommits and NOR44 signed provider references with original NOR45
full interpreter. The only algorithmic change is NOR45 full Merkle root.
Input sources/programs are SHA-frozen by a separate upstream CI job.
"""
from __future__ import annotations
import argparse,json,os,platform,random,statistics,sys,time
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor45_authorized_program import program_authority as authority
from research.nor44_session_protocol import session_protocol as session
from research.nor43_authenticated_witness import signed_witness as trees
from research.nor58_static_merkle import template
from research.nor59_authorized_merkle import authorized_root,cohort

SCHEMA="nor59.first_60_full_NOR45_signed_authorized_static_root_CPU.v1"
REGISTERED=cohort.REGISTERED
REPEATS=5
ARMS=(
    "original_authorized_full_root",
    "warm_authorized_static_root",
    "cold_authorized_static_root"
)
MODE_BY_ARM={ARMS[0]:"original",ARMS[1]:"warm",ARMS[2]:"cold"}
EPOCH_PRECOMMIT=1
EPOCH_VERIFY=2

def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")
def med(vals):
    return statistics.median(vals) if vals else None

def summarize(rows):
    if not rows:return {"count":0}
    return {
        "count":len(rows),
        "median_sourcewise_warm_over_original":med(r["warm_over_original"] for r in rows),
        "median_sourcewise_cold_over_original":med(r["cold_over_original"] for r in rows),
        "median_sourcewise_root_only_warm_over_original":med(
            r["untimed_root_only_warm_over_original"] for r in rows),
        "median_arm_ms":{arm:med(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS},
        "source_median_total_ms":{
            arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS},
        "median_geometry_template_build_ms":med(
            r["template_build_ms_outside_arms"] for r in rows)
    }

def bootstrap(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r["observed_degree"],r["program_profile"])].append(r)
    if len(groups)!=10 or any(len(r)!=6 for r in groups.values()):
        return {"status":"missing registered ten 6-case groups"}
    keys=sorted(groups)
    totals={arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS}
    results={}
    for j,arm in enumerate(ARMS[1:]):
        rng=random.Random(202611590010+j)
        draw=[]
        for _ in range(4096):
            selected=[r for block in (rng.choice(keys) for _ in keys)
                      for r in groups[block]]
            draw.append(sum(r["arm_medians_ms"][arm] for r in selected)/
                        sum(r["arm_medians_ms"][ARMS[0]] for r in selected))
        draw.sort()
        results[arm]={
            "source_median_ratio_of_sums_to_original":totals[arm]/totals[ARMS[0]],
            "descriptive_95pct_ten_block_interval":[draw[102],draw[3993]]
        }
    return {"block_count":10,"bootstrap_repeats":4096,
            "sum_source_median_ms":totals,"descriptive_ratios":results}

def prewarm_original_signed_geometries():
    ledger={}
    for d in cohort.DEGREES:
        for s in cohort.STRIDES:
            m=3*d//2+1
            start=time.perf_counter_ns()
            temp=authorized_root.warm_geometry_only_template(m,s)
            build_ns=time.perf_counter_ns()-start
            ledger[f"m{m}-s{s}"]={
                "m":m,"observed_degree":d,"source_stride":s,
                "template_SHA256":temp.template_sha256,
                "build_ns_outside_all_timed_arms":build_ns,
                "canonical_template_bytes":temp.canonical_template_bytes,
                "static_node_count":temp.static_node_count,
                "static_zero_leaf_count":temp.static_zero_leaf_count,
                "static_padding_leaf_count":temp.static_padding_leaf_count,
                "dynamic_supported_leaf_count":len(temp.dynamic_leaf_indices),
                "trusted_geometry_only":True
            }
    assert len(ledger)==6
    return ledger

def make_signed_bundles(query,program,degree,source_sha,full_output,expected_root):
    authority_signer=authority.test_authority_private_key()
    provider_signer=trees.test_private_key()
    verifier_signer=session.test_verifier_private_key()
    authority_pub=authority.public_key_of(authority_signer)
    provider_pub=trees.test_reference_public_key()
    verifier_pub=session.public_key_bytes(verifier_signer)
    authority.validate_program_spec(program,degree)
    tree=trees.OutputTree(full_output)
    if tree.root.hex()!=expected_root:
        raise ValueError("NOR43 signed provider tree not equal NOR45 original root")
    records=[]
    for rep in range(REPEATS):
        nonce=sha256(
            f"NOR59/VERIFIER-ISSUED-RESEARCH-NONCE/v1|{source_sha}|{rep}".encode("ascii")
        ).digest()
        start=time.perf_counter_ns()
        verifier_session=session.VerifierSession(
            source_sha256=source_sha,observed_degree=degree,
            coefficient_count=len(query),provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=provider_pub,
            verifier_signing_key=verifier_signer,expires_epoch=100,
            nonce=nonce,test_only=True)
        # Verifier precommits to the fully known public-source proposal BEFORE
        # receiving an executable program certificate and provider root.
        first_receipt=verifier_session.precommit(
            session.proposal_digest(query,verifier_session.context),
            now_epoch=EPOCH_PRECOMMIT)
        provider_ref=session.sign_provider_reference(
            tree,first_receipt,verifier_session.context,
            verifier_pub,provider_signer,now_epoch=EPOCH_VERIFY)
        certificate=authority.sign_authorized_program(
            program=program,query=query,
            context=verifier_session.context,
            precommit_receipt=first_receipt,
            pinned_verifier_public_key=verifier_pub,
            expected_output_merkle_root=expected_root,
            program_authority_signer=authority_signer,
            now_epoch=EPOCH_VERIFY)
        elapsed=time.perf_counter_ns()-start
        kwargs={
            "certificate":certificate,"query":query,
            "session_context":verifier_session.context,
            "precommit_receipt":first_receipt,
            "provider_reference":provider_ref,
            "pinned_program_authority_key":authority_pub,
            "expected_program_authority_id":authority.TEST_AUTHORITY_ID,
            "pinned_verifier_public_key":verifier_pub,
            "pinned_provider_public_key":provider_pub,
            "now_epoch":EPOCH_VERIFY
        }
        records.append({
            "kwargs":kwargs,
            "test_only_public_fixture_signer":True,
            "signed_setup_ns_outside_all_timed_arms":elapsed,
            "session_nonce_SHA256":sha256(nonce).hexdigest(),
            "signed_authority_cert_bytes":len(session.canonical(certificate)),
            "signed_verifier_precommit_bytes":len(session.canonical(first_receipt)),
            "signed_provider_reference_bytes":len(session.canonical(provider_ref)),
            "signed_context_SHA256":sha256(session.canonical(
                verifier_session.context)).hexdigest()
        })
    assert len({r["session_nonce_SHA256"] for r in records})==REPEATS
    return records

def run_arm(arm,signed_bundle):
    kwargs=signed_bundle["kwargs"]
    if arm==ARMS[0]:
        return nor45.verify_authorized_execution(**kwargs)
    return nor45.verify_authorized_execution(
        **kwargs,_root_policy=MODE_BY_ARM[arm])

def run(public_file:Path,repeats=REPEATS):
    if repeats!=REPEATS:raise ValueError("first NOR59 requires exactly five repeats")
    fixture,source_sha=cohort.read_fixture(public_file)
    ids={row["source_sha256"] for row in fixture["cases"]}
    assert len(ids)==60 and ids.isdisjoint(cohort.historical_ids())
    original_full_root_function=nor45.independently_computed_root
    original_program_function=nor45.execute_authorized_program
    warm_ledger=prewarm_original_signed_geometries()
    records=[]
    for ordinal,case in enumerate(fixture["cases"],start=1):
        identity={k:case[k] for k in (
            "source_sha256","observed_degree","full_length","source_stride",
            "program_profile","coefficient_magnitude_bits","replicate","family")}
        try:
            query=tuple(int(v) for v in case["public_query_coefficients_decimal"])
            degree=case["observed_degree"]
            program=case["authorized_program"]
            if len(query)!=case["full_length"]:
                raise ValueError("source length differs from original public roster")
            if authority.validate_program_spec(program,degree)!=program:
                raise ValueError("original NOR45 program not authorized")
            start=time.perf_counter_ns()
            output=nor45.execute_authorized_program(program,query,degree)
            reference_generation_ns=time.perf_counter_ns()-start
            start=time.perf_counter_ns()
            root=nor45.independently_computed_root(output)
            reference_root_ns=time.perf_counter_ns()-start
            s=authorized_root.certify_actual_NOR45_program_support(
                program=program,query=query,observed_degree=degree,output=output)
            if s!=case["source_stride"]:
                raise ValueError("signed source support closure changed registered stride")
            geom_key=f"m{len(output)}-s{s}"
            geom=warm_ledger[geom_key]
            cache=authorized_root.warm_geometry_only_template(len(output),s)
            if cache.root(output)!=root:
                raise ValueError("original full NOR45 Merkle root mismatch with static geometry")
            signed=make_signed_bundles(
                query,program,degree,case["source_sha256"],output,root)
            # Ensure three routes accept the SAME full signed authority,
            # signed provider and precommit receipts with the exact result.
            for arm in ARMS:
                actual=run_arm(arm,signed[0])
                if actual["values"]!=output or actual["root_sha256"]!=root:
                    raise ValueError("pretime signed NOR45 program/full Merkle mismatch "+arm)
            samples={arm:[] for arm in ARMS}
            for rep in range(REPEATS):
                order=list(ARMS)
                seed=int.from_bytes(
                    sha256(f"NOR59/ARM-ORDER/v1|{case['source_sha256']}|{rep}".encode(
                           "ascii")).digest(),"big")
                random.Random(seed).shuffle(order)
                for arm in order:
                    begin=time.perf_counter_ns()
                    actual=run_arm(arm,signed[rep])
                    elapsed=(time.perf_counter_ns()-begin)/1e6
                    if actual["values"]!=output or actual["root_sha256"]!=root:
                        raise ValueError("timed full signed NOR45 authorization mismatch "+arm)
                    if (nor45.independently_computed_root is not original_full_root_function or
                        nor45.execute_authorized_program is not original_program_function):
                        raise RuntimeError("NOR45 original authenticated helpers were monkeypatched")
                    samples[arm].append(elapsed)
            arm_medians={a:med(samples[a]) for a in ARMS}
            # The following root-only diagnostic runs AFTER complete signed
            # whole-operation timing and cannot influence its frozen first score.
            begin=time.perf_counter_ns()
            original_root=nor45.independently_computed_root(output)
            root_original_ns=time.perf_counter_ns()-begin
            begin=time.perf_counter_ns()
            warm_root=cache.root(output)
            root_warm_ns=time.perf_counter_ns()-begin
            if original_root!=warm_root or original_root!=root:
                raise ValueError("post-timing original NOR45 Merkle root-only parity failure")
            ratios={
                "warm_over_original":arm_medians[ARMS[1]]/arm_medians[ARMS[0]],
                "cold_over_original":arm_medians[ARMS[2]]/arm_medians[ARMS[0]],
                "untimed_root_only_warm_over_original":root_warm_ns/root_original_ns,
                "amortized_five_warm_with_template_build_over_original":
                    (5*arm_medians[ARMS[1]]+
                     geom["build_ns_outside_all_timed_arms"]/1e6)/(5*arm_medians[ARMS[0]])
            }
            records.append({
                **identity,"source_success":True,
                "authorized_program":program,
                "all_three_full_signed_authorized_program_outputs_and_original_roots_equal":True,
                "certified_public_support_stride":s,
                "original_full_merkle_root_sha256":root,
                "reference_program_execution_ns_outside_arms":reference_generation_ns,
                "reference_original_root_ns_outside_arms":reference_root_ns,
                "original_program_authority_signer_public_research_fixture":True,
                "fixture_five_signed_setup_ns_outside_arms":
                    sum(r["signed_setup_ns_outside_all_timed_arms"] for r in signed),
                "first_signed_certificate_bytes":signed[0]["signed_authority_cert_bytes"],
                "first_signed_verifier_receipt_bytes":signed[0]["signed_verifier_precommit_bytes"],
                "first_signed_provider_reference_bytes":signed[0]["signed_provider_reference_bytes"],
                "five_independent_signed_context_SHA256":[
                    r["signed_context_SHA256"] for r in signed],
                "five_distinct_verifier_session_nonce_SHA256":[
                    r["session_nonce_SHA256"] for r in signed],
                "template_geometry_only_key":geom_key,
                "template_SHA256":geom["template_SHA256"],
                "template_build_ms_outside_arms":
                    geom["build_ns_outside_all_timed_arms"]/1e6,
                "template_static_node_count":geom["static_node_count"],
                "template_canonical_bytes":geom["canonical_template_bytes"],
                "arm_medians_ms":arm_medians,
                "five_interleaved_samples_ms":samples,
                "untimed_root_only_original_ns":root_original_ns,
                "untimed_root_only_warm_ns":root_warm_ns,
                **ratios
            })
            print("NOR59 FIRST CPU",ordinal,"/60",f"D{degree}",f"s{s}",
                  f"P{case['program_profile']}",
                  "signed warm/orig",round(ratios["warm_over_original"],4),
                  "cold/orig",round(ratios["cold_over_original"],4),flush=True)
        except Exception as exc:
            records.append({**identity,"source_success":False,
                            "failure_type":type(exc).__name__,
                            "failure_message":str(exc)[:500]})
            print("NOR59 FIRST SOURCE FAIL",ordinal,"/60",type(exc).__name__,
                  str(exc)[:180],flush=True)
    assert len(records)==60 and len({r["source_sha256"] for r in records})==60
    successful=[r for r in records if r["source_success"]]
    assert Counter(r["observed_degree"] for r in records)=={512:30,1024:30}
    assert Counter(r["source_stride"] for r in records)=={1:20,4:20,8:20}
    sparse=[r for r in successful if r["source_stride"]>1]
    dense=[r for r in successful if r["source_stride"]==1]
    all_summary=summarize(successful)
    sparse_summary=summarize(sparse)
    dense_summary=summarize(dense)
    same_original=(nor45.independently_computed_root is original_full_root_function
                    and nor45.execute_authorized_program is original_program_function)
    gates={
        "forty_sparse_authorized_warm_over_original_median_under_1p02":(
            len(sparse)==40 and sparse_summary["median_sourcewise_warm_over_original"]<1.02),
        "forty_sparse_NOR45_root_only_warm_over_original_under_0p70":(
            len(sparse)==40 and sparse_summary[
                "median_sourcewise_root_only_warm_over_original"]<.70),
        "twenty_dense_cold_whole_authorized_over_original_under_1p25":(
            len(dense)==20 and dense_summary["median_sourcewise_cold_over_original"]<1.25),
        "all_60_signed_authorized_full_output_root_security_and_support":(
            len(successful)==60 and all(
                r["all_three_full_signed_authorized_program_outputs_and_original_roots_equal"]
                and r["certified_public_support_stride"]==r["source_stride"]
                for r in successful)),
        "first_60_public_program_sources_528_prior_exclusions_full_archive":(
            len(successful)==60 and fixture["public_only"] and same_original
            and ids.isdisjoint(cohort.historical_ids()) and len(warm_ledger)==6
            and all(all(len(r["five_interleaved_samples_ms"][arm])==5
                        for arm in ARMS) for r in successful)
            and all(len(r["five_distinct_verifier_session_nonce_SHA256"])==5
                    and len(set(r["five_distinct_verifier_session_nonce_SHA256"]))==5
                    and r["template_static_node_count"]>=0 for r in successful))
    }
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:rss=None
    return {
        "schema":SCHEMA,"design_issue":276,
        "first_registration_sha":REGISTERED,
        "first_public_only_source_sha256":source_sha,
        "source_count":60,"source_success_count":len(successful),
        "source_failure_count":60-len(successful),
        "excluded_prior_first_source_ID_count":528,
        "repeats":REPEATS,"arms":list(ARMS),
        "source_first_SHA_frozen_upstream_before_reference_or_timing":True,
        "original_NOR45_authority_precommit_provider_checks_same_in_every_arm":True,
        "each_arm_recomputes_all_original_signed_program_integer_coefficients":True,
        "warm_template_internal_geometry_only_no_result_cache":True,
        "cold_template_build_charged_per_signed_verification":True,
        "original_NOR45_program_and_Merkle_helpers_unmodified_runtime":same_original,
        "fixture_signing_and_reference_production_outside_whole_verifier_timing":True,
        "stateless_NOR45_verifier_per_session_nonce_binding":True,
        "first_signed_authentication_private_keys_public_test_only":True,
        "geometry_template_ledger":warm_ledger,
        "source_free_support_closure":authorized_root.source_free_closure_explanation(),
        "python_version":sys.version.split()[0],
        "cpu_model":platform.processor() or "unknown",
        "logical_cpu_count":os.cpu_count(),"maxrss_kib":rss,
        "overall":all_summary,
        "primary_forty_sparse_signed_authorized":sparse_summary,
        "primary_twenty_dense_cold":dense_summary,
        "by_degree":{str(d):summarize([r for r in successful if r["observed_degree"]==d])
                     for d in cohort.DEGREES},
        "by_stride":{str(s):summarize([r for r in successful if r["source_stride"]==s])
                     for s in cohort.STRIDES},
        "by_program_profile":{str(p):summarize([
            r for r in successful if r["program_profile"]==p]) for p in cohort.PROFILES},
        "secondary_ten_block_bootstrap":bootstrap(successful),
        "first_original_five_registered_gates":gates,
        "first_original_engineering_pass":all(gates.values()),
        "source_records":records,
        "scope":"Original NOR45 authorized signed program d512/1024 m769/1537, publicly known research signing fixture keys, honest program-authority/source premises, exact original full Merkle; no succinctness, ZK, truth-world authorization or deployed nonce store."
    }
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--public-file",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=REPEATS)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.public_file,args.repeats)
    raw=canonical(result)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(raw)
    print("NOR59 FIRST ORIGINAL CPU RAW BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR59 FIRST ORIGINAL FIVE REGISTERED GATES",result["first_original_five_registered_gates"])
    print("NOR59 FIRST ORIGINAL ENGINEERING PASS",result["first_original_engineering_pass"])
if __name__=="__main__":main()
