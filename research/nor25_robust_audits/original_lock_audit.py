"""NOR-25 immutable first raw-byte receipt auditor.

Checks the original hosted source-only fixture and complete outcome against
the SHA-locked public first-run receipt. Optional --regenerate regenerates
both files deterministically if the short-lived Actions artifact has expired.
The separate-process decision replay remains a different script.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
RECEIPT=HERE/"results"/"nor25_original_first_receipt.json"


def check(fixture_file, outcome_file):
    ref=json.loads(RECEIPT.read_text())
    raw_source=Path(fixture_file).read_bytes()
    raw_outcome=Path(outcome_file).read_bytes()
    for name,data in (("fixture",raw_source),("outcome",raw_outcome)):
        expected=ref[name]
        digest=sha256(data).hexdigest()
        assert len(data)==expected["bytes"],(name,"bytes",len(data))
        assert digest==expected["sha256"],(name,"sha256",digest)
    src=json.loads(raw_source)
    out=json.loads(raw_outcome)
    assert src["source_only"] and src["private_truth_absent"] and src["attack_sets_absent"]
    assert len(src["cases"])==12
    source_order=[x["full_public_input_sha256"] for x in src["cases"]]
    assert source_order==ref["source_public_sha256_order"]
    assert [x["public_sha256"] for x in out["source_records"]]==source_order
    assert len(out["primary_by_source"])==12
    assert out["registered_primary_pass"]==ref["primary"]["registered_pass"]
    assert out["wins_hybrid50_vs_uniform"]==ref["primary"]["wins_hybrid50_vs_uniform"]
    assert out["offtemplate_wins_hybrid50_vs_static"]==ref["primary"]["offtemplate_wins_hybrid50_vs_static"]
    assert isclose(out["source_mean_weighted_gain_vs_uniform"],
                   ref["primary"]["weighted_source_mean_hybrid50_minus_uniform"],
                   rel_tol=0,abs_tol=1e-15)
    assert isclose(out["source_mean_offtemplate_hybrid50"],
                   ref["primary"]["source_mean_offtemplate_hybrid50"],
                   rel_tol=0,abs_tol=1e-15)
    assert all(len(x["observations"])==360 and x["informed_attack_detected"]==0
               and x["delta_invariance_failures"]==0 and x["valid_target_failures"]==0
               for x in out["source_records"])
    print("NOR25 IMMUTABLE FIRST SOURCE AND OUTCOME PASS",
          len(raw_source),sha256(raw_source).hexdigest(),
          len(raw_outcome),sha256(raw_outcome).hexdigest())


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fixture",type=Path)
    ap.add_argument("--full",type=Path)
    ap.add_argument("--regenerate",action="store_true")
    args=ap.parse_args()
    if args.regenerate:
        if args.fixture or args.full:
            ap.error("regenerate is separate from checking supplied original files")
        from research.nor25_robust_audits import experiment as e
        with tempfile.TemporaryDirectory(prefix="nor25_locked_") as t:
            a=Path(t)/"nor25_fixture.json"
            b=Path(t)/"nor25_full.json"
            a.write_bytes(e.prior24.canonical(e.make_fixture()))
            b.write_bytes(e.prior24.canonical(e.evaluate(a)))
            check(a,b)
    elif args.fixture and args.full:
        check(args.fixture,args.full)
    else:
        ap.error("supply --fixture and --full, or --regenerate")
