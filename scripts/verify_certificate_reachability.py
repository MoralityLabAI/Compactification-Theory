"""QO-13 deterministic source-bound paired report; no model API calls."""
import argparse,base64,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.ingression_dynamics_v1.certificate_reachability_pilot import (
 full_report,SCENARIOS,SEEDS,POLICIES,CHECKPOINTS)
SOURCES=(
 "research/ingression_dynamics_v1/certificate_reachability.py",
 "research/ingression_dynamics_v1/test_certificate_reachability.py",
 "research/ingression_dynamics_v1/certificate_reachability_pilot.py",
 "research/ingression_dynamics_v1/test_certificate_reachability_pilot.py",
 "research/ingression_dynamics_v1/CERTIFICATE_REACHABILITY.md",
 "research/ingression_dynamics_v1/anytime_active_quotient.py",
 "research/ingression_dynamics_v1/two_step_gap_probe.py",
 "scripts/verify_certificate_reachability.py"
)
DEST=ROOT/"results/certificate_reachability/receipt.json"
def canonical():
 report=full_report()
 expected=len(SCENARIOS)*len(SEEDS)*len(POLICIES)
 if (report["counts"]!={"trajectories":expected,
     "snapshots":expected*len(CHECKPOINTS),
     "queries":expected*CHECKPOINTS[-1]}
     or len(report["groups"])!=len(SCENARIOS)*len(CHECKPOINTS)*len(POLICIES)
     or len(report["trial_rows"])!=expected*len(CHECKPOINTS)
     or len(report["trajectory_hashes"])!=expected):
  raise AssertionError("incomplete experiment")
 if any(x["false_cert"] and x.get("covered",x.get("true_rates_covered"))
        for x in report["trial_rows"]):
  raise AssertionError("invalid QO-07 certificate on coverage event")
 report["source_sha256"]={
  name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
  for name in SOURCES}
 return (json.dumps(report,sort_keys=True,separators=(",",":"))+"\n").encode("ascii")
def main():
 p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
 g.add_argument("--check",action="store_true");g.add_argument("--emit",action="store_true")
 a=p.parse_args();payload=canonical()
 if a.emit or (a.check and not DEST.exists()):
  print("__QO13_B64_BEGIN__",flush=True)
  v=base64.b64encode(payload).decode("ascii")
  for i in range(0,len(v),4096):print(v[i:i+4096],flush=True)
  print("__QO13_B64_END__",flush=True)
 if a.check:
  if not DEST.exists():raise FileNotFoundError("QO-13 receipt missing")
  if DEST.read_bytes()!=payload:raise AssertionError("QO-13 source-bound receipt mismatch")
  print("QO-13 verified SHA256",hashlib.sha256(payload).hexdigest())
if __name__=="__main__":main()
