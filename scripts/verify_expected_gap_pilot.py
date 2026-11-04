"""QO-11 compare-or-fail source-bound paired-pilot receipt."""
import argparse,base64,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.ingression_dynamics_v1.expected_gap_pilot import full_report,SCENARIOS,SEEDS,POLICIES,CHECKPOINTS
SOURCES=(
 "research/ingression_dynamics_v1/expected_gap_probe.py",
 "research/ingression_dynamics_v1/expected_gap_pilot.py",
 "research/ingression_dynamics_v1/test_expected_gap_probe.py",
 "research/ingression_dynamics_v1/EXPECTED_GAP_THEORY.md",
 "research/ingression_dynamics_v1/anytime_active_quotient.py",
 "research/ingression_dynamics_v1/paired_probe_pilot.py",
 "research/ingression_dynamics_v1/decision_boundary_probe.py",
 "scripts/verify_expected_gap_pilot.py",
)
DEST=ROOT/"results/expected_gap_pilot/receipt.json"
def canonical():
 r=full_report()
 n=len(SCENARIOS)*len(SEEDS)*len(POLICIES)
 assert r["run_counts"]=={"trajectories":n,"snapshots":n*len(CHECKPOINTS),"queries":n*CHECKPOINTS[-1],"stream_replays":n,"planning_traces":len(SCENARIOS)*len(SEEDS)*2}
 assert len(r["groups"])==len(SCENARIOS)*len(CHECKPOINTS)*len(POLICIES)
 assert len(r["paired_comparisons"])==len(SCENARIOS)*len(CHECKPOINTS)*2*(len(POLICIES)-1)
 assert len(r["trial_rows"])==n*len(CHECKPOINTS)
 assert all(not v["false_cert"] or not v["true_rates_covered"] for v in r["trial_rows"])
 assert all(p["gap_nonincreasing"] for p in r["planning_receipts"])
 r["source_sha256"]={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES}
 return (json.dumps(r,sort_keys=True,separators=(",",":"))+"\n").encode("ascii")
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
 g.add_argument("--check",action="store_true");g.add_argument("--emit",action="store_true")
 a=p.parse_args();expected=canonical()
 if a.emit or (a.check and not DEST.exists()):
  print("__QO11_B64_BEGIN__",flush=True)
  b64=base64.b64encode(expected).decode("ascii")
  for i in range(0,len(b64),4096): print(b64[i:i+4096],flush=True)
  print("__QO11_B64_END__",flush=True)
 if a.check:
  if not DEST.exists(): raise FileNotFoundError("QO-11 receipt missing")
  if DEST.read_bytes()!=expected: raise AssertionError("QO-11 receipt/source mismatch")
  print("QO-11 checked",hashlib.sha256(expected).hexdigest())
if __name__=="__main__": main()
