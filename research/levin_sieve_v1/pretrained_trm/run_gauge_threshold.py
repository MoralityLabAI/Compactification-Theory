"""Recalculate gauge quotient rank and sufficient transverse thresholds offline."""
import argparse
import gzip
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from gauge_threshold import proof_summary,verify_threshold_receipt


def read_full(path):
    p=Path(path)
    if p.suffix=='.gz':
        return json.loads(gzip.decompress(p.read_bytes()))
    return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source_receipt',type=Path)
    ap.add_argument('--source-lock',type=Path,default=HERE/'source_lock.json')
    ap.add_argument('--baseline',type=Path,default=HERE/'results'/'lsv13_arb_validated_64_20261104.json')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--verify',type=Path)
    a=ap.parse_args()
    source=read_full(a.source_receipt)
    lock=read_full(a.source_lock)
    baseline=read_full(a.baseline)
    results=proof_summary(source,lock,baseline)
    if a.verify:
        proof=read_full(a.verify)
        verify_threshold_receipt(proof,source,lock,baseline)
    if a.out:
        a.out.parent.mkdir(exist_ok=True,parents=True)
        a.out.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:results[k] for k in ('selected_cell_count','layernorm_gamma_all_nonzero',
        'complete_gauge_kernel_dimension','complete_layernorm_rank',
        'min_all_cases_top1_limiting_case','min_all_cases_top1_strict_radius',
        'min_all_cases_top_two_limiting_case','min_all_cases_top_two_strict_radius')},indent=2))

if __name__=='__main__':main()
