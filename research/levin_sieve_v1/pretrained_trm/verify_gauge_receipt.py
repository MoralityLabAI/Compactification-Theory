"""Standalone standard-library replay of LSV-15 transverse LayerNorm claims."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from gauge_quotient import replay_receipt

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('receipt',type=Path)
    p.add_argument('--baseline',type=Path,default=HERE/'results'/'lsv13_arb_validated_64_20261104.json')
    p.add_argument('--lock',type=Path,default=HERE/'source_lock.json')
    v=p.parse_args()
    out=replay_receipt(json.loads(v.receipt.read_text()),
                json.loads(v.lock.read_text()),json.loads(v.baseline.read_text()))
    print(json.dumps({'status':'REPLAY_OK', 'counts':out},indent=2,sort_keys=True))

if __name__=='__main__':main()
