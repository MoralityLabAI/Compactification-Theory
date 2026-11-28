"""OIC-30 standard-library deterministic certificate replay."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from oic30_prime_blocks import verify_main_witness

HERE=Path(__file__).resolve().parent
TARGET=HERE/'results'/'oic30_reference.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    current=verify_main_witness()
    if args.write:
        TARGET.parent.mkdir(parents=True,exist_ok=True)
        TARGET.write_text(json.dumps(current,indent=2,sort_keys=True)+'\n',encoding='utf8')
    if args.check:
        prior=json.loads(TARGET.read_text(encoding='utf8'))
        if current!=prior:
            raise AssertionError('OIC-30 frozen receipt mismatch')
        print('OIC-30 exact prime-block, two-marker, and all-flip checks passed')
    if not (args.write or args.check):
        print(json.dumps(current,indent=2,sort_keys=True))

if __name__=='__main__':main()
