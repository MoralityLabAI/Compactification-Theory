"""Reproduce OIC-20 exact minimum-marker reference and check frozen receipt."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from oic20_markers import reference

HERE = Path(__file__).resolve().parent
PINNED = HERE / 'results' / 'oic20_reference.json'


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--out', type=Path)
    args=parser.parse_args()
    current=reference()
    if args.check and current != json.loads(PINNED.read_text(encoding='utf-8')):
        raise SystemExit('OIC-20 EXACT REPLAY MISMATCH')
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(current,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('OIC-20 exact sensor designs:',len(current['cases']),
          'receipt check:', 'passed' if args.check else 'not requested')

if __name__=='__main__': main()
