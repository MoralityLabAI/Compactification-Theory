"""Replay exact small-prime fixed one-flip threshold enumeration."""
import argparse,json
from pathlib import Path
from oic32_small import reference
FILE=Path(__file__).resolve().parent/'results'/'oic32_exceptional.json'
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    current=reference()
    if args.check:
        if current!=json.loads(FILE.read_text(encoding='utf8')):
            raise AssertionError('OIC-32 exceptional small-prime receipt mismatch')
        print('OIC-32 exhaustive p=2 and p=3 minima verified')
    if args.write:
        FILE.parent.mkdir(parents=True,exist_ok=True)
        FILE.write_text(json.dumps(current,indent=2,sort_keys=True)+'\n',encoding='utf8')
    if not args.check and not args.write:
        print(json.dumps(current,indent=2,sort_keys=True))
