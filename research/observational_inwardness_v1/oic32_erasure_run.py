"""OIC-32 exact fixed one-erasure prime classification replay."""
import argparse,json
from pathlib import Path
from oic32_erasure import reference
FILE=Path(__file__).resolve().parent/'results'/'oic32_erasure.json'
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    current=reference()
    if args.check:
        if current!=json.loads(FILE.read_text(encoding='utf8')):
            raise AssertionError('erasure-reference replay mismatch')
        print('OIC-32 erasure exact general prime theorem and exceptional cases verified')
    if args.write:
        FILE.parent.mkdir(parents=True,exist_ok=True)
        FILE.write_text(json.dumps(current,sort_keys=True,indent=2)+'\n',encoding='utf8')
    if not args.check and not args.write:
        print(json.dumps(current,sort_keys=True,indent=2))
