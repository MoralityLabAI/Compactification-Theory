"""CPU-only exact-integer certificate runner for the infinite Dirac resolvent."""
import argparse
from pathlib import Path
from infinite_resolvent import build_payload,verify_payload,write_payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/infinite_resolvent_certificate.json')
    parser.add_argument('--verify',help='Verify a JSON certificate without any numerical solver')
    parser.add_argument('--cutoff',type=int,default=2048)
    a=parser.parse_args()
    if a.verify:
        import json
        result=json.loads(Path(a.verify).read_text())
        verify_payload(result)
    else:
        result=build_payload(cutoff=a.cutoff)
        write_payload(Path(a.output),result)
        verify_payload(result)
    for case in result['cases']:
        c=case['certificate']
        print(f"M={case['M']}: ||R_inf F - J U|| <= {c['error_upper']:.9g} "
              f"(near residual {c['near_residual_upper']:.8g}; "
              f"tail {c['tail_residual_upper']:.8g})")


if __name__=='__main__': main()
