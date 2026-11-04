"""Deterministic proof-coverage comparison, not an AI discovery benchmark."""
import json
from fractions import Fraction as Q
from small_gain import Certificate


def run() -> dict[str, int | str]:
    values = (Q(0), Q(1,10), Q(1,2), Q(9,10), Q(2), Q(4))
    uniform = weighted = rescued = failed = 0
    for a in values:
        for b in values:
            mat = ((Q(0),a),(b,Q(0)))
            baseline = max(a,b)<1
            if baseline:
                Certificate.from_weights(mat,(1,1))
                uniform += 1
            try:
                certificate = Certificate.synthesize(mat)
                certificate.validate()
                weighted += 1
                if not baseline:
                    rescued += 1
                assert a*b<1
            except ValueError:
                failed += 1
                assert a*b>=1
    return {"family":"2-node nonnegative cross-coupling grid",
            "grid_values": "0, 1/10, 1/2, 9/10, 2, 4",
            "cases":len(values)**2, "uniform_sup_norm_certified":uniform,
            "weighted_certificate_certified":weighted,
            "additional_certifiable_cases":rescued,
            "outside_small_gain_regime":failed}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
