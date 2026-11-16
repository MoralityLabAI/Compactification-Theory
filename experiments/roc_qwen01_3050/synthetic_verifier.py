"""Exact rational donor-routing identities. Synthetic, no neural closure claims."""
from fractions import Fraction as F
import json


def dot(x,y):
    return sum((a*b for a,b in zip(x,y)),F(0))


def combine(frame,c):
    return [sum((v[j]*c[i] for i,v in enumerate(frame)),F(0)) for j in range(len(frame[0]))]


def verify():
    u = [[F(3,5),F(4,5),F(0),F(0),F(0)], [F(0),F(0),F(1),F(0),F(0)]]
    v = [[F(-4,5),F(3,5),F(0),F(0),F(0)], [F(0),F(0),F(0),F(1),F(0)]]
    for frame in (u,v):
        assert [[dot(a,b) for b in frame] for a in frame] == [[F(1),F(0)],[F(0),F(1)]]
    assert all(dot(a,b)==0 for a in u for b in v)
    count = 0
    for c in ([F(3,5),F(4,5)],[F(5,13),F(12,13)],[F(1),F(0)]):
        for radius in (F(1,4),F(1,2),F(2)):
            observed = combine(u,[radius*a for a in c])
            dark = combine(v,[radius*a for a in c])
            assert dot(observed,observed) == dot(dark,dark) == radius*radius
            assert all(dot(a,dark)==0 for a in u)
            assert [dot(a,observed) for a in u] == [radius*a for a in c]
            # UV^T and VU^T preserve lengths on the declared source frames.
            assert combine(u,[dot(a,dark) for a in v]) == observed
            assert combine(v,[dot(a,observed) for a in u]) == dark
            count += 1
    return dict(source_type="synthetic_exact_rational", cases=count,
                statements=["orthogonal_frames", "norm_matched_isometries", "source_target_routing"],
                neural_claims=False, recurrent_closure_claim=False)


if __name__ == "__main__":
    print(json.dumps(verify(),indent=2))
