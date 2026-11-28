"""SHAFT-COMP-15: exact 2×2 example of readout-rank versus direction.

Two orthogonal rank-one monitors act on the same two-dimensional state.
The rank is unchanged across contrasts; the directional energy depends on
both the monitored coordinate and the state contrast.
"""
import json


def energy(readout,contrast):
    return sum(a*b for a,b in zip(readout,contrast))**2


def reference():
    learned=(1,0)
    native=(0,1)
    contrasts={"first_axis":(1,0),"second_axis":(0,1)}
    return {
        "status":"exact_rank_one_2x2_reference",
        "both_readout_ranks":1,
        "rows":{"learned":list(learned),"native":list(native)},
        "contrasts":{name:{
            "learned_direction_energy":energy(learned,v),
            "native_direction_energy":energy(native,v),
            "learned_rank":1,
            "native_rank":1,
        } for name,v in contrasts.items()},
        "note":"rank is a dimension count, while energy is sensitive to the observed state contrast"
    }


if __name__=="__main__":
    print(json.dumps(reference(),indent=2,sort_keys=True))
