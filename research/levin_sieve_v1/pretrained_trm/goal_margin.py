"""Post-hoc exact-rational goal-readout certificate (not an end-to-end dynamic proof).

Given two *recorded*, finite FP32 hidden states, certify the affine readout
label for any hidden state within their per-cell infinity-norm separation of
one clean hidden state. The worst-case ball is computed using stored binary
floating-point values interpreted as exact rationals. No PyTorch arithmetic
roundoff is included in the mathematical statement.
"""
from fractions import Fraction as Q
import torch


def rational(v):
    return Q.from_float(float(v))


def head_geometry(model):
    weights=tuple(tuple(map(rational,row)) for row in model.head.weight.detach().tolist())
    biases=tuple(map(rational,model.head.bias.detach().tolist()))
    worst=[]
    for r,row in enumerate(weights):
        worst.append(max(sum((abs(a-b) for a,b in zip(row,other)),Q(0))
                         for s,other in enumerate(weights) if s!=r))
    return weights,biases,tuple(worst)


def exact_head_label_and_margin(z, head):
    weights,biases,worst=head
    x=tuple(map(rational,z))
    logits=tuple(sum((w*v for w,v in zip(row,x)),b) for row,b in zip(weights,biases))
    order=sorted(range(len(logits)),key=lambda c:logits[c],reverse=True)
    return order[0],logits[order[0]]-logits[order[1]],worst[order[0]]


def certify_paired_hidden_states(model,clean,altered):
    """Per-cell readout margin vs measured per-cell hidden perturbation envelope.

    Returns a *post-hoc* guarantee about the final linear readout. The ball
    radius depends on measured altered hidden states, so it is not a
    prior-to-execution certificate of recovery of the recurrent dynamics.
    """
    if clean.shape!=altered.shape or clean.ndim!=3 or clean.shape[-1]!=48:
        raise ValueError('states must be paired [batch,81,48] arrays')
    head=head_geometry(model)
    n_cells=clean.shape[0]*clean.shape[1]
    successes=actual_exact_label_same=0
    max_possible=0
    for x,y in zip(clean.reshape(-1,48).detach().tolist(),
                   altered.reshape(-1,48).detach().tolist()):
        clean_label,margin,k=exact_head_label_and_margin(x,head)
        change=max((abs(rational(a)-rational(b)) for a,b in zip(x,y)),default=Q(0))
        certified=margin > k*change
        other_label,other_margin,_=exact_head_label_and_margin(y,head)
        actual_exact_label_same+=int(clean_label==other_label)
        successes+=int(certified)
        if certified and clean_label!=other_label:
            raise AssertionError('soundness violation in exact readout certificate')
        max_possible+=int(margin>0)
    return {'n_cells':n_cells,'certified_unchanged_cells':successes,
            'certified_fraction':round(successes/n_cells,6),
            'observed_exact_affine_same_label_cells':actual_exact_label_same,
            'observed_exact_affine_stability_fraction':round(actual_exact_label_same/n_cells,6),
            'positive_clean_margin_cells':max_possible,
            'scope':'real-arithmetic final affine head, measured per-cell hidden delta; not end-to-end perturbation guarantee'}
