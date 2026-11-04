"""QO-14: dependency-free linear and fully trained tiny tanh-RNN students.

The recurrent input and recurrent matrices are BOTH trained by
backpropagation through the two-cell sequence, using Adam, with a
two-output sigmoid head regressing QO-13 exact horizon-two action
values. All trainable weights and the optimizer are implemented with
standard-library floats, deterministic source-fixed iteration order.

The RNN is a genuinely trained 2-step recurrent neural model, rather
than a fixed reservoir. It makes no API calls or external model claims.
"""

from copy import deepcopy
from math import exp, tanh, sqrt
from random import Random

from research.ingression_dynamics_v1.certificate_student_data import (
    evidence_features
)

HIDDEN=8
INPUTS=6
OUTPUTS=2
STUDENT_SEED=20261104
EPOCHS=65
LEARNING_RATE=0.012
CLIP_NORM=3.0


def sigmoid(x):
    if x>=0:
        z=exp(-x)
        return 1.0/(1.0+z)
    z=exp(x)
    return z/(1.0+z)


def flatten(tokens):
    if len(tokens)!=2 or any(len(row)!=INPUTS for row in tokens):
        raise ValueError("student requires two six-feature public cell tokens")
    return tuple(float(v) for row in tokens for v in row)


def create_linear(seed=STUDENT_SEED):
    rng=Random(seed)
    return {
        "kind":"linear",
        "w":[[rng.uniform(-0.18,0.18) for _ in range(INPUTS*2)]
             for _ in range(OUTPUTS)],
        "b":[0.0,0.0],
    }


def create_rnn(seed=STUDENT_SEED):
    rng=Random(seed)
    return {
        "kind":"rnn",
        "wx":[[rng.uniform(-0.16,0.16) for _ in range(INPUTS)]
              for _ in range(HIDDEN)],
        "wh":[[rng.uniform(-0.13,0.13) for _ in range(HIDDEN)]
              for _ in range(HIDDEN)],
        "bh":[0.0]*HIDDEN,
        "wo":[[rng.uniform(-0.13,0.13) for _ in range(HIDDEN)]
              for _ in range(OUTPUTS)],
        "bo":[0.0]*OUTPUTS,
    }


def trainable_count(model):
    return sum(len(row) for name,mat in model.items()
               if name!="kind" for row in (
                   mat if isinstance(mat[0],list) else [mat]))


def _dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def forward(model,tokens,cache=False):
    x=flatten(tokens)
    if model["kind"]=="linear":
        preds=tuple(sigmoid(_dot(w,x)+b)
                    for w,b in zip(model["w"],model["b"]))
        return (preds,{"x":x}) if cache else preds
    if model["kind"]!="rnn":
        raise ValueError("unknown student architecture")
    activations=[[0.0]*HIDDEN]
    for token in tokens:
        prev=activations[-1]
        ht=[tanh(_dot(model["wx"][i],token)+
                 _dot(model["wh"][i],prev)+model["bh"][i])
            for i in range(HIDDEN)]
        activations.append(ht)
    last=activations[-1]
    preds=tuple(sigmoid(_dot(w,last)+b)
                for w,b in zip(model["wo"],model["bo"]))
    return (preds,{"tokens":tokens,"states":activations}) if cache else preds


def _zeros(model):
    return {name:([[0.0 for _ in row] for row in values]
                  if isinstance(values[0],list) else
                  [0.0 for _ in values])
            for name,values in model.items() if name!="kind"}


def _weight(example):
    """Precommitted weight gives more attention to informative teacher states."""
    a,b=map(float,example["q"])
    return 1.0 + 2.5*max(a,b) + 7.0*abs(a-b)


def loss(model,examples):
    if not examples:
        raise ValueError("nonempty data required")
    result=0.0
    for ex in examples:
        p=forward(model,ex["tokens"])
        target=tuple(map(float,ex["q"]))
        result+=_weight(ex)*sum((p[i]-target[i])**2 for i in range(2))/2
    return result/len(examples)


def gradients(model,example):
    """Hand-derived BPTT gradient for the complete two-cell RNN."""
    predictions,cache=forward(model,example["tokens"],True)
    target=tuple(map(float,example["q"]))
    scale=_weight(example)
    dz=[scale*(p-y)*p*(1-p) for p,y in zip(predictions,target)]
    g=_zeros(model)
    if model["kind"]=="linear":
        x=cache["x"]
        for k in range(OUTPUTS):
            g["b"][k]+=dz[k]
            for j in range(len(x)):
                g["w"][k][j]+=dz[k]*x[j]
        return g
    states=cache["states"]
    tokens=cache["tokens"]
    h=states[-1]
    for k in range(OUTPUTS):
        g["bo"][k]+=dz[k]
        for j in range(HIDDEN):
            g["wo"][k][j]+=dz[k]*h[j]
    back=[sum(model["wo"][k][j]*dz[k] for k in range(OUTPUTS))
          for j in range(HIDDEN)]
    for step in (1,0):
        cur=states[step+1]
        prev=states[step]
        x=tokens[step]
        da=[back[j]*(1-cur[j]*cur[j]) for j in range(HIDDEN)]
        for j in range(HIDDEN):
            g["bh"][j]+=da[j]
            for k in range(INPUTS):
                g["wx"][j][k]+=da[j]*x[k]
            for k in range(HIDDEN):
                g["wh"][j][k]+=da[j]*prev[k]
        back=[sum(model["wh"][j][k]*da[j]
                  for j in range(HIDDEN)) for k in range(HIDDEN)]
    return g


def _iterate_leaves(model):
    for key,val in model.items():
        if key=="kind":continue
        if isinstance(val[0],list):
            for i,row in enumerate(val):
                for j,v in enumerate(row):
                    yield key,i,j,v
        else:
            for j,v in enumerate(val):
                yield key,None,j,v


def _set_leaf(model,key,i,j,value):
    if i is None:model[key][j]=value
    else:model[key][i][j]=value


def gradient_norm(g):
    return sqrt(sum(v*v for _,_,_,v in _iterate_leaves({"kind":"grad",**g})))


def fit(model,train,validation,seed=STUDENT_SEED,
        epochs=EPOCHS,rate=LEARNING_RATE):
    """Deterministic per-example Adam with validation-only checkpointing."""
    if not train or not validation or type(epochs) is not int or epochs<1:
        raise ValueError("nonempty train and validation, positive epochs required")
    model=deepcopy(model)
    first=_zeros(model)
    m=deepcopy(first)
    v=deepcopy(first)
    step=0
    rng=Random(seed)
    history=[]
    best_val=float("inf")
    best=None
    best_epoch=None
    for epoch in range(epochs):
        indices=list(range(len(train)))
        rng.shuffle(indices)
        for ix in indices:
            g=gradients(model,train[ix])
            norm=gradient_norm(g)
            clip=min(1.0,CLIP_NORM/max(norm,1e-30))
            step+=1
            for key,i,j,old in _iterate_leaves(model):
                gval=(g[key][j] if i is None else g[key][i][j])*clip
                lastm=m[key][j] if i is None else m[key][i][j]
                lastv=v[key][j] if i is None else v[key][i][j]
                nowm=0.9*lastm+0.1*gval
                nowv=0.999*lastv+0.001*gval*gval
                _set_leaf(m,key,i,j,nowm)
                _set_leaf(v,key,i,j,nowv)
                correctedm=nowm/(1-0.9**step)
                correctedv=nowv/(1-0.999**step)
                _set_leaf(model,key,i,j,
                          old-rate*correctedm/(sqrt(correctedv)+1e-8))
        tr=loss(model,train)
        va=loss(model,validation)
        history.append((epoch+1,tr,va))
        if va<best_val:
            best_val=va
            best=deepcopy(model)
            best_epoch=epoch+1
    return {
        "weights":best,
        "best_epoch":best_epoch,
        "best_validation_loss":best_val,
        "final_train_loss":history[-1][1],
        "final_validation_loss":history[-1][2],
        "history":history,
        "update_steps":step,
        "parameters":trainable_count(model),
    }


def evaluate_supervised(model,data):
    """Labels are only used after training for test metrics."""
    if not data:raise ValueError("evaluation requires examples")
    error=0.0
    regret=0.0
    disagree=0
    informative=0
    for ex in data:
        pred=forward(model,ex["tokens"])
        q=tuple(map(float,ex["q"]))
        error+=sum(abs(pred[i]-q[i]) for i in range(2))/2
        pick=0 if pred[0]>=pred[1] else 1
        regret+=max(q)-q[pick]
        if max(q)-min(q)>1e-12:
            informative+=1
            disagree+=int(pick!=ex["teacher_action"])
    return {
        "examples":len(data),
        "informative":informative,
        "mean_absolute_action_value_error":error/len(data),
        "mean_teacher_action_regret":regret/len(data),
        "informative_action_disagreements":disagree,
    }


def inference_multiplies(model):
    if model["kind"]=="linear":
        return OUTPUTS*INPUTS*2
    if model["kind"]=="rnn":
        return 2*HIDDEN*(INPUTS+HIDDEN)+OUTPUTS*HIDDEN
    raise ValueError("unknown architecture")


def model_receipt(model):
    """Round-trip stable JSON-native trained parameter dictionaries."""
    return deepcopy(model)


def from_receipt(serialized):
    if serialized.get("kind") not in ("linear","rnn"):
        raise ValueError("unrecognized trained artifact")
    model=deepcopy(serialized)
    expected=create_linear(0) if model["kind"]=="linear" else create_rnn(0)
    if set(model)!=set(expected) or any(
        (len(model[k])!=len(expected[k]) or
         (isinstance(expected[k][0],list) and
          any(len(a)!=len(b) for a,b in zip(model[k],expected[k]))))
        for k in expected if k!="kind"):
        raise ValueError("unexpected model shape")
    return model
