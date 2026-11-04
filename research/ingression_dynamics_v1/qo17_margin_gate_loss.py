"""QO-17: full-BPTT margin and engagement loss on unchanged 26/138 parameters.

No additional trainable gates or recurrent units. Source-fixed objective:
    L = legacy weighted Q-value MSE
      + 2 |y0-y1| softplus(-sign(y0-y1)(p0-p1)/0.1)
      + .015 BCE(t, sigmoid((max(p)-.05)/.025))
where t is the exact teacher engagement decision. The original QO-14
per-example gradient and fit implement the MSE-only ablation unchanged.
"""
from copy import deepcopy
from math import exp, log1p, sqrt
from random import Random

from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    sigmoid, forward, gradients, fit, _weight, _zeros, _iterate_leaves,
    _set_leaf, gradient_norm, create_linear, create_rnn, trainable_count,
    EPOCHS, LEARNING_RATE, CLIP_NORM, STUDENT_SEED, HIDDEN, INPUTS, OUTPUTS,
)

LAMBDA_RANK=2.0
RANK_TEMPERATURE=0.10
LAMBDA_GATE=0.015
GATE_TEMPERATURE=0.025
ENGAGEMENT_GATE=0.05

def softplus(z):
    """Stable log(1 + exp(z))."""
    return max(0.0,z)+log1p(exp(-abs(z)))

def extra_terms(pred, ex):
    """Return added loss and output-gradient dL/dp, before sigmoid."""
    y=tuple(float(v) for v in ex["q"])
    margin=y[0]-y[1]
    d=[0.0,0.0]
    extra=0.0
    if margin:
        sign=1 if margin>0 else -1
        z=-sign*(pred[0]-pred[1])/RANK_TEMPERATURE
        extra+=LAMBDA_RANK*abs(margin)*softplus(z)
        derivative=-LAMBDA_RANK*abs(margin)*sign*sigmoid(z)/RANK_TEMPERATURE
        d[0]+=derivative
        d[1]-=derivative
    maxi=0 if pred[0]>=pred[1] else 1
    z=(pred[maxi]-ENGAGEMENT_GATE)/GATE_TEMPERATURE
    target=1 if max(y)>=ENGAGEMENT_GATE else 0
    extra+=LAMBDA_GATE*(softplus(z)-target*z)
    d[maxi]+=LAMBDA_GATE*(sigmoid(z)-target)/GATE_TEMPERATURE
    return extra,d

def augmented_loss_one(model,ex):
    p=forward(model,ex["tokens"])
    y=tuple(float(v) for v in ex["q"])
    base=_weight(ex)*sum((p[i]-y[i])**2 for i in range(2))/2
    extra,_=extra_terms(p,ex)
    return base+extra

def augmented_loss(model,examples):
    if not examples:
        raise ValueError("nonempty validation or train required")
    return sum(augmented_loss_one(model,x) for x in examples)/len(examples)

def _output_gradient(model,tokens,d_dp):
    """Exact chain-rule BPTT for arbitrary dL/dpred (two sigmoid outputs)."""
    preds,cache=forward(model,tokens,True)
    dz=[d_dp[k]*preds[k]*(1-preds[k]) for k in range(OUTPUTS)]
    g=_zeros(model)
    if model["kind"]=="linear":
        x=cache["x"]
        for k in range(OUTPUTS):
            g["b"][k]+=dz[k]
            for j in range(len(x)):
                g["w"][k][j]+=dz[k]*x[j]
        return g
    if model["kind"]!="rnn":
        raise ValueError("invalid model architecture")
    states=cache["states"]
    for k in range(OUTPUTS):
        g["bo"][k]+=dz[k]
        for j in range(HIDDEN):
            g["wo"][k][j]+=dz[k]*states[-1][j]
    back=[sum(model["wo"][k][j]*dz[k] for k in range(OUTPUTS))
          for j in range(HIDDEN)]
    for step in (1,0):
        cur,prev=states[step+1],states[step]
        token=tokens[step]
        da=[back[j]*(1-cur[j]*cur[j]) for j in range(HIDDEN)]
        for j in range(HIDDEN):
            g["bh"][j]+=da[j]
            for k in range(INPUTS):
                g["wx"][j][k]+=da[j]*token[k]
            for k in range(HIDDEN):
                g["wh"][j][k]+=da[j]*prev[k]
        back=[sum(model["wh"][j][k]*da[j] for j in range(HIDDEN))
              for k in range(HIDDEN)]
    return g

def augmented_gradients(model,ex):
    pred=forward(model,ex["tokens"])
    _,extra=extra_terms(pred,ex)
    g=gradients(model,ex)
    add=_output_gradient(model,ex["tokens"],extra)
    for key,i,j,v in _iterate_leaves(model):
        if i is None:
            g[key][j]+=add[key][j]
        else:
            g[key][i][j]+=add[key][i][j]
    return g

def fit_augmented(model,train,validation,seed=STUDENT_SEED,
                  epochs=EPOCHS,rate=LEARNING_RATE):
    """Source-fixed Adam, validation-only epoch selection, full BPTT."""
    if not train or not validation or type(epochs) is not int or epochs<1:
        raise ValueError("invalid training examples or epochs")
    model=deepcopy(model)
    m=_zeros(model)
    v=_zeros(model)
    rng=Random(seed)
    step=0
    best=None
    best_val=float("inf")
    best_epoch=0
    history=[]
    for epoch in range(epochs):
        order=list(range(len(train)))
        rng.shuffle(order)
        for ix in order:
            g=augmented_gradients(model,train[ix])
            clip=min(1.0,CLIP_NORM/max(gradient_norm(g),1e-30))
            step+=1
            for key,i,j,old in _iterate_leaves(model):
                gv=(g[key][j] if i is None else g[key][i][j])*clip
                last_m=m[key][j] if i is None else m[key][i][j]
                last_v=v[key][j] if i is None else v[key][i][j]
                nm=0.9*last_m+0.1*gv
                nv=0.999*last_v+0.001*gv*gv
                _set_leaf(m,key,i,j,nm)
                _set_leaf(v,key,i,j,nv)
                corrected=nm/(1-0.9**step)
                variance=nv/(1-0.999**step)
                _set_leaf(model,key,i,j,
                          old-rate*corrected/(sqrt(variance)+1e-8))
        tr=augmented_loss(model,train)
        va=augmented_loss(model,validation)
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
        "loss":"QO-14 weighted MSE + fixed rank softplus + fixed gate BCE",
    }
