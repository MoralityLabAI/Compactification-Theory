"""QO-14: CPU-only tiny reservoir RNN, trained to QO-13 Bellman Q-values.

A fixed random tanh recurrence scans the TWO observed per-cell QO-07
evidence tokens. Only the output projection is learned, by ridge least
squares. This is a genuine small recurrent representation with a
trained readout, NOT full backpropagation-through-time or a learned
recurrent transition. The flat linear ridge control gets identical
features and targets; both train on exactly the same source split.

No hidden true rate is an input. Predicting Q-values does not change,
weaken, or replace the QO-07 statistical certificate.
"""
import json
from fractions import Fraction as Q
import numpy as np

from research.ingression_dynamics_v1.recurrent_teacher_data import (
    INPUT_WIDTH, audit_fixed_split
)

HIDDEN=24
INIT_SEED=20261104
RIDGE=Q(1,1000)
FLAT_RIDGE=Q(1,100)
VALUE_THRESHOLD=0.07


def _arrays(rows):
    x=np.asarray([r["tokens"] for r in rows],dtype=np.float64)
    y=np.asarray([[float(v) for v in r["q"]] for r in rows],
                 dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(2,INPUT_WIDTH) or y.shape!=(len(rows),2):
        raise ValueError("unexpected QO-14 evidence and teacher target dimensions")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite input/teacher targets")
    if np.min(y)<0 or np.max(y)>1:
        raise ValueError("teacher action values outside [0,1]")
    return x,y


def _reservoir_weights(hidden=HIDDEN,seed=INIT_SEED):
    if type(hidden) is not int or hidden<2:
        raise ValueError("reservoir hidden size >=2 required")
    rng=np.random.default_rng(int(seed))
    wx=rng.normal(0,0.60/np.sqrt(INPUT_WIDTH),(hidden,INPUT_WIDTH))
    wh=rng.normal(0,0.60/np.sqrt(hidden),(hidden,hidden))
    bias=rng.normal(0,0.10,(hidden,))
    # Fixed, documented contraction scale. Learning changes only readout.
    return wx,wh,bias


def recurrent_embedding(x, wx, wh, bias):
    if x.ndim!=3 or x.shape[1:]!=(2,INPUT_WIDTH):
        raise ValueError("two public cell tokens required")
    if wh.shape!=(len(bias),len(bias)) or wx.shape!=(len(bias),INPUT_WIDTH):
        raise ValueError("inconsistent recurrence shapes")
    h=np.zeros((x.shape[0],len(bias)),dtype=np.float64)
    for i in range(2):
        h=np.tanh(x[:,i,:]@wx.T + h@wh.T + bias)
    return h


def _ridge_fit(features, target, sample_weights, reg):
    features=np.asarray(features,dtype=np.float64)
    target=np.asarray(target,dtype=np.float64)
    weights=np.asarray(sample_weights,dtype=np.float64)
    if np.any(weights<=0) or features.shape[0]!=len(weights):
        raise ValueError("positive aligned sample weights required")
    augmented=np.concatenate((features,np.ones((len(features),1))),axis=1)
    root=np.sqrt(weights)[:,None]
    matrix=(augmented*root).T@(augmented*root)
    diag=np.eye(augmented.shape[1],dtype=np.float64)*float(reg)
    diag[-1,-1]=0.0
    matrix+=diag
    rhs=(augmented*root).T@(target*root)
    return np.linalg.solve(matrix,rhs)


def train_students(training_rows, hidden=HIDDEN, seed=INIT_SEED):
    x,y=_arrays(training_rows)
    # Reweight positive/decisive states to avoid a degenerate all-zero
    # predictor when most Bellman queries are uninformative.
    weights=np.asarray([
        0.30 + 3.0*int(r["teacher_value"]>0 and not r["certified"])
        +2.0*int(r["decisive"])
        for r in training_rows],dtype=np.float64)
    wx,wh,bias=_reservoir_weights(hidden,seed)
    hidden_state=recurrent_embedding(x,wx,wh,bias)
    output=_ridge_fit(hidden_state,y,weights,RIDGE)
    baseline=_ridge_fit(x.reshape((len(x),-1)),y,weights,FLAT_RIDGE)
    checkpoint={
        "contract":"QO-14 trainable linear readout over fixed tanh recurrent reservoir",
        "init_seed":int(seed),
        "input_width":INPUT_WIDTH,"hidden_size":hidden,
        "ridge_regularization":str(RIDGE),
        "flat_regularization":str(FLAT_RIDGE),
        "training_examples":len(training_rows),
        "train_rate_pairs":audit_fixed_split()["train_rate_pairs"],
        "reservoir": {
            "wx":wx.tolist(),"wh":wh.tolist(),"bias":bias.tolist(),
            "readout":output.tolist(),
        },
        "flat_linear":baseline.tolist(),
        "estimated_bytes_float32":{
            "recurrent":int(4*(wx.size+wh.size+bias.size+output.size)),
            "linear":int(4*baseline.size),
        },
        "recurrent_macs_per_inference_estimate":int(
            2*(hidden*INPUT_WIDTH+hidden*hidden)+hidden*2),
        "linear_macs_per_inference_estimate":int(2*INPUT_WIDTH*2),
    }
    return checkpoint


def predict_q(checkpoint,tokens,model="recurrent"):
    x=np.asarray(tokens,dtype=np.float64).reshape((1,2,INPUT_WIDTH))
    if model=="recurrent":
        data=checkpoint["reservoir"]
        h=recurrent_embedding(
            x,np.asarray(data["wx"],dtype=np.float64),
            np.asarray(data["wh"],dtype=np.float64),
            np.asarray(data["bias"],dtype=np.float64))
        features=np.concatenate((h,np.ones((1,1))),axis=1)
        output=features@np.asarray(data["readout"],dtype=np.float64)
    elif model=="linear":
        features=np.concatenate((x.reshape((1,-1)),np.ones((1,1))),axis=1)
        output=features@np.asarray(checkpoint["flat_linear"],dtype=np.float64)
    else:
        raise ValueError("model must be recurrent or linear")
    return tuple(float(v) for v in np.clip(output[0],0,1))


def evaluate_student_on_teacher(rows, checkpoint, model):
    """Grounded held-out Q-value regression and teacher action regret.

    Decision-sensitive accuracy is scored ONLY when q0 != q1, to
    avoid inflating agreement by arbitrary ties. Teacher regret is
    calculated against the *true exact teacher Q values*, not an
    observed action reward.
    """
    if model not in ("linear","recurrent"):
        raise ValueError("unknown student")
    n=len(rows)
    if n==0:raise ValueError("nonempty holdout required")
    squared=0.
    value_squared=0.
    regret=0.
    sensitive=0
    correct=0
    positive=0
    positive_mse=0.
    selected={0:0,1:0}
    for r in rows:
        a,b=predict_q(checkpoint,r["tokens"],model)
        q=[float(x) for x in r["q"]]
        squared+=(a-q[0])**2+(b-q[1])**2
        value_squared+=(max(a,b)-float(r["teacher_value"]))**2
        chosen=int(b>a)
        selected[chosen]+=1
        regret+=max(q)-q[chosen]
        if r["decisive"]:
            sensitive+=1
            correct+=int(chosen==r["teacher_action"])
        if r["teacher_value"]>0:
            positive+=1
            positive_mse+=(max(a,b)-float(r["teacher_value"]))**2
    return {
        "model":model,"examples":n,
        "teacher_positive_examples":positive,
        "teacher_decision_sensitive":sensitive,
        "q_rmse":float(np.sqrt(squared/(2*n))),
        "value_rmse":float(np.sqrt(value_squared/n)),
        "positive_value_rmse":float(np.sqrt(positive_mse/positive))
                              if positive else None,
        "mean_teacher_action_regret":regret/n,
        "sensitive_action_accuracy":correct/sensitive if sensitive else None,
        "sensitive_action_correct":correct,
        "action_counts":[selected[0],selected[1]],
        "trained_parameters": checkpoint["estimated_bytes_float32"][model]//4,
        "inference_macs_per_state_estimate":(
            checkpoint["recurrent_macs_per_inference_estimate"]
            if model=="recurrent" else checkpoint["linear_macs_per_inference_estimate"]),
    }
