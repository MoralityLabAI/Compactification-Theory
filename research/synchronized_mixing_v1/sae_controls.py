"""SMC-06: untrained projection and PCA nulls, same source activations."""
import json
import numpy as np
import torch
from sae_train import SAE,teacher_data,train
from approx_geometry import recover_with_one_swap

def control_suite(seed=101):
    torch.manual_seed(seed)
    x,D,masks=teacher_data()
    signals=torch.tensor(masks)@D
    outputs={}
    # Random untrained neural features, identical 15-channel ReLU readout.
    torch.manual_seed(100000+seed)
    random_model=SAE(x.shape[1],15)
    with torch.no_grad():
        acts=torch.relu(random_model.encoder(signals)).numpy()
    try:
        fit=recover_with_one_swap(acts[:12]);out={'accepted':True,'repairs':len(fit['repairs'])}
    except ValueError as e:
        out={'accepted':False,'reason':str(e)}
    outputs['untrained_relu_15']=out
    # 15-dimensional PCA reconstruction subspace, absolute PC magnitudes
    # as proxy sparse activations. PCA has good MSE, but lacks SAE sparsity.
    arr=x.numpy()
    center=arr.mean(axis=0,keepdims=True)
    _,_,vt=np.linalg.svd(arr-center,full_matrices=False)
    components=vt[:15,:]
    recon=(arr-center)@components.T@components+center
    mse=float(np.mean(np.sum((arr-recon)**2,axis=1)))
    acts=np.abs((signals.numpy()-center)@components.T)
    try:
        fit=recover_with_one_swap(acts[:12]);out={'accepted':True,'repairs':len(fit['repairs'])}
    except ValueError as e:
        out={'accepted':False,'reason':str(e)}
    out['reconstruction_squared_l2']=mse
    outputs['pca_absolute_15']=out
    return {'seed':seed,'controls':outputs,
            'scope':'untrained readout and PCA abs coordinates, no pretrained model'}

if __name__=='__main__':
    print(json.dumps([control_suite(s) for s in (101,102,103)],indent=2))
