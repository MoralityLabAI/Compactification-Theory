"""SMC-26: old-source-locked new independent teacher training and calibration.

Trains fresh B811/B919 (600 source-native updates) and one Top4 SAE71 per
teacher, with original SMC-13 train/validation only. A separate fixed
temperature is fitted from the OLD 64 SMC13 calibration puzzles. No SMC26
novel puzzles exist during this stage.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc13_train_teachers import (
    train_teacher,sae_fit,stable_hash_tensors,source_sae_fingerprint,tensors
)
from smc07_real_sae import TopKSAE
from upstream_arch import TinyRecursiveModel
from smc13_family import fingerprint
from smc14_causal_transfer import goal_positions
from smc25_candidate_effects import (
    load_source,clean_native,TEMPERATURES
)

OLD_TEACHERS=("B501","B701")
NEW_TEACHERS=("B811","B919")
TEACHERS=OLD_TEACHERS+NEW_TEACHERS
SAE_SEED=71
NEW_SEEDS=(811,919)
SCHEMA="SMC26/original-source-trained-four-teachers-v1"
TRAIN_STEPS=600
SAE_STEPS=300


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def protocol_sha():
    return digest((HERE/"SMC26_PROTOCOL.md").read_bytes())


def train_and_calibrate(original,independent,conditioned,allow_fetch,bundle_path):
    torch.set_num_threads(2)
    teachers,comps,old,snapshot=load_source(original,independent,conditioned,allow_fetch)
    calibr=old["feature_calibration"]
    x,y=tensors(calibr)
    pos=goal_positions(x)
    train_x,_=tensors(old["teacher_train"])
    val_x,_=tensors(old["teacher_val"])
    new_bundle={}
    new_meta={}
    for seed in NEW_SEEDS:
        teacher,details=train_teacher(old["teacher_train"],old["teacher_val"],seed,steps=TRAIN_STEPS)
        with torch.no_grad():
            hidden_train=teacher.reason_states(train_x)[1].detach()
            hidden_val=teacher.reason_states(val_x)[1].detach()
        sae,mean,scale,fit=sae_fit(hidden_train,hidden_val,seed=SAE_SEED)
        name=f"B{seed}"
        teacher_fp=stable_hash_tensors(teacher.state_dict())
        sae_fp=source_sae_fingerprint(sae,mean,scale)
        new_bundle[name]={
            "teacher_state":{k:v.cpu().detach().clone() for k,v in teacher.state_dict().items()},
            "sae_state":{k:v.cpu().detach().clone() for k,v in sae.state_dict().items()},
            "sae_mean":mean.cpu().detach().clone(),
            "sae_scale":scale.cpu().detach().clone(),
        }
        new_meta[name]={
            "seed":seed,"teacher_optimizer_updates":TRAIN_STEPS,
            "sae_optimizer_updates":SAE_STEPS,
            "sae_seed":SAE_SEED,
            "teacher_sha256":teacher_fp,"sae_sha256":sae_fp,
            "teacher_training":details,
            "sae_validation_only_best_step":fit["best_step"],
            "sae_validation_only_best_mse":fit["best_val_mse"],
        }
        teachers[name]=teacher
        comps[f"{name}_SAE71"]=(name,sae,mean,scale)
        print("SMC26_NEW_TRAINED",name,"best",details["selected_step"],
              "teacher_sha",teacher_fp,"sae_sha",sae_fp,flush=True)
    bundle_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(new_bundle,bundle_path)
    bundle_hash=digest(bundle_path.read_bytes())
    # Independently verify the stored new teacher and SAE tensor bytes.
    recovered=torch.load(bundle_path,map_location="cpu",weights_only=True)
    for name,data in recovered.items():
        model=TinyRecursiveModel(48,4)
        model.load_state_dict(data["teacher_state"],strict=True)
        sae=TopKSAE(48,96,4)
        sae.load_state_dict(data["sae_state"],strict=True)
        if stable_hash_tensors(model.state_dict())!=new_meta[name]["teacher_sha256"]:
            raise ValueError("new teacher snapshot fails source canonical SHA")
        if source_sae_fingerprint(sae,data["sae_mean"],data["sae_scale"])!=new_meta[name]["sae_sha256"]:
            raise ValueError("new SAE snapshot fails canonical SHA")
    temps={}
    for name in TEACHERS:
        teacher=teachers[name].eval()
        with torch.no_grad():
            hidden=teacher.reason_states(x)[1].detach()
        native=copy.deepcopy(teacher).double().eval()
        clean=clean_native(native,hidden)
        gather=clean.gather(1,pos.unsqueeze(-1).expand(-1,-1,10))[:,:,1:10]
        truth=y.gather(1,pos)-1
        nll={}
        for T in TEMPERATURES:
            logp=torch.log_softmax(gather/T,dim=-1)
            nll[str(T)]=float(-logp.gather(2,truth.unsqueeze(-1)).mean())
        selected=min(TEMPERATURES,key=lambda t:(nll[str(t)],t))
        correct=int((gather.argmax(-1)==truth).sum())
        temps[name]={"temperature":float(selected),
                     "temperature_nll":nll,
                     "calibration_correct":correct,"n_calibration_goals":512,
                     "calibration_goal_accuracy":correct/512.,
                     "capability_match":bool(.40<=correct/512.<=.60)}
        print("SMC26_CALIBRATED",name,"T",selected,"old_acc",correct/512.,
              "capability",temps[name]["capability_match"],flush=True)
    return {
        "schema":SCHEMA,"protocol_sha256":protocol_sha(),
        "source_snapshot":snapshot,
        "source_train_roster_sha256":fingerprint(old["teacher_train"]),
        "source_val_roster_sha256":fingerprint(old["teacher_val"]),
        "source_calibration_roster_sha256":fingerprint(calibr),
        "bundle_sha256":bundle_hash,"bundle_filename":bundle_path.name,
        "new_teacher_seed_roster":list(NEW_SEEDS),
        "new_teacher_model_SAE_training":new_meta,
        "all_teacher_temperatures":temps,
        "teacher_names":list(TEACHERS),
        "new_puzzles_generated":False,
        "all_four_capability_matched":all(t["capability_match"] for t in temps.values()),
        "torch":torch.__version__,
    }


def load_all_models(original,independent,conditioned,bundle,calibration,fetch_source):
    """Read the OLD frozen sources and the separately archived trained bundle."""
    models,comps,old,snapshot=load_source(original,independent,conditioned,fetch_source)
    saved=json.loads(Path(calibration).read_bytes())
    if saved["schema"]!=SCHEMA or saved["protocol_sha256"]!=protocol_sha() or saved["source_snapshot"]!=snapshot:
        raise ValueError("calibration source/protocol changed after registration")
    if (saved["bundle_filename"]!=Path(bundle).name or
        digest(Path(bundle).read_bytes())!=saved["bundle_sha256"] or
        tuple(saved["teacher_names"])!=TEACHERS):
        raise ValueError("frozen new teacher/SAE bundle changed")
    payload=torch.load(bundle,weights_only=True,map_location="cpu")
    if set(payload)!=set(NEW_TEACHERS):
        raise ValueError("new independently trained teacher population differs")
    for name,state in payload.items():
        teacher=TinyRecursiveModel(48,4).eval()
        teacher.load_state_dict(state["teacher_state"],strict=True)
        sae=TopKSAE(48,96,4).eval()
        sae.load_state_dict(state["sae_state"],strict=True)
        meta=saved["new_teacher_model_SAE_training"][name]
        if (stable_hash_tensors(teacher.state_dict())!=meta["teacher_sha256"] or
            source_sae_fingerprint(sae,state["sae_mean"],state["sae_scale"])!=meta["sae_sha256"]):
            raise ValueError("new frozen teacher/SAE model tensor fingerprint changed")
        models[name]=teacher
        comps[f"{name}_SAE71"]=(name,sae,state["sae_mean"],state["sae_scale"])
    for name in OLD_TEACHERS:
        original_name=f"newB_{name[1:]}_SAE_71"
        if original_name not in comps:
            raise ValueError("old source-frozen SAE 71 missing")
        comps[f"{name}_SAE71"]=comps[original_name]
    return models,comps,old,saved


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-artifact",type=Path,required=True)
    p.add_argument("--independent-artifact",type=Path,required=True)
    p.add_argument("--conditioned-receipt",type=Path,required=True)
    p.add_argument("--fetch-source",action="store_true")
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=train_and_calibrate(a.original_artifact,a.independent_artifact,
                               a.conditioned_receipt,a.fetch_source,a.bundle)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SMC26_TRAIN_CALIBRATION_COMPLETE",report["all_four_capability_matched"],
          report["bundle_sha256"],flush=True)


if __name__=="__main__":
    main()
