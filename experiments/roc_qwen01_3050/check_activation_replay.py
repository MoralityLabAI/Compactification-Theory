"""Two new zero-hook controls check cached activation reproducibility."""
import json
import numpy as np
import preflight


def main():
    original=preflight.RESULTS
    replay=original/"replay"
    replay.mkdir(exist_ok=True)
    # Keep the campaign's hardware/memory/cache receipts intact.
    preflight.RESULTS=replay
    from smoke import torch,load
    from runtime import Runtime
    from core import LAYERS
    model,tokenizer,manifest,_=load()
    runtime=Runtime(torch,model,manifest)
    rows=json.loads((original/"dataset.json").read_text())
    acts=np.load(original/"activations.npy",mmap_mode="r")
    margins=np.load(original/"margins.npy",mmap_mode="r")
    output=[]
    for rule in ("copy","flip"):
        i=next(i for i,r in enumerate(rows) if r["split"]=="heldout" and r["rule"]==rule)
        receipt,vectors=runtime.forward(rows[i],delta=np.zeros(1024),layer=12,capture=LAYERS)
        errors=[float(np.max(np.abs(vectors[layer].astype(float)-acts[i,j].astype(float))))
                for j,layer in enumerate(LAYERS)]
        exact=all(np.array_equal(vectors[layer],acts[i,j]) for j,layer in enumerate(LAYERS))
        margin_error=abs(receipt["margin_a_minus_b"]-float(margins[i]))
        output.append(dict(state_id=rows[i]["state_id"],layers=list(LAYERS),
            per_layer_max_absolute_error=errors,bitwise_equal=exact,margin_absolute_error=margin_error,
            actual_zero_hook_l2=receipt["actual_l2"],forward_key=receipt["key"],
            passed=max(errors)<=0.002 and margin_error<=0.03125 and receipt["actual_l2"]==0))
    runtime.close()
    result=dict(status="passed" if all(x["passed"] for x in output) else "failed",states=output,
                hidden_absolute_tolerance=0.002,margin_absolute_tolerance=0.03125,
                source_hash=runtime.source_hash,script_sha256=preflight.sha(preflight.ROOT/"check_activation_replay.py"),
                condition="new_zero_hook_identity_control; different setting from extraction",
                successful_forward_passes=runtime.successes)
    preflight.save(replay/"activation_replay.json",result)
    print(json.dumps(result,indent=2))
    if result["status"]!="passed":
        raise RuntimeError("Activation identity replay failed declared tolerances")


if __name__=="__main__":
    main()
