"""Original MathCan source, paired advertisements, independent native certificates."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

PIN="6d02b611c7358216379cb36952512a104e0bdb8e"
ALLOWED={"propext","Classical.choice","Quot.sound"}
# module, declaration, good advertisement, proof term, false advertisement, proof of its negation.
PAIRS=[
("RHWindowExponents","group_log_floor_of_depth_and_k_window",
 "∀ L n m k : Nat, L + m ≤ k → k ≤ 2*n + m → L ≤ 2*n",
 "by intro L n m k h1 h2; exact group_log_floor_of_depth_and_k_window L n m k h1 h2",
 "∀ L n m k : Nat, L + m ≤ k → L ≤ 2*n",
 "by intro h; exact (by decide : ¬ ((1 : Nat) ≤ 2*0)) (h 1 0 0 1 (by decide))"),
("RHWindowExponents","four_window_error_terms",
 "∀ Q a b c d : Nat, a ≤ 2*Q → b ≤ 2*Q → c ≤ 4*Q → d ≤ 2*Q → a+b+c+d ≤ 10*Q",
 "by intro Q a b c d h1 h2 h3 h4; exact four_window_error_terms Q a b c d h1 h2 h3 h4",
 "∀ Q a b c d : Nat, a ≤ 2*Q → b ≤ 2*Q → d ≤ 2*Q → a+b+c+d ≤ 10*Q",
 "by intro h; exact (by decide : ¬ ((0 : Nat)+0+1+0 ≤ 10*0)) (h 0 0 0 1 0 (by decide) (by decide) (by decide))"),
("RHWindowExponents","normalized_character_error_26",
 "∀ G X Z : Nat, X ≤ Z → G ≤ 4*X+22*Z → G ≤ 26*Z",
 "by intro G X Z hx hg; exact normalized_character_error_26 G X Z hx hg",
 "∀ G X Z : Nat, G ≤ 4*X+22*Z → G ≤ 26*Z",
 "by intro h; exact (by decide : ¬ ((4 : Nat) ≤ 26*0)) (h 4 1 0 (by decide))"),
("RHMainLogic","row_proof_reused",
 "∀ Row RecTExists : Prop, Row → RecTExists → OpenLiteratureProxy Row RecTExists",
 "by intro R C hr hc; exact row_proof_reused hr hc",
 "∀ Row RecTExists : Prop, Row → OpenLiteratureProxy Row RecTExists",
 "by intro h; exact (h True False True.intro).localLanglandsExists"),
("RHMainLogic","final_rh_route_from_supplied_mathematical_bridges",
 "∀ Row RecTExists VWin EveryCharacter RH : Prop, Row → RecTExists → (Row → VWin) → (OpenLiteratureProxy Row RecTExists → VWin → EveryCharacter) → (EveryCharacter → RH) → RH",
 "by intro R C V E H hr hc hw he hh; exact final_rh_route_from_supplied_mathematical_bridges hr hc hw he hh",
 "∀ Row RecTExists VWin EveryCharacter RH : Prop, Row → RecTExists → (Row → VWin) → (OpenLiteratureProxy Row RecTExists → VWin → EveryCharacter) → RH",
 "by intro h; exact h True True True True False True.intro True.intro (fun _ => True.intro) (fun _ _ => True.intro)"),
("RHMainLogic","independent_theorem_with_literature_parameter",
 "∀ P : Prop, P → P",
 "by intro P hp; exact independent_theorem_with_literature_parameter hp (row_proof_reused True.intro True.intro)",
 "∀ P : Prop, P",
 "by intro h; exact h False"),
("FiniteHorizonGap","full_horizon_recovers",
 "∀ (xs ys : List Bool) (T : Nat), xs.length ≤ T → ys.length ≤ T → prefixEquivalent T xs ys → xs = ys",
 "by intro xs ys T hx hy he; exact full_horizon_recovers xs ys T hx hy he",
 "∀ (xs ys : List Bool) (T : Nat), xs.length ≤ T → prefixEquivalent T xs ys → xs = ys",
 "by intro h; exact (by decide : ([] : List Bool) ≠ [true]) (h [] [true] 0 (by decide) (by rfl))"),
("FiniteHorizonGap","epsilon_sixth_clearance",
 "∀ d e : Nat, 6*e ≤ d → d+2*e ≤ 2*(d-2*e)",
 "by intro d e he; exact epsilon_sixth_clearance d e he",
 "∀ d e : Nat, d+2*e ≤ 2*(d-2*e)",
 "by intro h; exact (by decide : ¬ ((0 : Nat)+2*1 ≤ 2*(0-2*1))) (h 0 1)"),
("NOR19StreamedVerifier","streamed_strictly_fewer_materialized_slots",
 "∀ n : Nat, 0 < n → publicWorkspaceSlots n 4 0 < publicWorkspaceSlots n 4 4",
 "by intro n hn; exact streamed_strictly_fewer_materialized_slots n hn",
 "∀ n : Nat, publicWorkspaceSlots n 4 0 < publicWorkspaceSlots n 4 4",
 "by intro h; exact (by decide : ¬ (publicWorkspaceSlots 0 4 0 < publicWorkspaceSlots 0 4 4)) (h 0)"),
("NOR19StreamedVerifier","streamed_exactly_iff_materialized_equal",
 "∀ a b : List Bool, streamingEquals a b = true ↔ a = b",
 "by intro a b; exact streamed_exactly_iff_materialized_equal a b",
 "∀ a b : List Bool, a.length = b.length → streamingEquals a b = true",
 "by intro h; exact (by decide : streamingEquals [true] [false] ≠ true) (h [true] [false] rfl)"),
("PhaseLatch","answer_dominant_nearer_future",
 "∀ x y : Int × Int, (x.1-y.1)*(x.1-y.1) < (y.2-x.2)*(y.2-x.2) → squaredDistance (x.1,y.2) y < squaredDistance (x.1,y.2) x",
 "by intro x y h; exact answer_dominant_nearer_future x y h",
 "∀ x y : Int × Int, squaredDistance (x.1,y.2) y < squaredDistance (x.1,y.2) x",
 "by intro h; exact (by decide : ¬ (squaredDistance (0,0) (0,0) < squaredDistance (0,0) (0,0))) (h (0,0) (0,0))"),
("AuthorizationQuotient","toy_no_public_classifier",
 "∀ classify : Unit → Bool, classify (toyObservation .granted) ≠ toyAuthorization .granted ∨ classify (toyObservation .revoked) ≠ toyAuthorization .revoked",
 "by intro c; exact toy_no_public_classifier c",
 "∀ classify : Unit → Bool, classify (toyObservation .granted) ≠ toyAuthorization .granted ∧ classify (toyObservation .revoked) ≠ toyAuthorization .revoked",
 "by intro h; exact (h (fun _ => true)).1 rfl"),
]

def sha(x):return hashlib.sha256(x).hexdigest()
def run(cmd,cwd,env,log):
    p=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=240)
    s=p.stdout+p.stderr
    log.write_text(s)
    assert p.returncode==0 and not re.search(r"error:|sorryAx|uses 'sorry'",s),s[-12000:]
    return s

def parse_axioms(s):
    result={}
    for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",s):
        assert m[1] not in result
        result[m[1]]={v.strip() for v in (m[3] or "").split(",") if v.strip()}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    a=p.parse_args()
    source=a.source.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()==PIN
    assert (source/"lean-toolchain").read_text().strip()=="leanprover/lean4:v4.35.0"
    assert len(PAIRS)==12 and len({(m,d) for m,d,*_ in PAIRS})==12
    cache=out/"olean";(cache/"MathCan").mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,LEAN_PATH=str(cache))
    modules={}
    for m in sorted({x[0] for x in PAIRS}):
        rel=Path("MathCan")/(m+".lean")
        raw=(source/rel).read_bytes()
        assert raw.decode().splitlines()[0]=="import Init"
        modules[m]=raw
        run([a.lean,"-o",str(cache/rel.with_suffix(".olean")),str(rel)],source,env,out/(m+"-source.log"))
    prompts=[];gold=[];receipts=[]
    for m,d,good,proof,bad,neg in PAIRS:
        module="MathCan."+m
        raw=modules[m]
        pair=sha((PIN+module+d).encode())[:16]
        for supported,target,body in [(True,good,proof),(False,bad,neg)]:
            cid=sha((pair+target).encode())[:20]
            checked=target if supported else "¬ ("+target+")"
            lean="\n".join(["import "+module,"open "+module,"namespace CG05Audit",
                           "theorem cert : "+checked+" := "+body,
                           "#print axioms "+module+"."+d,
                           "#print axioms cert","end CG05Audit"])+"\n"
            path=out/(cid+".lean");path.write_text(lean)
            log=run([a.lean,path.name],out,env,path.with_suffix(".log"))
            ax=parse_axioms(log)
            assert set(ax)=={module+"."+d,"CG05Audit.cert"},ax
            assert all(not (xs-ALLOWED) for xs in ax.values()),ax
            inp={"id":cid,"source_sha256":sha(raw),"module":module,
                 "source":raw.decode(),"selected":module+"."+d,"advertised":target}
            prompts.append(inp)
            gold.append({"id":cid,"pair":pair,"split":"dev",
                         "status":"SUPPORTED" if supported else "REFUTED",
                         "checked_proposition":checked,"proof":body})
            receipts.append({"id":cid,"pair":pair,"status":gold[-1]["status"],
                             "source_sha256":sha(raw),"certificate_sha256":sha(lean.encode()),
                             "axioms":{k:sorted(v) for k,v in ax.items()}})
    assert len(prompts)==len(gold)==len(receipts)==24
    assert len({x["id"] for x in prompts})==24
    for name,rows in [("inputs",prompts),("gold",gold),("receipts",receipts)]:
        (out/(name+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in sorted(rows,key=lambda x:x["id"])))
    print("NATIVE",len(modules),"original modules",len(PAIRS),"pairs",len(receipts),"certificates",2*len(receipts),"axiom checks",flush=True)

if __name__=="__main__":main()
