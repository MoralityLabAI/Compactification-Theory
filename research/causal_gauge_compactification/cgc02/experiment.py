"""CGC-02 source-locked trained weight-tied causal compactification experiment.

All rank selectors are fitted on development data; fresh roots are fixed in
PROTOCOL.md. The protected current output is the full softmax([h0,h4,0]).
The physical null h0=h4=0 therefore conserves every normalized class exactly.
This is a compact supervised toy, not an SAE/sparse-BPD discovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch import nn

D, WIDTH, XD = 12, 32, 4
KIDX = (1, 2, 3, 5, 6, 7, 8, 9, 10, 11)
RANK, TRAIN_STEPS, FACTOR_STEPS = 2, 650, 360
CAL_SIZE, TEST_SIZE = 256, 512
TRAIN_DEPTHS = (2, 4, 6)
TEST_DEPTHS = (2, 4, 6, 8)
FRESH_ROOTS = (2601, 2607, 2617, 2621)
EXCLUDED_ROOTS = (901, 902)
GRAM_WEIGHTS = {2: .75, 4: 1., 6: .5}


def rng(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def source_batch(seed: int, n: int, family: str = "train"):
    if n < 1 or family not in ("train", "heldout_anticorrelated"):
        raise ValueError("registered families: train and heldout_anticorrelated")
    g = rng(seed)
    z = torch.rand((n, 3), generator=g)*1.6-.8
    a, b, p = z.unbind(-1)
    if family == "heldout_anticorrelated":
        a = torch.rand((n,), generator=g)*1.6-.8
        b = (-.8*a + .2*torch.randn((n,), generator=g)).clamp(-.8, .8)
    role = torch.randint(0, 2, (n,), generator=g).float()*2-1
    h0 = torch.zeros((n, D))
    h0[:, 0], h0[:, 4] = p, -.2*p
    event = torch.stack((a, b, role, torch.ones_like(a)), dim=1)
    label = torch.stack((torch.where(role > 0, a, b),
                         .5*(a+b)+.2*a*b), dim=1)
    return h0, event, label


def matched_source_pairs(seed: int, n: int):
    h0, event, label = source_batch(seed, n, "heldout_anticorrelated")
    g = rng(seed+8001)
    donor = event.clone()
    chosen = torch.where(event[:, 2] > 0, 0, 1)
    direction = torch.where(torch.rand(n, generator=g) < .5, -1., 1.)
    offset = direction * (.30 + .35*torch.rand(n, generator=g))
    row = torch.arange(n)
    donor[row, chosen] = (donor[row, chosen] + offset).clamp(-.8, .8)
    a, b, role = donor[:, 0], donor[:, 1], donor[:, 2]
    donor_label = torch.stack(
        (torch.where(role > 0, a, b), .5*(a+b)+.2*a*b), dim=1)
    if not torch.equal(event[:, 2:], donor[:, 2:]):
        raise AssertionError("role and source-write marker changed")
    if not ((event-donor).abs() > 0).sum(1).eq(1).all():
        raise AssertionError("exactly one private value must change")
    return h0, event, donor, label, donor_label


class RecurrentTeacher(nn.Module):
    """The same physical (A,B,C,b) is used at every shared no-op pass."""
    def __init__(self, seed: int):
        super().__init__()
        g = rng(seed+4401)
        self.A = nn.Parameter(torch.randn(D, WIDTH, generator=g)*.13)
        self.B = nn.Parameter(torch.randn(XD, WIDTH, generator=g)*.13)
        self.bias = nn.Parameter(torch.zeros(WIDTH))
        self.C = nn.Parameter(torch.randn(WIDTH, D, generator=g)*.07)

    def features(self, state, event):
        return torch.tanh(state @ self.A + event @ self.B + self.bias)

    def step(self, state, event):
        return state + self.features(state, event) @ self.C

    def no_op(self, state, depth: int):
        zero = torch.zeros((len(state), XD), dtype=state.dtype, device=state.device)
        for _ in range(depth):
            state = self.step(state, zero)
        return state

    @staticmethod
    def task(state):
        return state[:, -2:]

    @staticmethod
    def protected(state):
        logits = torch.stack((state[:, 0], state[:, 4],
                              torch.zeros_like(state[:, 0])), dim=1)
        return torch.softmax(logits, dim=-1)


def train_teacher(seed: int, updates: int = TRAIN_STEPS):
    torch.set_num_threads(1)
    model = RecurrentTeacher(seed)
    opt = torch.optim.Adam(model.parameters(), lr=.008)
    for step in range(updates):
        h0, event, label = source_batch(seed*100000+step, 128)
        state = model.step(h0, event)
        loss = .025*(state[:, [0, 4]]-h0[:, [0, 4]]).square().mean()
        for k in range(1, 7):
            state = model.step(state, torch.zeros_like(event))
            if k in GRAM_WEIGHTS:
                loss = loss + GRAM_WEIGHTS[k]*(
                    model.task(state)-label).square().mean()
            loss = loss + .025*(state[:, [0, 4]]-h0[:, [0, 4]]).square().mean()
        loss = loss + .0001*(model.A.square().mean()+
                              model.B.square().mean()+model.C.square().mean())
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite teacher fit, seed {seed}, step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval().requires_grad_(False)


def fit_factors(model: RecurrentTeacher, seed: int, restart: int,
                updates: int = FACTOR_STEPS):
    if restart not in (0, 1):
        raise ValueError("two independent registered physical refits")
    g = rng(seed+9205+restart*10337)
    v = nn.Parameter(torch.randn(WIDTH, D, generator=g)*.1)
    u = nn.Parameter(torch.randn(D, D, generator=g)*.1)
    opt = torch.optim.Adam((v, u), lr=.03)
    true = model.C.detach()
    for _ in range(updates):
        loss = (v @ u-true).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return v.detach(), u.detach()


def protected_project(delta):
    out = delta.clone()
    out[:, 0] = 0
    out[:, 4] = 0
    return out


def protected_coordinates():
    return torch.eye(D)[:, list(KIDX)]


def task_jacobian(model: RecurrentTeacher, state, depth: int):
    h = state.detach().requires_grad_(True)
    task = model.task(model.no_op(h, depth))
    gradients = []
    for output in range(2):
        g = torch.autograd.grad(task[:, output].sum(), h,
                                retain_graph=(output == 0))[0]
        gradients.append(g[:, list(KIDX)])
    return torch.stack(gradients, dim=1).detach()


def top_task_basis(gram: torch.Tensor, rank=RANK):
    vals, vec = torch.linalg.eigh(gram.double())
    selected = vec[:, -rank:].float()
    basis = protected_coordinates() @ selected
    gap = float(vals[-rank]-vals[-rank-1]) if rank < len(vals) else None
    return basis, gap, [float(x) for x in vals.flip(0)[:6]]


def physical_weight_basis(weight, rank=RANK):
    _, singular, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    raw = vh[:rank, :].T.float()
    raw[[0, 4], :] = 0
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    if torch.linalg.matrix_rank(raw.double(), tol=1e-9) < rank:
        raise ValueError("protected physical SVD lost rank")
    gap = float(singular[rank-1]-singular[rank])
    return basis, gap


def subspaces(model: RecurrentTeacher, first_weight, second_weight, seed):
    h0, event, _ = source_batch(seed*17+19013, CAL_SIZE, "train")
    with torch.no_grad():
        state = model.step(h0, event)
    grams = {}
    for depth in TRAIN_DEPTHS:
        j = task_jacobian(model, state, depth)
        grams[depth] = torch.einsum("bmi,bmj->ij", j, j)/len(j)
    joint = .75*grams[2]+grams[4]+.5*grams[6]
    task_joint, gap_joint, top_joint = top_task_basis(joint)
    task_4, gap_4, _ = top_task_basis(grams[4])
    weight_bpd, gap_weight = physical_weight_basis(first_weight)
    weight_teacher, gap_teacher = physical_weight_basis(model.C.detach())
    restart_basis, _ = physical_weight_basis(second_weight)
    projector_drift = float(
        (weight_bpd @ weight_bpd.T-restart_basis @ restart_basis.T).norm())
    random = torch.randn((len(KIDX), RANK), generator=rng(seed+989181))
    qr, _ = torch.linalg.qr(random)
    random_basis = protected_coordinates() @ qr
    return {
        "task_joint": task_joint, "task_depth4": task_4,
        "bpd_weight_svd": weight_bpd,
        "teacher_weight_svd": weight_teacher,
        "random_protected_basis": random_basis,
    }, {
        "task_joint_gap": gap_joint, "task_depth4_gap": gap_4,
        "bpd_weight_svd_gap": gap_weight,
        "teacher_weight_svd_gap": gap_teacher,
        "factor_restart_weight_svd_projector_distance": projector_drift,
        "task_joint_leading_eigenvalues": top_joint,
        "calibration_examples": CAL_SIZE,
        "calibration_depths": list(TRAIN_DEPTHS),
    }


def tangent_matched_sham(j: torch.Tensor, selected: torch.Tensor, g: torch.Generator):
    """Same protected L2 norm AND same signed 2-vector future task Jacobian.

    All computations below use double-precision least squares in the protected
    10D subspace. The null-space randomization is deliberately privileged.
    """
    j64 = j.double()
    v = selected[:, list(KIDX)].double()
    jj = j64 @ j64.transpose(-1, -2)
    pseudo = torch.linalg.pinv(jj, rtol=1e-12)
    pin = j64.transpose(-1, -2) @ pseudo
    target = j64 @ v.unsqueeze(-1)
    minimum = (pin @ target).squeeze(-1)
    noise = torch.randn(v.shape, generator=g, dtype=torch.float64)
    noise = noise-(pin @ (j64 @ noise.unsqueeze(-1))).squeeze(-1)
    missing = (v.square().sum(1)-minimum.square().sum(1)).clamp_min(0).sqrt()
    noise = noise/noise.norm(dim=1).clamp_min(1e-14).unsqueeze(-1)
    sham = minimum+noise*missing.unsqueeze(-1)
    full = torch.zeros_like(selected)
    full[:, list(KIDX)] = sham.float()
    derivative_error = float((j64 @ sham.unsqueeze(-1)-target).abs().max())
    norm_error = float((sham.norm(dim=1)-v.norm(dim=1)).abs().max())
    return full, derivative_error, norm_error


@torch.no_grad()
def score_edit(model, recipient_state, donor_state, delta, eligible, depth):
    if not torch.equal(protected_project(delta), delta):
        raise AssertionError("candidate escaped exact protected affine kernel")
    actual = model.protected(recipient_state+delta)
    original = model.protected(recipient_state)
    protected_error = float((actual-original).abs().max())
    if protected_error > 1e-6:
        raise AssertionError(f"full normalized protected readout changed: {protected_error}")
    new = model.no_op(recipient_state+delta, depth)
    donor = model.no_op(donor_state, depth)
    unedited_future = model.no_op(recipient_state, depth)
    mismatch = (model.task(new)-model.task(donor))
    abs_future_public = (
        model.protected(new)-model.protected(unedited_future)).abs()
    n = int(eligible.sum())
    return {
        "eligible": n,
        "donor_mse_eligible": float(mismatch[eligible].square().mean()) if n else None,
        "donor_recovery_eligible": float(
            ((mismatch.abs().amax(1) <= .10)[eligible]).float().mean()) if n else None,
        "donor_mse_all": float(mismatch.square().mean()),
        "donor_recovery_all": float((mismatch.abs().amax(1)<=.10).float().mean()),
        "mean_physical_dose": float(delta.norm(dim=1).mean()),
        "immediate_normalized_readout_max_error": protected_error,
        "future_normalized_readout_mean_abs_change": float(abs_future_public.mean()),
    }


@torch.no_grad()
def reduced_one_step_defect(model, state, basis):
    """Zero-fill, public-anchored projected next-state closure diagnostic."""
    reduced = torch.zeros_like(state)
    reduced[:, [0, 4]] = state[:, [0, 4]]
    reduced = reduced+(state @ basis) @ basis.T
    zero = torch.zeros((len(state), XD))
    original_step = model.step(state, zero)
    reduced_step = model.step(reduced, zero)
    return float(((original_step-reduced_step) @ basis).norm(dim=1).mean())


def assay_seed(seed: int, teacher: RecurrentTeacher, first, second,
               n: int = TEST_SIZE, null_arms: int = 32,
               tangent_arms: int = 16):
    with torch.no_grad():
        physical = first[0] @ first[1]
        second_physical = second[0] @ second[1]
        factors = [float((w-teacher.C).square().mean())
                   for w in (physical, second_physical)]
        shear = torch.eye(D)
        shear[0, 1] = .7
        equivalent = (first[0] @ shear) @ torch.linalg.solve(shear, first[1])
        gauge_error = float((physical-equivalent).abs().max())
    bases, selection = subspaces(teacher, physical, second_physical, seed)
    h0, event, donor_event, goal, donor_goal = matched_source_pairs(seed*741+77, n)
    with torch.no_grad():
        recipient = teacher.step(h0, event)
        donor = teacher.step(h0, donor_event)
        featdelta = teacher.features(h0, donor_event)-teacher.features(h0, event)
        delta = protected_project(featdelta @ physical)
        full_teacher_delta = protected_project(featdelta @ teacher.C)
        exact_dark_delta = protected_project(donor-recipient)
        clean4_r = teacher.task(teacher.no_op(recipient, 4))
        clean4_d = teacher.task(teacher.no_op(donor, 4))
        correct_r = (clean4_r-goal).abs().amax(1) <= .08
        correct_d = (clean4_d-donor_goal).abs().amax(1) <= .08
        donor_change = (clean4_r-clean4_d).abs().amax(1) >= .15
        current_gap = (
            teacher.protected(recipient)-teacher.protected(donor)).abs().amax(1)
        equal_current = current_gap <= .02
        eligible = correct_r & correct_d & donor_change & equal_current
        n_eligible = int(eligible.sum())
        edits = {
            "zero_edit": torch.zeros_like(delta),
            "physical_BPD_full": delta,
            "physical_teacher_full": full_teacher_delta,
            "exact_dark_donor": exact_dark_delta,
        }
        for name, basis in bases.items():
            edits[name] = protected_project((delta @ basis) @ basis.T)
        closings = {name: reduced_one_step_defect(teacher, recipient, basis)
                    for name, basis in bases.items()}
        clean = {}
        for depth in TEST_DEPTHS:
            clean[depth] = {
                "recipient_task_mae_vs_oracle": float(
                    (teacher.task(teacher.no_op(recipient, depth))-goal).abs().mean()),
                "donor_task_mae_vs_oracle": float(
                    (teacher.task(teacher.no_op(donor, depth))-donor_goal).abs().mean()),
            }
        score = {str(depth): {
            name: score_edit(teacher,recipient,donor,patch,eligible,depth)
            for name,patch in edits.items()
        } for depth in TEST_DEPTHS}

    generator = rng(seed+88700)
    random_controls = {}
    for name in ("task_joint", "bpd_weight_svd"):
        dose = edits[name].norm(dim=1,keepdim=True)
        recs, errors = [], []
        for _ in range(null_arms):
            u = torch.randn((n,D),generator=generator)
            u = protected_project(u)
            u = u / u.norm(dim=1,keepdim=True).clamp_min(1e-12)*dose
            metric = score_edit(teacher,recipient,donor,u,eligible,8)
            recs.append(metric["donor_recovery_eligible"] or 0.)
            errors.append(metric["donor_mse_eligible"])
        random_controls[name] = {
            "arms": null_arms, "matched_at_depth": 8,
            "mean_donor_recovery_eligible": sum(recs)/null_arms,
            "maximum_donor_recovery_eligible": max(recs),
            "mean_donor_mse_eligible": sum(x or 0 for x in errors)/null_arms,
        }

    tangent_controls = {}
    for depth in (4,8):
        j = task_jacobian(teacher,recipient,depth)
        recoveries = []
        max_derivative, max_norm = 0.,0.
        for _ in range(tangent_arms):
            null,derivative_error,norm_error = tangent_matched_sham(
                j,edits["task_joint"],generator)
            max_derivative = max(max_derivative,derivative_error)
            max_norm = max(max_norm,norm_error)
            out = score_edit(teacher,recipient,donor,null,eligible,depth)
            recoveries.append(out["donor_recovery_eligible"] or 0.)
        tangent_controls[str(depth)] = {
            "arms": tangent_arms,
            "match": "same full 2-output signed Jacobian and physical L2 norm",
            "mean_recovery_eligible": sum(recoveries)/tangent_arms,
            "best_recovery_eligible": max(recoveries),
            "max_signed_task_derivative_error": max_derivative,
            "max_physical_norm_error": max_norm,
        }
    return {
        "seed":seed,
        "training_family":"independent_scalar_source",
        "evaluation_family":"heldout_anticorrelated_scalar_source",
        "generated_pairs":n,
        "eligibility":{
            "recipient_clean_task_correct":int(correct_r.sum()),
            "donor_clean_task_correct":int(correct_d.sum()),
            "both_clean_task_correct":int((correct_r&correct_d).sum()),
            "clean_future_donor_change":int(donor_change.sum()),
            "clean_current_full_softmax_gap_within_0p02":int(equal_current.sum()),
            "eligible":n_eligible,
            "mean_clean_current_softmax_difference":float(current_gap.mean()),
        },
        "factor_physical_mse_by_restart":factors,
        "factor_complete_GL_gauge_max_error":gauge_error,
        "subspace_selection":selection,
        "reduced_zero_fill_one_step_closure_defects":closings,
        "clean_by_depth":{str(k):v for k,v in clean.items()},
        "causal_by_depth":score,
        "same_dose_random_K_controls_depth8":random_controls,
        "privileged_signed_full_task_jacobian_controls":tangent_controls,
    }


def engineering_and_science_gates(rows):
    engineering = []
    portability = []
    incremental = []
    for r in rows:
        eligible = r["eligibility"]["eligible"]
        c4 = r["clean_by_depth"]["4"]["recipient_task_mae_vs_oracle"]
        c8 = r["clean_by_depth"]["8"]["recipient_task_mae_vs_oracle"]
        factor_ok = max(r["factor_physical_mse_by_restart"]) <= 1e-4
        max_read = max(o["immediate_normalized_readout_max_error"]
                       for d in r["causal_by_depth"].values() for o in d.values())
        max_tan = max(max(c["max_signed_task_derivative_error"],
                          c["max_physical_norm_error"])
                      for c in r["privileged_signed_full_task_jacobian_controls"].values())
        engineering.append(
            factor_ok and c4<=.08 and c8<=.15 and max_read<=1e-6
            and r["factor_complete_GL_gauge_max_error"]<=2e-5 and max_tan<=1e-4
        )
        task = r["causal_by_depth"]["8"]["task_joint"]
        weight = r["causal_by_depth"]["8"]["bpd_weight_svd"]
        random = r["same_dose_random_K_controls_depth8"]["task_joint"]
        signed = r["privileged_signed_full_task_jacobian_controls"]["8"]
        portability.append(
            eligible>=200 and task["donor_mse_eligible"] is not None
            and task["donor_mse_eligible"]<=.8*weight["donor_mse_eligible"]
            and task["donor_recovery_eligible"]>=random["mean_donor_recovery_eligible"]+.10
        )
        incremental.append(
            eligible>=200 and task["donor_recovery_eligible"] >=
            signed["mean_recovery_eligible"]+.05
        )
    return {
        "engineering_roots_passed":sum(engineering),
        "engineering_all_pass":len(rows)==4 and all(engineering),
        "exploratory_unseen_depth_portability_roots":sum(portability),
        "exploratory_unseen_depth_portability_gate_pass":len(rows)==4 and sum(portability)>=3,
        "strong_incremental_vs_privileged_tangent_roots":sum(incremental),
        "strong_incremental_gate_pass":len(rows)==4 and sum(incremental)>=3,
        "independent_scientific_inference":"none; 4 same-source-family roots, uncalibrated",
    }


def run(phase="excluded",factor_steps=FACTOR_STEPS,teacher_steps=TRAIN_STEPS,
        n=TEST_SIZE,null_arms=32,tangent_arms=16):
    if phase not in ("excluded","fresh"):
        raise ValueError("only preregistered excluded/fresh phases")
    seeds = EXCLUDED_ROOTS if phase=="excluded" else FRESH_ROOTS
    torch.set_num_threads(1)
    rows=[]
    for seed in seeds:
        teacher = train_teacher(seed,updates=teacher_steps)
        first = fit_factors(teacher,seed,0,updates=factor_steps)
        second = fit_factors(teacher,seed,1,updates=factor_steps)
        rows.append(assay_seed(seed,teacher,first,second,n,null_arms,tangent_arms))
    result={
        "experiment":"CGC-02 trained task-protected recurrent causal subspaces",
        "phase":phase,"evidence_class":("excluded development" if phase=="excluded"
                                     else "4-root source-locked exploratory"),
        "torch_version":torch.__version__,
        "config":{"teacher_steps":teacher_steps,"factor_steps":factor_steps,
                  "heldout_pairs":n,"rank":RANK,
                  "calibration_examples":CAL_SIZE,
                  "depths":list(TEST_DEPTHS),
                  "random_null_arms":null_arms,
                  "signed_jacobian_null_arms":tangent_arms},
        "roots":list(seeds),"rows":rows,
    }
    if phase=="fresh":
        result["registered_gates"]=engineering_and_science_gates(rows)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("excluded","fresh"),required=True)
    p.add_argument("--teacher-steps",type=int,default=TRAIN_STEPS)
    p.add_argument("--factor-steps",type=int,default=FACTOR_STEPS)
    p.add_argument("--pairs",type=int,default=TEST_SIZE)
    p.add_argument("--random-nulls",type=int,default=32)
    p.add_argument("--tangent-nulls",type=int,default=16)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=run(args.phase,args.factor_steps,args.teacher_steps,args.pairs,
               args.random_nulls,args.tangent_nulls)
    print(json.dumps(result,separators=(",",":")))
    if args.out:
        args.out.write_text(json.dumps(result,indent=2)+"\n")
