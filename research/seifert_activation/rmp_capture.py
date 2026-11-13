"""SCM-02 exploratory RMP cyclic-intervention capture (without retraining RMP).

Run this script in a checkout containing MoralityLabAI/LoopedTransformers and its
checkpoint/dictionary artifacts. This file is isolated from RMP's sealed splits.
CPU fake-model tests in tests/test_rmp_capture.py validate the hook contract;
those fixtures are not observations of trained models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

from knot import InvalidBraidTrace, extract_closed_braid, normalized_alexander, braid_seifert_surface
from winding import verify_pure_braid
from nonabelian import ArtinWordBudgetExceeded, artin_witness
from milnor import token_order_milnor

SCHEMA = "scm04/rmp-cyclic/v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    # Stable little-endian float64 bytes without NumPy, which some clean CPU
    # PyTorch environments do not install by default. Hashes must replay even
    # when only torch and the Python standard library are available.
    values = tensor.detach().cpu().to(torch.float64).contiguous().reshape(-1).tolist()
    raw = struct.pack(f'<{len(values)}d', *values)
    return hashlib.sha256(raw).hexdigest()


def chart_from_decoder(decoder: torch.Tensor, features: tuple[int,int]) -> torch.Tensor:
    """Freeze a 2D linear readout from two registered D-shared decoder rows.

    Decoder rows have input dimension H and represent directions in residual
    space. We do not use sparse *codes* as chart points: the nonlinear TopK
    switching would make chart continuity analysis harder.
    """
    if decoder.ndim != 2 or len(features) != 2 or features[0] == features[1]:
        raise ValueError('Need two distinct features and [width, H] decoder')
    if any(not 0 <= index < decoder.shape[0] for index in features):
        raise ValueError('Decoder feature index outside dictionary')
    chart = decoder[list(features)].detach().float().T.contiguous()
    chart /= chart.norm(dim=0,keepdim=True).clamp_min(1e-12)
    gram = chart.T @ chart
    if float(torch.linalg.det(gram)) < 1e-6:
        raise ValueError('Chart feature directions are linearly dependent')
    if not torch.isfinite(chart).all():
        raise ValueError('Non-finite feature chart')
    return chart


@torch.no_grad()
def capture_cycle(model: Any, tokens: torch.Tensor, chart: torch.Tensor, *,
                  inject_token: int, observed_tokens: Sequence[int],
                  radius: float, angles: int = 32,
                  inject_after_visit: int = 0,
                  intervention_axes: torch.Tensor | None = None) -> dict:
    """Run one RMP AnalysisModel through a closed 2D perturbation circle.

    `model.run(tokens, hook)` must match the RMP AnalysisModel contract:
    per-visit hook VisitContext{visit,after}; output.visit_outputs after hooks.
    Both endpoint evaluations have identical floating point inputs, allowing
    deterministic repeatability to be checked without sin(2*pi) roundoff.
    Radius is measured in the unpatched residual's per-token RMS units.
    """
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError('One fixed token sequence at a time, shape [1, sequence]')
    if chart.ndim != 2 or chart.shape[1] != 2:
        raise ValueError('Chart must have shape [hidden,2]')
    if angles < 8 or not (math.isfinite(radius) and radius > 0):
        raise ValueError('At least eight angles and a positive finite radius required')
    if not (0 <= inject_token < tokens.shape[1]) or inject_after_visit < 0:
        raise ValueError('Invalid injection site')
    observed = tuple(int(x) for x in observed_tokens)
    if len(observed) < 2 or len(set(observed)) != len(observed):
        raise ValueError('Two or more unique observed positions required')
    if any(x < 0 or x >= tokens.shape[1] for x in observed):
        raise ValueError('Observed token outside sequence')
    if inject_token in observed:
        raise ValueError('Observed tokens must exclude the directly injected token')
    device = tokens.device
    chart = chart.to(device=device, dtype=torch.float32)
    axes = (chart if intervention_axes is None else intervention_axes).to(device=device, dtype=torch.float32)
    if axes.shape != chart.shape or not torch.isfinite(axes).all():
        raise ValueError('Intervention axes must be finite and match chart [hidden,2]')
    u, v = axes[:,0], axes[:,1]
    visits: list[list[list[list[float]]]] | None = None
    predictions: list[int] = []
    peak_delta = 0.0
    baseline = model.run(tokens)
    baseline_logits = baseline.logits.detach().float()
    if baseline_logits.shape[0] != 1:
        raise ValueError('Unexpected model logits batch dimension')
    for k in range(angles + 1):
        theta = 2 * math.pi * k / angles
        ca, sa = (1.0, 0.0) if k == angles else (math.cos(theta), math.sin(theta))
        modified = [False]
        def hook(context):
            if context.visit != inject_after_visit:
                return None
            after = context.after
            if after.shape[0] != 1 or after.shape[2] != chart.shape[0]:
                raise ValueError('Unexpected residual shape for injected chart')
            # Clone to prevent modifying the clean output or model internals in-place.
            changed = after.clone()
            scale = after[0, inject_token].float().pow(2).mean().sqrt().clamp_min(1e-6)
            direction = ca * u + sa * v
            changed[0, inject_token] = after[0, inject_token] + (radius * scale * direction).to(after.dtype)
            modified[0] = True
            return changed
        result = model.run(tokens, hook)
        if not modified[0]:
            raise ValueError('Injection visit was never reached')
        if len(result.visit_outputs) != len(baseline.visit_outputs):
            raise ValueError('Inconsistent number of visits')
        if visits is None:
            visits = [[] for _ in result.visit_outputs]
        for visit, state in enumerate(result.visit_outputs):
            frame = (state[0, list(observed)].float() @ chart).cpu().tolist()
            visits[visit].append(frame)
        predictions.append(int(result.logits[0,-1].argmax().item()))
        peak_delta = max(peak_delta, float((result.logits.float()-baseline_logits).abs().max()))
    assert visits is not None
    # Require each named strand's path to close for every visit; this is a pure braid.
    for visit, frames in enumerate(visits):
        for strand in range(len(observed)):
            d = math.dist(frames[0][strand], frames[-1][strand])
            if d > 1e-5:
                raise InvalidBraidTrace(f'Visit {visit}, strand {strand} fails cyclic endpoint equality: {d}')
    if predictions[0] != predictions[-1]:
        raise InvalidBraidTrace('Output prediction differs at identical endpoint intervention')
    return {
        'angles': [2*math.pi*k/angles for k in range(angles+1)],
        'frames_by_visit': visits,
        'observed_token_positions': list(observed),
        'injected_position': inject_token,
        'injected_after_visit': inject_after_visit,
        'radius_in_token_rms': radius,
        'chart_sha256': sha256_tensor(chart),
        'intervention_axes_sha256': sha256_tensor(axes),
        'baseline_prediction': int(baseline_logits[0,-1].argmax().item()),
        'predictions': predictions,
        'changed_prediction_fraction': sum(p != predictions[0] for p in predictions[:-1])/angles,
        'max_abs_logit_change_from_clean': peak_delta,
    }


def analyze_cycle(trace: dict, *, clearance: float = 1e-7,
                  max_artin_letters: int = 20_000) -> list[dict]:
    """Conservative analyzer: rejected observations remain explicit records."""
    result=[]
    for visit, frames in enumerate(trace['frames_by_visit']):
        try:
            braid=extract_closed_braid(frames,clearance=clearance)
            linking=verify_pure_braid(frames,braid,clearance=clearance)
            surface=braid_seifert_surface(braid.n_strands,braid.word)
            item={
                'visit':visit+1,'valid':True,'braid_word':list(braid.word),
                'pair_winding':{f'{i},{j}':int(value) for (i,j),value in linking.items()},
                'boundary_components':surface.boundary_components,
                'canonical_seifert_genus':surface.total_genus,
                'alexander':str(normalized_alexander(braid.n_strands,braid.word)),
                'n_crossings':len(braid.word),
                'sampling_certificate':'polygonal_only',
            }
            try:
                witness=artin_witness(braid.n_strands, braid.word,
                                      max_letters=max_artin_letters)
                item['artin']={
                    'status':'exact',
                    'nonidentity':not witness['is_identity'],
                    'sha256':witness['sha256'],
                    'image_lengths':witness['image_lengths'],
                    'scope':witness['scope'],
                }
                # Small witnesses support collision-free exact replay comparisons;
                # long witnesses keep a hash for receipt integrity only.
                if sum(witness['image_lengths'])<=1_024:
                    item['artin']['images']=witness['images']
            except ArtinWordBudgetExceeded as exc:
                item['artin']={'status':'budget_exceeded', 'reason':str(exc)}
            if braid.n_strands == 3:
                try:
                    item['milnor'] = token_order_milnor(
                        frames, braid.word, max_letters=max_artin_letters)
                except ArtinWordBudgetExceeded as exc:
                    item['milnor'] = {'status': 'budget_exceeded', 'reason': str(exc)}
            else:
                item['milnor'] = {'status': 'not_applicable', 'reason': 'three-strand invariant'}
            result.append(item)
        except (InvalidBraidTrace, ValueError, ArithmeticError) as exc:
            result.append({'visit':visit+1,'valid':False,'reject_reason':str(exc)})
    return result


def compare_double_resolution(coarse: dict, fine: dict, *,
                              max_artin_letters: int = 20_000) -> list[dict]:
    """Compare invariant values by visit. Braid words may differ by Artin relations."""
    left,right=(analyze_cycle(coarse,max_artin_letters=max_artin_letters),
                analyze_cycle(fine,max_artin_letters=max_artin_letters))
    if len(left)!=len(right):
        raise ValueError('Different number of visits')
    report=[]
    for a,b in zip(left,right):
        abelian_match=bool(a['valid'] and b['valid'] and all(a[k]==b[k] for k in
                 ('pair_winding','alexander','boundary_components')))
        aw,bw=a.get('artin',{}),b.get('artin',{})
        artin_available=(aw.get('status')=='exact' and bw.get('status')=='exact')
        exact_images_available=artin_available and 'images' in aw and 'images' in bw
        if exact_images_available:
            nonabelian_match=aw['images']==bw['images']
            comparison='exact_images'
        elif artin_available:
            nonabelian_match=aw['sha256']==bw['sha256']
            comparison='sha256_receipt_only'
        else:
            nonabelian_match=False
            comparison='unavailable'
        am, bm = a.get('milnor', {}), b.get('milnor', {})
        milnor_available = (am.get('status') in ('exact_integer', 'modulo_indeterminacy')
                            and bm.get('status') in ('exact_integer', 'modulo_indeterminacy'))
        milnor_match = (am['mu123_token_order'] == bm['mu123_token_order']
                        and am['indeterminacy_gcd'] == bm['indeterminacy_gcd']) if milnor_available else None
        closure_invariants_match = bool(abelian_match and (milnor_match if milnor_available else
                          am.get('status') == bm.get('status') == 'not_applicable'))
        report.append({'visit':a['visit'],'match':bool(abelian_match and nonabelian_match),
                       'milnor_match': milnor_match,
                       'closure_invariants_match': closure_invariants_match,
                       'closure_scope': 'equality_of_tested_invariants_not_link_isotopy', 
                       'abelian_match':abelian_match,
                       'nonabelian_match':bool(nonabelian_match) if artin_available else None,
                       'nonabelian_comparison':comparison,
                       'coarse_valid':a['valid'],'fine_valid':b['valid']})
    return report


def _calibration_rows(data: dict, *, size: int, seed: int) -> list[int]:
    """Balanced, deterministic examples from RMP's calibration SELECTION half.

    Never opens the reporting half or sealed test. De-duplicates by group ID.
    """
    from research_gym.interp.phase2 import CORE, selection_half
    mask=selection_half(data['groups'])
    families=list(CORE)
    per, extra=divmod(size,len(families))
    selected=[]
    for fam_idx,family in enumerate(families):
        candidates=[]
        visited=set()
        for idx in range(len(data['ids'])):
            if not mask[idx] or data['families'][idx]!=family:
                continue
            group=data['groups'][idx]
            if group in visited:
                continue
            visited.add(group)
            digest=hashlib.sha256(f'{seed}/{family}/{group}/{data["ids"][idx]}'.encode()).hexdigest()
            candidates.append((digest,idx))
        quota=per+int(fam_idx<extra)
        if len(candidates)<quota:
            raise ValueError(f'Only {len(candidates)} calibration groups in {family}; need {quota}')
        selected.extend(idx for _,idx in sorted(candidates)[:quota])
    return selected


def main(argv: Sequence[str] | None = None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rmp-root',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--monitor-bundle',type=Path,required=True)
    parser.add_argument('--dictionary',type=Path,required=True)
    parser.add_argument('--model',default='LOOP-T s1')
    parser.add_argument('--features',type=int,nargs=2,required=True)
    parser.add_argument('--chart-control',choices=['sae','random-orthogonal'],default='sae')
    parser.add_argument('--inject-token',type=int,default=4)
    parser.add_argument('--observed-tokens',type=int,nargs='+',default=[16,32,48])
    parser.add_argument('--radius',type=float,default=0.05)
    parser.add_argument('--angles',type=int,default=32)
    parser.add_argument('--double-sampling',action='store_true')
    parser.add_argument('--examples',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20261113)
    parser.add_argument('--clearance',type=float,default=1e-7)
    parser.add_argument('--max-artin-letters',type=int,default=20_000)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.examples<4 or args.angles<8:
        parser.error('At least four examples and eight angles required')
    if args.out.exists():
        parser.error('Refusing to overwrite an existing receipt; choose a new run ID')
    if not args.rmp_root.is_dir():
        parser.error('RMP checkout unavailable: --rmp-root must point to LoopedTransformers')
    # Import code without mutating the original RMP program or its sealed outputs.
    sys.path.insert(0,str(args.rmp_root.resolve()))
    from research_gym.interp.phase2 import MODELS, core_examples, load_model
    from research_gym.interp.dictionaries import load_dictionary
    if args.model not in MODELS:
        parser.error(f'Unknown RMP model; supported {sorted(MODELS)}')
    cell,label,_=MODELS[args.model]
    checkpoint=args.artifacts/cell/'checkpoints'/f'{label}.pt'
    for required in (checkpoint,args.dictionary):
        if not required.is_file():
            parser.error(f'Required immutable artifact not found: {required}')
    device=torch.device(args.device)
    model=load_model(args.model,args.runs,args.artifacts,device)
    dictionary,summary=load_dictionary(args.dictionary,device=device)
    if summary.get('config',{}).get('kind')!='shared':
        parser.error('SCM-02 requires a registered D-shared dictionary for its chart')
    injection_axes=chart_from_decoder(dictionary.decoder,tuple(args.features))
    if args.chart_control=='sae':
        chart=injection_axes
    else:
        generator=torch.Generator(device='cpu').manual_seed(args.seed+91247)
        random_tensor=torch.randn(injection_axes.shape,generator=generator)
        chart=torch.linalg.qr(random_tensor,mode='reduced')[0].to(device)
    if chart.shape[0]!=model.model.hidden_size:
        parser.error('Dictionary hidden dimension mismatches model')
    data=core_examples(args.monitor_bundle,'monitor_calibration')
    indices=_calibration_rows(data,size=args.examples,seed=args.seed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    digests={
        'checkpoint_sha256':sha256_file(checkpoint),
        'dictionary_sha256':sha256_file(args.dictionary),
        'rmp_root':str(args.rmp_root.resolve()),
        'chart_sha256':sha256_tensor(chart),
        'intervention_axes_sha256':sha256_tensor(injection_axes),
        'chart_control':args.chart_control,
    }
    with args.out.open('x',encoding='utf-8') as writer:
        for idx in indices:
            tokens=data['tokens'][idx:idx+1].to(device)
            trace=capture_cycle(model,tokens,chart,inject_token=args.inject_token,
                  observed_tokens=args.observed_tokens,radius=args.radius,angles=args.angles,
                  intervention_axes=injection_axes)
            record={
                'schema':SCHEMA,'status':'exploratory_unsealed',
                'model':args.model,'seed':args.seed,'source_type':'hash_verified_RMP_checkpoint',
                'task_id_sha256':hashlib.sha256(str(data['ids'][idx]).encode()).hexdigest(),
                'group_id_sha256':hashlib.sha256(str(data['groups'][idx]).encode()).hexdigest(),
                'family':str(data['families'][idx]),'target':int(data['targets'][idx]),
                'token_sequence_sha256':sha256_tensor(tokens),
                'features':list(args.features),**digests,
                'capture':trace,'topology':analyze_cycle(trace,clearance=args.clearance,
                                                   max_artin_letters=args.max_artin_letters),
            }
            if args.double_sampling:
                fine=capture_cycle(model,tokens,chart,inject_token=args.inject_token,
                      observed_tokens=args.observed_tokens,radius=args.radius,angles=2*args.angles,
                      intervention_axes=injection_axes)
                record['double_resolution']=compare_double_resolution(trace,fine,
                                              max_artin_letters=args.max_artin_letters)
            writer.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
            writer.flush()
    print(json.dumps({'schema':SCHEMA,'out':str(args.out),'examples':len(indices),
                      'checkpoint_sha256':digests['checkpoint_sha256'],
                      'chart_sha256':digests['chart_sha256']},indent=2))

if __name__=='__main__':
    main()
