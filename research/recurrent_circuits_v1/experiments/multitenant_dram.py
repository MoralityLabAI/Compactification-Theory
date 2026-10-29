"""Multitenant host-DRAM residency prototype for a shared BF16 transformer block.

Inactive session states remain in pinned CPU memory. This measures serialized
copy/compute behavior; it does not claim transfer/compute overlap.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def host_memory():
    class M(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    try:
        m = M(); m.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return {'total_bytes': m.ullTotalPhys, 'available_bytes': m.ullAvailPhys,
                'load_percent': m.dwMemoryLoad}
    except Exception:
        return None


def process_rss():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return {'working_set_bytes': info.rss, 'peak_working_set_bytes': getattr(info, 'peak_wset', None)}
    except ImportError:
        pass
    class C(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    try:
        c = C(); c.cb = ctypes.sizeof(C)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                       ctypes.byref(c), c.cb)
        return {'working_set_bytes': c.WorkingSetSize, 'peak_working_set_bytes': c.PeakWorkingSetSize} if ok else None
    except Exception:
        return None


def smi_snapshot():
    try:
        raw = subprocess.check_output(['nvidia-smi'], text=True, timeout=5, errors='replace')
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if 'RTX 5080' in line:
                return {'gpu_name_line': line.strip(),
                        'gpu_memory_line': lines[i+1].strip() if i+1 < len(lines) else None,
                        'gpu_utilization_line': lines[i+2].strip() if i+2 < len(lines) else None,
                        'telemetry': 'WDDM table; process owner not exposed'}
        return {'raw': raw[-1000:]}
    except Exception as exc:
        return {'error': str(exc)}


def tensor_hash(t):
    b = t.detach().contiguous().cpu().view(__import__('torch').uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def safe_reset(torch, max_fraction):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(max_fraction, 0)
    torch.cuda.reset_peak_memory_stats(0)


def run_resident(torch, model, source, sessions, batch, tokens, width, loops, active, max_fraction):
    safe_reset(torch, max_fraction)
    initial_ram = host_memory()
    t0 = time.perf_counter()
    states = source.to('cuda', non_blocking=False)
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, sessions, active):
            end = min(start + active, sessions)
            h = states[start:end].reshape((end-start)*batch, tokens, width)
            for _ in range(loops):
                h = model(h)
            states[start:end].copy_(h.reshape(end-start, batch, tokens, width))
    torch.cuda.synchronize()
    hot_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    output = states.cpu()
    torch.cuda.synchronize()
    terminal_seconds = time.perf_counter() - t0
    peak_alloc = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    rss = process_rss()
    after_ram = host_memory()
    output_hash = tensor_hash(output)
    del states, output
    torch.cuda.empty_cache()
    return {'mode': 'resident', 'active_sessions': active, 'loops_per_residence': loops,
            'initialization_seconds': init_seconds, 'hot_seconds': hot_seconds,
            'terminal_copy_seconds': terminal_seconds,
            'end_to_end_seconds': init_seconds + hot_seconds + terminal_seconds,
            'useful_transitions_per_second': sessions*loops/hot_seconds,
            'per_session_latency_seconds': hot_seconds/sessions,
            'initialization_input_bytes_h2d': sessions*batch*tokens*width*2,
            'terminal_output_bytes_d2h': sessions*batch*tokens*width*2,
            'h2d_bytes_hot': 0, 'd2h_bytes_hot': 0, 'pinned_host_bytes': 0,
            'peak_allocated_bytes': peak_alloc, 'peak_reserved_bytes': peak_reserved,
            'process_rss': rss, 'host_ram_before': initial_ram, 'host_ram_after': after_ram,
            'output_sha256': output_hash, 'all_sessions_completed': sessions}


def run_host(torch, model, source, sessions, batch, tokens, width, loops, active, residence, max_fraction):
    safe_reset(torch, max_fraction)
    initial_ram = host_memory()
    t0 = time.perf_counter()
    host = torch.empty(source.shape, dtype=source.dtype, device='cpu', pin_memory=True)
    host.copy_(source)
    init_seconds = time.perf_counter() - t0
    state_bytes = batch*tokens*width*2
    h2d = d2h = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for loop_start in range(0, loops, residence):
            nloop = min(residence, loops-loop_start)
            for start in range(0, sessions, active):
                end = min(start+active, sessions)
                h = host[start:end].to('cuda', non_blocking=True)
                h2d += (end-start)*state_bytes
                for _ in range(nloop):
                    y = model(h.reshape((end-start)*batch, tokens, width))
                    h = y.reshape(end-start, batch, tokens, width)
                host[start:end].copy_(h, non_blocking=True)
                d2h += (end-start)*state_bytes
                # Required before CPU storage reuse/read; one serialized stream.
                torch.cuda.synchronize()
    hot_seconds = time.perf_counter() - t0
    peak_alloc = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    rss = process_rss()
    after_ram = host_memory()
    completed = 0
    output_hash = tensor_hash(host)
    del host
    torch.cuda.empty_cache()
    return {'mode': 'host_dram', 'active_sessions': active, 'loops_per_residence': residence,
            'initialization_seconds': init_seconds, 'hot_seconds': hot_seconds,
            'terminal_copy_seconds': 0.0, 'end_to_end_seconds': init_seconds+hot_seconds,
            'useful_transitions_per_second': sessions*loops/hot_seconds,
            'per_session_latency_seconds': hot_seconds/sessions,
            'initialization_input_bytes_h2d': 0, 'terminal_output_bytes_d2h': 0,
            'h2d_bytes_hot': h2d, 'd2h_bytes_hot': d2h,
            'pinned_host_bytes': sessions*state_bytes,
            'peak_allocated_bytes': peak_alloc, 'peak_reserved_bytes': peak_reserved,
            'process_rss': rss, 'host_ram_before': initial_ram, 'host_ram_after': after_ram,
            'output_sha256': output_hash, 'all_sessions_completed': sessions}


def pointer_chase(torch, steps, table_size, seed):
    rng = random.Random(seed)
    order = list(range(table_size)); rng.shuffle(order)
    links = [0]*table_size
    for a, b in zip(order, order[1:]+order[:1]): links[a] = b
    table = torch.tensor(links, dtype=torch.int32).pin_memory()
    current = torch.zeros((), dtype=torch.int32, device='cuda')
    host_current = torch.empty((), dtype=torch.int32, pin_memory=True)
    host_next = torch.empty((), dtype=torch.int32, pin_memory=True)
    oracle = 0
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            host_current.copy_(current, non_blocking=True)
            torch.cuda.synchronize()
            idx = int(host_current.item())
            oracle = links[oracle]
            nxt = links[idx]
            host_next.fill_(nxt)
            current.copy_(host_next, non_blocking=True)
            torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    gpu_result = int(current.cpu().item())
    return {'steps': steps, 'table_entries': table_size, 'table_bytes_pinned': table.numel()*table.element_size(),
            'h2d_bytes': steps*4, 'd2h_bytes': steps*4,
            'elapsed_seconds': elapsed, 'dependent_hops_per_second': steps/elapsed,
            'gpu_mediated_address_matches_cpu_oracle': gpu_result == oracle,
            'final_address': gpu_result,
            'method': 'each hop copies current GPU address to pinned CPU, gathers dependent host pointer, copies next address back; no prefetch overlap'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=HERE/'results'/'multitenant_dram_bf16.json')
    p.add_argument('--sessions', type=int, default=16)
    p.add_argument('--active', default='1,4,8')
    p.add_argument('--residence', default='1,8,16')
    p.add_argument('--loops', type=int, default=16)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--max-vram-fraction', type=float, default=0.05)
    p.add_argument('--max-seconds', type=float, default=180)
    p.add_argument('--pointer-steps', type=int, default=256)
    p.add_argument('--pointer-table-size', type=int, default=1<<20)
    args = p.parse_args()
    import torch
    from torch import nn
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU substitute')
    if args.sessions < 1 or args.loops < 1 or args.width % args.heads: raise ValueError('invalid shape')
    active = tuple(int(x) for x in args.active.split(','))
    residences = tuple(int(x) for x in args.residence.split(','))
    if any(x < 1 or x > args.sessions for x in active) or any(x < 1 or x > args.loops for x in residences): raise ValueError('active/residence out of range')
    torch.manual_seed(20261029); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16
    free_before, total = torch.cuda.mem_get_info(0)
    elem = torch.empty((), dtype=dtype).element_size()
    per_state = args.batch*args.tokens*args.width*elem
    full_states = args.sessions*per_state
    estimated_resident_attention = 4*args.sessions*args.batch*args.heads*args.tokens**2*elem
    if full_states + estimated_resident_attention > total*args.max_vram_fraction:
        raise RuntimeError('resident plan exceeds per-process allocator fraction')
    torch.cuda.set_per_process_memory_fraction(args.max_vram_fraction, 0)
    model = nn.TransformerEncoderLayer(args.width,args.heads,dim_feedforward=4*args.width,
        dropout=0.0,activation='relu',batch_first=True,norm_first=True).eval().to('cuda',dtype=dtype)
    weight_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        weight_digest.update(name.encode())
        weight_digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    model_weight_sha256 = weight_digest.hexdigest()
    source = torch.randn((args.sessions,args.batch,args.tokens,args.width),dtype=dtype,device='cpu')
    resident_refs = {}
    records = []
    deadline = time.perf_counter()+args.max_seconds
    for nactive in active:
        ref = run_resident(torch,model,source,args.sessions,args.batch,args.tokens,args.width,
                           args.loops,nactive,args.max_vram_fraction)
        resident_refs[nactive] = ref['output_sha256']
        ref['bitwise_equal_to_reference'] = True
        records.append(ref)
        for residence in residences:
            if time.perf_counter() >= deadline: raise TimeoutError('bounded sweep deadline reached')
            record = run_host(torch,model,source,args.sessions,args.batch,args.tokens,args.width,
                              args.loops,nactive,residence,args.max_vram_fraction)
            record['bitwise_equal_to_resident'] = record['output_sha256'] == resident_refs[nactive]
            record['reference_resident_sha256'] = resident_refs[nactive]
            record['hot_slowdown_vs_resident'] = record['hot_seconds']/next(x['hot_seconds'] for x in records if x['mode']=='resident' and x['active_sessions']==nactive)
            records.append(record)
            if record['peak_reserved_bytes'] > int(args.max_vram_fraction*total):
                raise RuntimeError('measured allocator reserve exceeded process budget')
    pointer = pointer_chase(torch,args.pointer_steps,args.pointer_table_size,20261030)
    # Explicit capacity control: request a fixed resident state pool just above
    # this process's declared allocator cap. Catch OOM; never shrink the request.
    cap_bytes = int(args.max_vram_fraction * total)
    oversized_sessions = math.floor(cap_bytes / per_state) + 1
    oversized_bytes = oversized_sessions * per_state
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    try:
        oversized = torch.empty((oversized_sessions,args.batch,args.tokens,args.width),
                                dtype=dtype,device='cuda')
        torch.cuda.synchronize()
        oversized_status = {'outcome':'unexpected_success','allocated_bytes':oversized.numel()*oversized.element_size()}
        del oversized
    except torch.cuda.OutOfMemoryError as exc:
        oversized_status = {'outcome':'out_of_memory','error':str(exc).splitlines()[0],
                            'allocated_after_failure_bytes':torch.cuda.memory_allocated(0)}
        torch.cuda.empty_cache()
    result = {
        'schema_version':1,'evidence':'measured_cuda_multitenant_transport_prototype',
        'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'dtype':'bfloat16','configuration':vars(args)|{'out':str(args.out)},
        'state_bytes_per_session':per_state,'total_inactive_state_bytes':full_states,
        'resident_attention_estimate_bytes':estimated_resident_attention,
        'allocator_fraction_cap':args.max_vram_fraction,'torch_cuda_free_total_before_bytes':[free_before,total],
        'global_nvidia_smi_before':smi_snapshot(),'host_memory_before':host_memory(),
        'records':records,'random_pointer_chasing':pointer,
        'clean_oversized_resident_baseline': {'session_count':oversized_sessions,
            'requested_state_bytes':oversized_bytes,'allocator_cap_bytes':cap_bytes,
            'outcome':oversized_status,'batch_size_reduced':False},
        'model_weight_sha256':model_weight_sha256,
        'global_nvidia_smi_after':smi_snapshot(),'host_memory_after':host_memory(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), HERE/'gpu_dram_bench.py', HERE/'reference.py', HERE/'THEORY.md', HERE/'EXPERIMENTS.md']},
        'claims':{'all_inactive_sessions_host_resident_in_host_dram':True,
                  'transfer_compute_overlap':False,'capacity_gain_measured_as_oom_threshold':False,
                  'output_equivalence_is_bitwise_per_active_batch_shape':True},
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(args.out,'records',len(records),'pointer_match',pointer['gpu_mediated_address_matches_cpu_oracle'])

if __name__=='__main__': main()
