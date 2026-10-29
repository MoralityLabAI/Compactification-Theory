"""Build a deterministic index receipt for the saved RC-01 CUDA run JSON files."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent
RESULTS=HERE/'results'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    files=sorted([*RESULTS.glob('gpu_nand_*.json'),*RESULTS.glob('gpu_transformer_*.json')])
    runs=[]
    for path in files:
        p=json.loads(path.read_text(encoding='utf-8'))
        c=p['configuration']
        runs.append({'file':path.name,'result_json_sha256':sha(path),
          'kernel':c['kernel'],'dtype':c['dtype'],'mode':c['mode'],'chunk':c['chunk'],
          'loops':c['loops'],'batch':c['batch'],'tokens':c['tokens'],'width':c['width'],
          'repeats':c['repeats'],'warmup':c['warmup'],'output_sha256':p['output_sha256'],
          'bitwise_equal_to_resident':p['bitwise_equal_to_resident'],
          'boolean_oracle_checked':p['boolean_oracle_checked'],
          'median_wall_seconds':p['median_wall_seconds'],'wall_seconds':p['wall_seconds'],
          'loop_steps_per_second':p['loop_steps_per_second'],
          'peak_allocated_bytes':p['gpu_peak_allocated_bytes'],
          'peak_reserved_bytes':p['gpu_peak_reserved_bytes'],
          'pinned_host_bytes':p['pinned_host_bytes'],
          'h2d_bytes_per_repeat':p['h2d_bytes_per_measured_repeat'],
          'd2h_bytes_per_repeat':p['d2h_bytes_per_measured_repeat'],
          'host_rss_after_run_bytes':p['host_rss_after_run_bytes']})
    multi=RESULTS/'multitenant_dram_bf16.json'
    hashes={p.name:sha(p) for p in [Path(__file__),HERE/'experiments'/'multitenant_dram.py',HERE/'gpu_dram_bench.py',HERE/'reference.py',HERE/'THEORY.md',HERE/'EXPERIMENTS.md',HERE/'results'/'reference_v1.json']}
    out={'schema_version':1,'evidence':'index_of_saved_cuda_measurements',
         'device':'NVIDIA GeForce RTX 5080 Laptop GPU','torch':'2.11.0+cu128','cuda_runtime':'12.8',
         'baseline_reference_commit':'8a79b9668d2fc4f2e2028b8914379aded33285f7',
         'runs':runs,'multitenant_file':multi.name,'multitenant_json_sha256':sha(multi),
         'multitenant_source_sha256':json.loads(multi.read_text(encoding='utf-8'))['source_sha256'],
         'source_sha256':hashes}
    target=RESULTS/'gpu_matrix_summary.json'
    target.write_bytes((json.dumps(out,indent=2,sort_keys=True)+'\n').encode('utf-8'))
    print(target,'runs',len(runs))
if __name__=='__main__': main()
