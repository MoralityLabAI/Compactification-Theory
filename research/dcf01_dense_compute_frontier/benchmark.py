"""DCF-01: matched low-rank recurrence; CPU pilot, not an omega test."""
import argparse, json, os, platform, statistics, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np


def run(seed, d, b, loops, reps):
    rng = np.random.default_rng(seed)
    r = d // 16
    u = (rng.standard_normal((d, r)) / np.sqrt(d)).astype(np.float32)
    v = (rng.standard_normal((r, d)) / np.sqrt(d)).astype(np.float32)
    w = u @ v
    x = rng.standard_normal((b, d)).astype(np.float32)
    def dense():
        z = x
        for _ in range(loops):
            z = np.tanh(z @ w)
        return z
    def factored():
        z = x
        for _ in range(loops):
            z = np.tanh((z @ u) @ v)
        return z
    a, f = dense(), factored()
    error = float(np.linalg.norm(a-f) / max(float(np.linalg.norm(a)), 1e-12))
    assert error < 1e-4, (seed,d,b,loops,error)
    # Equal-spectrum identity has best rank-r relative Frobenius error sqrt(1-r/d).
    # It is a negative control for generic low-rank compression, not a timing arm.
    floor = float(np.sqrt(1-r/d))
    for _ in range(3): dense(); factored()
    times = [[], []]
    fs = [dense, factored]
    for i in range(reps):
        for j in (i % 2, 1-i % 2):
            start = time.perf_counter_ns(); fs[j](); times[j].append(time.perf_counter_ns()-start)
    a_ns, f_ns = map(statistics.median,times)
    return dict(seed=seed,d=d,batch=b,loops=loops,rank=r,relative_l2_error=error,
                full_rank_control_error_floor=floor,dense_ns=a_ns,factored_ns=f_ns,
                speedup=a_ns/f_ns,weight_bytes_dense=w.nbytes,weight_bytes_factored=u.nbytes+v.nbytes,
                classical_mac_ratio=d/(2*r))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--reps',type=int,default=21); args=p.parse_args()
    assert args.reps >= 3
    rows=[run(s,d,b,l,args.reps) for s in (11,23,47) for d in (256,512,1024) for b in (1,16,128) for l in (1,4)]
    print(json.dumps(dict(environment=dict(python=platform.python_version(),numpy=np.__version__,machine=platform.machine(),processor=platform.processor(),threads=1,reps=args.reps),rows=rows),indent=2))

if __name__ == '__main__': main()
